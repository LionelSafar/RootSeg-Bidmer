"""
Contains all NN modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys

from torchvision.models import swin_t, Swin_T_Weights, swin_b, Swin_B_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Dict, Union, Optional
from einops import rearrange


class ConvBlock2D(nn.Module):
    """
    adjustable 2D ConvBlock module 

    2DConv -> SiLu -> GroupNorm -> 2DConv -> SiLu -> GroupNorm
    """
    def __init__(self, out_channels, first: int = None, padding="valid"):
        super().__init__()
        if first:
            self.conv1 = nn.Conv2d(first, out_channels, kernel_size=3, padding=padding)
        else:
            self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.gn1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.gn2 = nn.GroupNorm(32, out_channels)

    def forward(self, x):
        x = self.gn1(F.silu(self.conv1(x)))
        x = self.gn2(F.silu(self.conv2(x)))
        return x

class TransformerBlock(nn.Module):
    """
    Transformerblock module with multiheaded self-attention.

    """
    def __init__(self, dim, num_heads=8, dropout=0.0, mult_factor: int=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            batch_first=True,
            device="cpu"
        )
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mult_factor*dim),
            nn.SiLU(),
            nn.Linear(mult_factor*dim, dim),
        )
    
    def forward(self, x):
        # Adjust format for torch's multihead attention module and use self attention with residual
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # (B, H*W, C) = (B, L, C)
        residual = x  
        x, _ = self.attn(x, x, x)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = residual + x   

        # reshape to original form
        x = x.transpose(1, 2).reshape(B, C, H, W)  
        return x


def center_crop(x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """Center crop tensor (N, C, H, W) to target spatial size (N, C, H_new, W_new)."""
    _, _, h, w = x.shape
    th, tw = target_size
    i = (h - th) // 2
    j = (w - tw) // 2
    return x[:, :, i:i+th, j:j+tw]


def patch_expansion(x):
    """
    Patch expansion module for ViT-based Encoder structures.

    takes from (N, H, W, C) --> (N, 2*H, 2*W, C/4) --> (N, C/4, 2*H, 2*W) patch expansion
    """
    C = x.shape[-1]
    x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
    return x.permute(0, 3, 1, 2)


class UNet(nn.Module):
    """
    Vanilla U-Net with transformerblock at bottleneck.
    Uses Groupnorm and Swish activation function. No padding used for Convolutional layers.
    
    """
    def __init__(self, initial_hidden_channels, out_channels, depth):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder
        for i in range(depth):
            hidden = initial_hidden_channels * (2 ** i)
            self.encoders.append(ConvBlock2D(hidden, first=3 if i==0 else hidden // 2))

        # Bottleneck
        self.bottleneck = ConvBlock2D(initial_hidden_channels * (2 ** depth), 
            first=initial_hidden_channels * (2 ** (depth - 1))
        )
        self.transformerblock = TransformerBlock(initial_hidden_channels * (2 ** depth))

        # Decoder
        for i in range(depth):
            hidden = initial_hidden_channels * (2 ** (depth - i - 1))
            self.decoders.append(
                nn.ModuleDict({
                    "upconv": nn.ConvTranspose2d(hidden*2, hidden, kernel_size=2, stride=2),
                    "gn": nn.GroupNorm(32, hidden),
                    "convblock": ConvBlock2D(hidden, first=2*hidden)
                })
            )
        self.final_conv = nn.Conv2d(initial_hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skips = [] # gather skip connections
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)
        x = self.transformerblock(x)

        for dec in self.decoders:
            x = dec["upconv"](x)
            x = dec["gn"](x)
            skip = skips.pop()
            skip = center_crop(skip, x.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = dec["convblock"](x)

        return self.final_conv(x)


class ConvBlock_eff(nn.Module):
    """
    Convblock2D for backboned models.
    Uses Batchnorm instead of Groupnorm due to variable sizes in output channels at intermediate stages
    
    """
    def __init__(self, in_ch, out_ch, padding: int=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True), 
        )
    def forward(self, x):
        return self.conv(x)


class SwinT_UNet(nn.Module):
    """
    U-Net with Swin-T backbone.
    Patch expansion is used after each encoder output to semantically match patch <-> pixel processing
    """
    def __init__(self, out_channels=1, pretrained: bool=True, weight_path: str=None):
        super().__init__()
        if pretrained == True:
            try:
                weights = Swin_T_Weights.IMAGENET1K_V1
                base = swin_t(weights=weights)
            except Exception as e:
                print("Could not load pretrained weights remotely, try accessing local file.")
                if weight_path is not None:
                    weights_path = weight_path
                else:
                    print("Please provide a local filepath for the backbone weights")
                    sys.exit()
                base = swin_t(weights=None)
                state_dict = torch.load(weights_path, map_location="cpu")
                base.load_state_dict(state_dict)
        else: 
            base = swin_t(weights=None)

        return_nodes = {
            "features.1": "skip_1_4",      # 1/4 res, 96C
            "features.3": "skip_1_8",      # 1/8 res, 192C
            "features.5": "skip_1_16",     # 1/16 res, 384C
            "features.7": "bottleneck",    # 1/32 res, 768C
        }

        # Create feature extractor backbone
        self.encoder = create_feature_extractor(
            base, 
            return_nodes=return_nodes
        )

        # Channel dimensions for Swin-T extracted blocks
        self.channels = [12, 24, 48, 96, 192]

        # Decoderblocks
        self.decoders = nn.ModuleDict({
            "up4": self._decoder_block(self.channels[4], self.channels[3]),
            "up3": self._decoder_block(self.channels[3], self.channels[2]),
            "up2": self._decoder_block(self.channels[2], self.channels[1]),
            "up1": self._decoder_block(self.channels[1], self.channels[0]),
        })

        self.topconv = nn.Conv2d(3, 12, 1)
        self.bottleneck_conv = ConvBlock_eff(self.channels[4], self.channels[4])
        self.final_conv = nn.Conv2d(self.channels[0], out_channels, 1)

        # Init normalisation to Imagenet pretrained values
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("IMAGENET_MEAN", imagenet_mean)
        self.register_buffer("IMAGENET_STD", imagenet_std)

    def _input_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet mean and standard deviation normalization to the input tensor."""
        return (x - self.IMAGENET_MEAN) / self.IMAGENET_STD

    def _decoder_block(self, in_ch: int, out_ch: int):
        """U-Net type decoder block"""
        return nn.ModuleDict({
            "upconv": nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            "bn": nn.BatchNorm2d(out_ch * 2),
            "conv": ConvBlock_eff(out_ch * 2, out_ch)
        })


    def forward(self, x):
        # Encoder inc patch expansion
        x = self._input_norm(x)
        s0 = self.topconv(x)
        feats = self.encoder(x)
        s1, s2, s3, s4 = feats["skip_1_4"], feats["skip_1_8"], feats["skip_1_16"], feats["bottleneck"]
        s1 = patch_expansion(s1)
        s2 = patch_expansion(s2)
        s3 = patch_expansion(s3)
        x = patch_expansion(s4)
        x = self.bottleneck_conv(x)

        # Decoder
        skip_connections = [s3, s2, s1, s0]
        decoder_keys = ["up4", "up3", "up2", "up1"]
        for key, skip in zip(decoder_keys, skip_connections):
            block = self.decoders[key]
            x = block["upconv"](x)
            skip = center_crop(skip, x.shape[2:])
            x = torch.cat([x, skip], 1)
            x = block["bn"](x)
            x = block["conv"](x)
        x = self.final_conv(x)

        return x


class SwinB_UNet(nn.Module):
    """
    U-Net with Swin-B backbone.
    """
    def __init__(self, out_channels=1, pretrained: bool=True, weight_path: int = None):
        super().__init__()
        if pretrained == True:
            try:
                weights = Swin_B_Weights.IMAGENET1K_V1
                base = swin_b(weights=weights)
            except Exception as e:
                print("Could not load pretrained weights remotely, try accessing local file.")
                if weight_path is not None:
                    weights_path = weight_path
                else:
                    print("Please provide a local filepath for the backbone weights")
                    sys.exit()
                base = swin_b(weights=None)
                state_dict = torch.load(weights_path, map_location="cpu")
                base.load_state_dict(state_dict)
        else: 
            base = swin_b(weights=None)

        return_nodes = {
            "features.1": "skip_1_4",      # 1/4 res, 128C
            "features.3": "skip_1_8",      # 1/8 res, 256C
            "features.5": "skip_1_16",     # 1/16 res, 512C
            "features.7": "bottleneck",    # 1/32 res, 1024C
        }
        self.encoder = create_feature_extractor(
            base, 
            return_nodes=return_nodes
        )
        self.channels = [16, 32, 64, 128, 256]
        self.decoders = nn.ModuleDict({
            "up4": self._decoder_block(self.channels[4], self.channels[3]),
            "up3": self._decoder_block(self.channels[3], self.channels[2]),
            "up2": self._decoder_block(self.channels[2], self.channels[1]),
            "up1": self._decoder_block(self.channels[1], self.channels[0]),
        })

        self.topconv = nn.Conv2d(3, 16, 1)
        self.bottleneck_conv = ConvBlock_eff(self.channels[4], self.channels[4])
        self.final_conv = nn.Conv2d(self.channels[0], out_channels, 1)

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("IMAGENET_MEAN", imagenet_mean)
        self.register_buffer("IMAGENET_STD", imagenet_std)

    def _input_norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.IMAGENET_MEAN) / self.IMAGENET_STD

    def _decoder_block(self, in_ch, out_ch):
        return nn.ModuleDict({
            "upconv": nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            "bn": nn.BatchNorm2d(out_ch * 2),
            "conv": ConvBlock_eff(out_ch * 2, out_ch)
        })

    def _center_crop(self, x, target_size):
        _, _, h, w = x.shape
        th, tw = target_size
        i = (h - th) // 2
        j = (w - tw) // 2
        return x[:, :, i:i+th, j:j+tw]

    def forward(self, x):
        # Encoder
        x = self._input_norm(x)
        s0 = self.topconv(x)
        feats = self.encoder(x)
        s1, s2, s3, s4 = feats["skip_1_4"], feats["skip_1_8"], feats["skip_1_16"], feats["bottleneck"]
        s1 = patch_expansion(s1)
        s2 = patch_expansion(s2)
        s3 = patch_expansion(s3)
        x = patch_expansion(s4)

        x = self.bottleneck_conv(x)

        # Decoder
        skip_connections = [s3, s2, s1, s0]
        decoder_keys = ["up4", "up3", "up2", "up1"]
        for key, skip in zip(decoder_keys, skip_connections):
            block = self.decoders[key]
            x = block["upconv"](x)
            skip = center_crop(skip, x.shape[2:])
            x = torch.cat([x, skip], 1)
            x = block["bn"](x)
            x = block["conv"](x)
        x = self.final_conv(x)

        return x