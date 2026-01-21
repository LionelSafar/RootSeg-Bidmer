from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import optuna

from rootseg.training.logger import DataLogger
from rootseg.training.datasets import PrefetchWrapper


def dilate(image: torch.Tensor, kernel_size: int=3) -> torch.Tensor:
    """Morphological dilation"""
    return F.max_pool2d(image, kernel_size, stride=1, padding=kernel_size//2)


def erode(image: torch.Tensor, kernel_size:int =3) -> torch.Tensor:
    """Morphological erosion"""
    return -F.max_pool2d(-image, kernel_size, stride=1, padding=kernel_size//2)


def morphological_gradient(image: torch.Tensor, kernel_size: int=3) -> torch.Tensor:
    """Morphological gradient - Creates a mask of ~2px thickness at boundaries of a binary image"""
    image = image.float()
    boundary_px = dilate(image, kernel_size) - erode(image, kernel_size)
    return (boundary_px > 0).float()


def dice_loss(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    """
    Dice loss function, that allows to calculate the mean loss of multiple classes as well.
    """
    # Flatten to (N, C, H*W)
    probs_flat = probs.flatten(start_dim=2)
    labels_flat = labels.flatten(start_dim=2)

    if mask is not None:
        mask_flat = mask.flatten(start_dim=2)
        probs_flat = probs_flat * mask_flat
        labels_flat = labels_flat * mask_flat

    # Get per-class dice loss (N, C)
    intersection = torch.sum(probs_flat * labels_flat, dim=2)
    union = torch.sum(probs_flat, dim=2) + torch.sum(labels_flat, dim=2)
    dice_score_per_class = (2.0 * intersection + 1e-10) / (union + 1e-10)
    dice_loss_per_class = 1.0 - dice_score_per_class

    return dice_loss_per_class.mean()


def focal_loss_fn(
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        gamma: int=2.5, 
        reduction: Literal['mean', 'sum']='mean', 
        mask: torch.Tensor=None
) -> torch.Tensor:
    """
    Focal loss for multi-class segmentation.

    Args:
        logits (torch.Tensor): (N, C, H, W)
        targets (torch.Tensor): (N, H, W)  - integer class labels
        gamma (int): gamma factor of the loss
        reduction (Literal['mean', 'sum']): mean or sum of focal loss over the image
        mask (torch.Tensor): (H, W) If provided, masks the focal loss
    
    Returns:
        torch.Tensor: Either sum or mean focal loss error value
    """
    #do not consider alpha here.
    ce_loss = F.cross_entropy(logits, targets, reduction='none')  # (N, H, W)
    pt = torch.exp(-ce_loss)  # = prob of the true class
    focal_loss = (1 - pt) ** gamma * ce_loss

    if mask is not None:
        focal_loss = focal_loss * mask

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss


def confusion_metrics(
        probs: torch.Tensor, 
        labels: torch.Tensor, 
        mask: torch.Tensor=None, 
        adjust_dim: bool=False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates confusion metrics from model probabilities and labels

    Args:
        probs (torch.Tensor): Output probabilities of the model (softmax / sigmoid applied)
        labels (torch.Tensor): gt labels with classes n in N
        mask (torch.Tensor): binary mask to mask out region (e.g. boundary regions) for metrics calculation
        adjust_dim (bool): Whether to shift probabilities and include a background class

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple containing per-class confusion-matrix components of shape [C,]:

            - TP: True positives
            - FP: False positives
            - FN: False negatives
            - TN: True negatives
    """
    # Multiclass, predictions contain a background class
    if probs.shape[1] > 1 and not adjust_dim:
        _, C, _, _ = probs.shape
        predictions = torch.argmax(probs, dim=1) 
        preds = torch.nn.functional.one_hot(predictions, num_classes=C)  # [N, H, W, C]
        preds = preds.permute(0, 3, 1, 2).float()  # -> [N, C, H, W]

    # Multiclass, labels, predictions do not contain background class
    elif probs.shape[1] > 1 and adjust_dim:
        N, C, H, W = probs.shape
        preds = torch.argmax(probs, dim=1) 
        preds = F.one_hot(preds, num_classes=C)  # [N, H, W, C]
        zero_channel = torch.zeros(N, 1, H, W, device=preds.device, dtype=preds.dtype)
        preds = preds.permute(0, 3, 1, 2).float()  # -> [N, C, H, W]
        preds = torch.cat((zero_channel, preds), dim=1)

    # Binary case
    else:
        preds = (probs > 0.5).to(torch.int32)
    if mask is None:
        mask = torch.ones_like(labels)
    red_dims = tuple(i for i in range(probs.dim()) if i != 1) # reduce over all dims except channels

    # Calculate (masked) confusion metrics
    TP = torch.sum(preds * labels * mask, dim=red_dims)
    FP = torch.sum(preds * (1 - labels) * mask, dim=red_dims)
    FN = torch.sum((1 - preds) * labels * mask, dim=red_dims)
    TN = torch.sum((1 - preds) * (1 - labels) * mask, dim=red_dims)

    return TP, FP, FN, TN


def masked_loss(
        model: torch.nn.Module, 
        images: torch.Tensor, 
        labels: torch.Tensor, 
        alpha: float=0.3, 
        gamma: float=2.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    multiclass combined loss (focal loss + dice loss) with masking.

    Args:
        model (torch.nn.Module): NN model
        images (torch.Tensor): Input image batch (N, C, H, W)
        labels (torch.Tensor): Ground truth labels (N, 1, H, W)
        alpha (float): Scaling factor for the focal loss
        gamma (float): Gamma factor inside focal loss

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (loss, confusion_stats)
    """

    # Forward pass and softmax over channel dim
    logits = model(images) 
    probs = F.softmax(logits, dim=1)

    # Create binary rootmask for masked loss
    root_mask = (labels > 0).to(torch.float32) # (N, C, H, W)

    # # shifted labels but put -1 back to 0 to remove background class for focal loss calculation
    labels_red = labels - 1.0 
    labels_red = torch.clamp(labels_red, min=0)
    labels_red = labels_red.squeeze(1).long()
    fc_loss = focal_loss_fn(logits, labels_red, gamma=gamma, reduction='mean', mask=root_mask)
    labels_onehot = F.one_hot(
        labels_red, 
        num_classes=logits.shape[1]
    ).permute(0, 3, 1, 2).to(torch.float32)

    # OHE labels including background class
    labels_true = F.one_hot(
        labels.squeeze(1).long(), 
        num_classes=logits.shape[1]+1 # C + background classes
    ).permute(0, 3, 1, 2).to(torch.float32)

    # Get mean dice loss over non-background classes
    foreground_dice_loss = dice_loss(probs, labels_onehot, root_mask)

    # Gather loss and masked confusion metrics
    loss =  foreground_dice_loss + alpha * fc_loss
    TP, FP, FN, TN = confusion_metrics(probs.detach(), labels_true, mask=root_mask, adjust_dim=True)
    stats = torch.stack((TP[1:], FP[1:], FN[1:], TN[1:]), dim=0).to(logits.device)

    return loss, stats


def combined_loss(
        model: torch.nn.Module, 
        images: torch.Tensor, 
        labels: torch.Tensor, 
        alpha: float=0.3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single class (binary) combined loss function (weighted CE + dice loss)

    Args:
        model (torch.nn.Module): NN model
        images (torch.Tensor): Input image batch (N, 1, H, W)
        labels (torch.Tensor): Ground truth labels (N, 1, H, W)
        alpha (float): Scaling factor for the weighted cross entropy loss

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (loss, confusion_stats)
    """

    # Forward pass
    logits = model(images)
    probs = F.sigmoid(logits)
    
    # Weighted binary cross entropy loss, with lower weighting at boundaries
    boundary_mask = morphological_gradient(labels)
    weighted_mask = torch.where(boundary_mask > 0, 0.85, 1.0)
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    bce_loss = (bce_loss * weighted_mask).mean()

    # total loss
    d_loss = dice_loss(probs, labels)
    loss = d_loss + alpha * bce_loss
    
    # Confusion metrics - Ignore boundary region for evaluation
    mask_inv = (1.0 - boundary_mask)
    TP, FP, FN, TN = confusion_metrics(probs.detach(), labels, mask_inv)
    stats = torch.stack((TP, FP, FN, TN), dim=0)

    return loss, stats


def training_step(model, optimizer, images, labels, alpha=0.3, gamma=2.5, multiclass: bool=False, epoch=None):
    """
    Forward pass and backpropagation for single- or multiclass training.
    Applies Gradient clipping after the 5th epoch.
    """
    optimizer.zero_grad()
    if multiclass:
        loss, stats = masked_loss(model, images, labels, alpha, gamma=gamma)
    else:
        loss, stats = combined_loss(model, images, labels, alpha)
    loss.backward()
    if epoch > 5:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2) # Gradient clipping
    optimizer.step()

    return loss.item(), stats

def val_step(model, images, labels, alpha=0.3, gamma=2.5, multiclass: bool=False):
    """
    Forward pass without gradients for single- or multiclass training.
    """

    with torch.no_grad():
        if multiclass:
            #loss, stats = multiclass_loss(model, images, labels, alpha, gamma=gamma)
            loss, stats = masked_loss(model, images, labels, alpha, gamma=gamma)
        else:
            loss, stats = combined_loss(model, images, labels, alpha)

    return loss.item(), stats


def training_loop(
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler._LRScheduler, 
        train_loader: PrefetchWrapper, 
        val_loader: PrefetchWrapper, 
        epochs: int=50, 
        alpha: float=0.3, 
        gamma: float=2.0, 
        save_path: str=None, 
        logger: DataLogger=None, 
        device: torch.device='cpu', 
        trial: optuna.Trial = None, 
        N_classes: int=1
):
    """
    main training function for single- and multiclass models.

    Args:
        model (torch.nn.Module): torch NN model
        optimizer (torch.optim.Optimizer): torch optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): torch LR scheduler (default is Cosine annealing)
        train_loader (PrefetchWrapper): wrapped training dataloader
        val_loader (PrefetchWrapper): wrapped validation dataloader
        epochs (int): amount of epochs to train
        alpha (float): scaling factor for the focal / bce loss calculation
        gamma (float): gamma factor for focal loss in case of multiclass segmentation
        save_path (str): saving directory
        logger (DataLogger): Datalogger class instance 
        device (torch.device): torch device 'gpu' or 'cpu'
        trial (optuna.Trial): In case of hyperparameter optimisation, for trial report and pruning.
        N_classes (int): Amount of classes that are predicted (excluding background)
    """

    best_f1 = 0.0 # store best f1 to track the best performing model

    for epoch in tqdm(range(1, epochs + 1), desc='Training Progress'):

        # Initialise train and val loss / stat tracking
        train_loss_deque = []
        val_loss_deque = []
        train_stats_total = torch.zeros(4, N_classes+1).to(device)
        val_stats_total = torch.zeros(4, N_classes+1).to(device)
        
        # Training
        model.train()
        for images, labels in train_loader:
            loss, stats = training_step(
                model, optimizer, images, labels, alpha, gamma=gamma, multiclass=N_classes>1, epoch=epoch
            )
            train_loss_deque.append(loss)
            train_stats_total += stats
        scheduler.step()

        # Move stats to cpu
        train_stats_total = train_stats_total.cpu()
        train_loss = np.mean(train_loss_deque)
        
        # Validation
        model.eval()
        for images, labels in val_loader:
            val_loss, stats = val_step(
                model, images, labels, alpha, gamma=gamma, multiclass=N_classes>1
            )
            val_loss_deque.append(val_loss)
            val_stats_total += stats

        # Move stats to cpu
        val_loss = np.mean(val_loss_deque)
        val_stats_total = val_stats_total.cpu()
        TP, FP, FN, _ = val_stats_total
        val_f1 = (2*TP / (2*TP + FP + FN + 1e-10)).mean()

        # Log epoch stats
        if logger:
            logger.log((train_stats_total, train_loss), 'Train')
            logger.log((val_stats_total, val_loss), 'Val')
            logger.print_last_metrics()
        
        # Save if best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            if save_path:
                torch.save(model.state_dict(), save_path+f'model_best.pth')
                print(f"New best model saved at epoch {epoch} with F1 {best_f1:.3}")

        # Reporting to optuna in case of hyperparameter optimisation run
        if trial:
            trial.report(val_f1, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    print("Training complete.")
    return model, logger



class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    """
    Modification of CosineAnnealingWarmRestarts scheduler such that the maximum learning rate decays 
    at a given rate each restart.
    
    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int): A factor increases T_i after a restart.
        eta_min (float): Minimum learning rate.
        lr_decay_factor (float): The factor by which to decay the learning rate at each restart.
        last_epoch (int): The index of last completed epoch.
    """
    def __init__(
            self, 
            optimizer: torch.optim.Optimizer, 
            T_0: int, 
            T_mult: int=1, 
            eta_min: float=0, 
            lr_decay_factor: float=0.8, 
            last_epoch: int=-1
    ):
        # Validate the decay factor to be in (0, 1]
        if not 0.0 < lr_decay_factor <= 1.0:
            raise ValueError("lr_decay_factor must be between 0.0 and 1.0.")
        
        # Store initial base LRs before they are modified
        self.lr_decay_factor = lr_decay_factor
        self.initial_base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)

    def step(self, epoch=None):
        # At epoch just before a restart, decay the base learning rates
        if self.T_i - self.last_epoch == 1:
            # decay the base learning rates
            self.base_lrs = [base_lr * self.lr_decay_factor for base_lr in self.base_lrs]

            # update the optimizer's param_group to reflect this new base
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i]

        # perform regular cosine annealing step from parent's class
        super().step(epoch)
        
    def _get_closed_form_lr(self):
        # The parent's method uses self.base_lrs, which we are now modifying
        return super()._get_closed_form_lr()