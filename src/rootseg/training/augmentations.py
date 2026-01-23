import numpy as np
import torch
from torchvision.transforms import ColorJitter
from scipy.ndimage import map_coordinates, gaussian_filter


class Augmenter:
    """
    Image augmentation pipeline.

    Applies the following augmentations (in order):

    - **Elastic deformation** (applied with probability 2/3)
    - **Gaussian noise**
    - **Salt-and-pepper noise**
    - **Random flipping**
        - left-right flip with probability 0.5
        - up-down flip with probability 0.5
    - **Color jitter**
        - brightness
        - contrast
        - saturation
        - hue
    """
    def __init__(self, norm2: bool=False):
        self.color_jitter = ColorJitter(
            brightness=0.3,
            contrast=0.3, 
            saturation=0.2,
            hue=0.001
        )
        self.norm2 = norm2 # Alternative normalisation zero-centered to [-0.5, 0.5] instead of [0, 1]

    def _salt_pepper(self, image, num_pixels = int(572**2*0.001)):
        """Salt-Pepper for images [C, Y, X] with range in [0, 1]"""
        x_coords_white = np.random.randint(0, image.shape[1], size=num_pixels)
        y_coords_white = np.random.randint(0, image.shape[0], size=num_pixels)
        x_coords_black = np.random.randint(0, image.shape[1], size=num_pixels)
        y_coords_black = np.random.randint(0, image.shape[0], size=num_pixels)

        img_out = image.clone()
        img_out[:, y_coords_white, x_coords_white] = 1.0
        img_out[:, y_coords_black, x_coords_black] = 0.0

        return img_out

    def _gaussian_noise(self, img):
        sigma = torch.abs(torch.normal(mean=0, std=0.09, size=(1,))).item()  # scalar std dev
        img = img + torch.randn_like(img) * sigma
        img = torch.clamp(img, 0, 1)
        return img

    def _MinMax(self, img):
        return (img - img.min()) / (img.max() - img.min())

    def __call__(self, img, annotation):
        # 1) Transform to torch tensors + split annotation
        img = img.permute(2, 0, 1)
        annotation = annotation.permute(2, 0, 1)

        img = self._MinMax(img)

        # 2) Elastic Transform
        if torch.rand(1).item() < 0.66:
            combined = torch.cat([img, annotation], dim=0)  #(C_img + C_mask, H, W)
            combined = elastic_deform(combined)
            img = combined[:3, :, :]
            annotation = combined[3:, :, :].to(torch.float16)

        # 3) Gaussian Noise + S&P
        img = self._gaussian_noise(img)
        img = self._salt_pepper(img)

        #4) Flip + Color jit
        if torch.rand(1).item() < 0.5:
            img = torch.fliplr(img)
            annotation = torch.fliplr(annotation)
        if torch.rand(1).item() < 0.5:
            img = torch.flipud(img)
            annotation = torch.flipud(annotation)
        
        img = self.color_jitter(img)
        img = self._MinMax(img)
        if not self.norm2:
            img -= 0.5
            img *= 2

        img = img.permute(1, 2, 0)
        annotation = annotation.permute(1, 2, 0)

        return img, annotation


def elastic_deform(image: torch.Tensor, padding: int=60) -> torch.Tensor:
    """
    Elastic deformation based on cognitivemedium.com/assets/rmnist/Simard.pdf
    """
    image = image.numpy()
    image = np.pad(
        image,
        [(0, 0), (padding, padding), (padding, padding)],
        mode="constant",
        constant_values=0
    )
    elastic_map = get_elastic_map(image.shape[1:])
    # If c>2 (annotation channel), use nearest neighbors (order=0) to not interpolate class values
    warped = np.stack(
        [map_coordinates(image[c, ...], elastic_map, order=0 if c > 2 else 1)
        for c in range(image.shape[0])],
        axis=0
    )
    warped = image[:, padding:-padding, padding:-padding]
    warped = torch.from_numpy(warped)
    return warped


def get_indices(im_shape, scale, sigma):
    randx = np.random.uniform(low=-1.0, high=1.0, size=im_shape)
    randy = np.random.uniform(low=-1.0, high=1.0, size=im_shape)
    x_filtered = gaussian_filter(randx, sigma, mode="reflect") * scale
    y_filtered = gaussian_filter(randy, sigma, mode="reflect") * scale
    x_coords, y_coords = np.mgrid[0:im_shape[0], 0:im_shape[1]]
    x_deformed = x_coords + x_filtered
    y_deformed = y_coords + y_filtered
    return x_deformed, y_deformed


def get_elastic_map(im_shape):
    min_alpha = 200
    max_alpha = 2500
    scale = np.random.uniform()
    intensity = np.random.uniform()

    min_sigma = 15
    max_sigma = 60
    alpha = min_alpha + ((max_alpha-min_alpha) * scale)
    alpha *= intensity
    sigma = min_sigma + ((max_sigma-min_sigma) * scale)
    return get_indices(im_shape, scale=alpha, sigma=sigma)