"""
This module contains helper functions for the preprocessing of raw root images.

"""

import numpy as np
from typing import Tuple, List, Any
import cv2
import os
from datetime import datetime
from skimage.feature import match_template
from scipy.ndimage import binary_dilation
import warnings
import logging


def get_level(filename:str) -> str:
    """
    Assumes structure {name}_{tube}_{level}_{date}_...
    """
    filename = os.path.basename(filename)
    level = filename.split("_")[2]
    level = int(level[1])
    return level


def get_date(filename: str) -> datetime:
    """
    Extract the date from the filename. The date is expected to be in the format "DD.MM.YY".

    Args:
        filename (str) : name of the file containing the date

    Returns:
        datetime : extracted date

    """
    filename = os.path.basename(filename)
    date = filename.split("_")[3]
    try:
        date = datetime.strptime(date, "%d.%m.%y") #NOTE: CHANGE FORMAT
    except ValueError:
        try:
            date = datetime.strptime(date, "%Y.%m.%d")
        except ValueError:
            raise ValueError(f"Date format not recognized for '{date}'")

    return date


def fix_image_size(
        img: np.ndarray, 
        name: str = '',
        size: Tuple[int, int] = (8784, 10200), 
        crop_left: bool = True,
) -> np.ndarray:
    """
    Fix the size of an image by cropping or padding (mirrored pad).
    
    Args:
        img (np.array) : image to be processed
        size (Tuple[int, int]) : target size of the image
        crop_left (bool) : crop from left side of the image, else from right side
                           In case of stitched images this removes tape parts instead of soil

    Returns:
        img (np.array) : processed image
    
    """
    # Get image size and return if it already has the correct size
    h, w = img.shape[:2]
    h_target, w_target = size
    if (h, w) == (h_target, w_target): 
        return img
    elif (w, h) == (h_target, w_target):
        logging.warning(f'Image {name} seems to be rotated! Fix manually please')
    
    # Crop or pad the image to the target size
    x_pad, y_pad = max(0, w_target - w), max(0, h_target - h)
    if x_pad > 0 or y_pad > 0:
        if crop_left: # pad from left side of the image
            img = np.pad(img, ((0, y_pad), (x_pad, 0), (0, 0)))
            if x_pad:
                img[:, :x_pad, :] = [255, 0, 0]
            if y_pad:
                img[-y_pad:, :, :] = [255, 0, 0]
        else:
            img = np.pad(img, ((0, y_pad), (0, x_pad), (0, 0)))
            if x_pad:
                img[:, -x_pad:, :] = [255, 0, 0]
            if y_pad:
                img[-y_pad:, :, :] = [255, 0, 0]
    else: 
        if crop_left: # crop from left side of the image
            img = img[:h_target, -w_target:]
        else:
            img = img[:h_target, :w_target]
    #black_mask = np.all(img == [0, 0, 0], axis=-1)
    #img[black_mask] = [255, 0, 0]
    img = img.astype(np.uint8)
    return img

def get_stitch_coords(
        L_top: np.ndarray, 
        L_bot: np.ndarray, 
        tube: int, 
        clahe: cv2.CLAHE
) -> Tuple[int, int, float]:
    """
    Use cv2.PhaseCorrelate to get the shift of maximum correlation between two images

    Input:
        L_top (np.ndarray): top image to be stitched
        L_bot (np.ndarray): bottom image to be stitched
        tube (int): tube - in case of warning report
        clahe (cv2.CLAHE): opencv clahe to improve phase correlation performance

    Returns:
        dx: shift in x-direction
        dy: shift in y-direction
        corr: maximum correlation at (dx, dy)

    """
    if L_top is None or L_bot is None:
        logging.error(f"{tube}: missing image detected -> return zero correlation for stitching process!")
        return ("-", "-", 0.0)
    
    L1_gray = np.float32(clahe.apply(cv2.cvtColor(L_top, cv2.COLOR_BGR2GRAY)))
    L2_gray = np.float32(clahe.apply(cv2.cvtColor(L_bot, cv2.COLOR_BGR2GRAY)))
    shift = cv2.phaseCorrelate(L1_gray[::5, ::5], L2_gray[::5, ::5])

    # If correlation is small, try to preprocess images first and retry - usually slightly improves correlation
    if shift[1] < 0.1:
        L1_gray = norm_grayscale(L1_gray)
        L2_gray = norm_grayscale(L2_gray)
        h = L1_gray.shape[0]

        # Take advantage of 360 deg scan in y-dir
        L1_gray = np.vstack([L1_gray[-h//2:], L1_gray, L1_gray[:h//2]])
        L2_gray = np.vstack([L2_gray[-h//2:], L2_gray, L2_gray[:h//2]])
        shift_new = cv2.phaseCorrelate(L1_gray[::5, ::5], L2_gray[::5, ::5])
        if shift_new[1] > shift[1]:
            logging.info(f"{tube}: enhance images for stitching: Correlation went from {shift[1]} to {shift_new[1]}")
            shift = shift_new
    dx, dy = int(round(shift[0][0])), int(round(shift[0][1]))
    dx *= 5
    dy *= 5
    shift_data = (dx, dy, shift[1])
    return shift_data

def iterative_stitching(
        img_list: List[np.ndarray], 
        shift_list: List[Tuple[int, int, float]], 
        prev_length: int = None, 
        tube: str = None
) -> Tuple[np.array, int]:
    """
    Stitch multiple images together from a list of images and shifts. In case of empty images != L1, leave blank
    and fix the image size to the previous stitched image.
    Under the assumption that the bottom part is a fixed position in each image, we stitch from bottom upwards

    Args:
        img_list (List[np.ndarray]): List of images to stitch, sorted from top down
        shift_list: (List[Tuple[int, int, float]]): List of shifts calculated from cv2.PhaseCorrelate
        prev_length (int): previous image length - In case of missing images, output length is fixed by this
        tube (str): Tube number in case of error messages

    Returns:
        new_img (np.array): Stitched image
        length (int): length of the new image - to reference for the subsequent image
    """
    height = img_list[0].shape[0]
    length = 0
    try:
        for i, img in enumerate(img_list):
            length += img.shape[1]
            if i > 0:
                if shift_list[i-1][0] > 0:
                    length -= shift_list[i-1][0]
                elif shift_list[i-1][0] == 0: # case of identical images
                    raise ValueError(f"{tube}: Zero x-shift (dx=0) detected, which is considered invalid.")
                else: # wrapped correlation case
                    shift_list[i-1] = (shift_list[i-1][0]+img.shape[1], shift_list[i-1][1], shift_list[i-1][2])
                    length -= shift_list[i-1][0]
    except Exception as e:
        print('run prev size')
        print(f"An error occurred: {e}") # This line prints the exception
        if not prev_length:
            logging.critical(f"{tube}: First date of the series does not contain all scan levels -- Aborting!")
            raise RuntimeError(f"{tube}: Cannot proceed because the first date of the series does not contain all scan levels.")
        length = prev_length
    
    # Mask background blue
    blue_array = np.array([0, 0, 255], dtype=np.uint8)
    new_img = np.full((height, length, 3), blue_array,  dtype=np.uint8)
    w = 0
    for i, img in enumerate(img_list):
        if i == 0: # L1 image case
            w += img.shape[1]
            new_img[:, :w, :] = img
            continue

        dx, dy = shift_list[i-1][:2]
        # shift y-axis to align and crop dx 
        if img is not None and dx > 0: # if empty or no x-shift occurs - leave it blue
            img = np.roll(img, -dy, axis=0)
            img = img[:, dx:, :]
            h2, w2 = img.shape[:2]
            new_img[:, w:w+w2, :] = img
            w += img.shape[1]
        else:
            break
    return new_img, length


def norm_grayscale(img_gray: np.ndarray) -> np.ndarray:
    """Normalize grayscale images to attempt better matching in case of stitching and time series alignment"""
    exclude_mask = (img_gray < 10) | (img_gray > 200)
    img = img_gray.copy()
    img[exclude_mask] = np.nan
    mean = np.nanmean(img, axis=(0, 1), keepdims=True)
    with warnings.catch_warnings(): # suppress RuntimeWarning: Mean of empty slice
        warnings.simplefilter("ignore", category=RuntimeWarning)
        colmean = np.nanmean(img, axis=0, keepdims=True)
    colmean = np.where(np.isnan(colmean), 0, colmean)
    img += mean - colmean
    img_gray += mean - colmean 

    mean = np.nanmean(img, axis=(0, 1), keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        rowmean = np.nanmean(img, axis=1, keepdims=True)
    rowmean = np.where(np.isnan(rowmean), 0, rowmean)
    img_gray += mean - rowmean
    img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

    # Contrast and brightness norm
    mask = np.full(img_gray.shape, 255, dtype=np.uint8)
    mask[img_gray == 0] = 0
    ref_contrast = 21
    ref_brightness = 90

    # Get current contrast and brightness
    mean, stddev = cv2.meanStdDev(img_gray, mask=mask)
    contrast = float(ref_contrast / (stddev[0][0] + 1e-8))

    img_gray = cv2.addWeighted(img_gray, contrast, img_gray, 0, 0)

    mean, _ = cv2.meanStdDev(img_gray, mask=mask)
    brightness = float(ref_brightness - mean[0][0])

    img_gray = cv2.addWeighted(img_gray, 1, img_gray, 0, brightness)

    return np.float32(np.clip(img_gray, 0, 255).astype(np.uint8))


def crop_rightmost_nonblack(img_gray: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Extract an image region from the right side, neglecting black columns on the right border.

    Input:
        img_gray (np.ndarray): grayscale image

    Returns:
        img_gray (np.ndarray): cropped grayscale image
        rightmost (int): index of the rightmost non-black column
    """
    # Sum pixel intensities column-wise
    col_sums = np.sum(img_gray, axis=0)  # shape: (W,)

    # Find indices of non-black columns
    non_black_cols = np.where(col_sums > 0)[0]

    if non_black_cols.size == 0:
        raise ValueError("Image is completely black!")

    rightmost = non_black_cols[-1]  # last non-black column index

    # Calculate start and end for slicing
    #start = max(0, rightmost - width + 1)
    end = rightmost + 1

    return img_gray[:, :end], rightmost

def rem_tape(img: np.ndarray, mask_list: List[str], L1_name: str, tube: str) -> np.ndarray:
    """
    Removes tape from an image using skimage.feature.match_template from given tape templates

    Input:
        img (np.ndarray): RGB image
        mask_list (List[str]): List of mask paths to consider the tape removal. 
                               Should consist of a pool of mask for a given tube
        L1_name (str): Name of the RGB image - In case of error reports
        tube (str): Tube name - In case of error reports

    Returns:
        img: RGB image with tape masked out if the best match has >10% correlation, else returns the input

    """
    if not mask_list: # In case of an empty list, return
        return img
    
    # Wrap half the image on top and bottom to account for periodic BC on y-axis for template matching
    H, W = img.shape[:2]
    img_gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    wrapped_img = np.float32(np.vstack((img_gray[:H//2, :W], img_gray, img_gray[H//2:, :W])))
    best_score = 0
    # Iterate through all masks and keep the best fitting mask
    for mask_path in mask_list:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_float = np.float32(mask)
        matched = match_template(wrapped_img[::5, ::5], mask_float[::5, ::5])
        if best_score < np.max(matched): 
            best_score = np.max(matched)
            best_match = matched
            mask_keep = mask
            
    if best_score < 0.2:
        logging.warning(f"{tube}: No matching mask found for {L1_name} - returning image")
        return img
    elif best_score < 0.2: # consider 10%-20% as lower threshold for warning..
        logging.warning(f"{tube}: Low correlation to match any mask ({best_score:.2f}) for image {L1_name}")

    # apply the mask
    ij = np.unravel_index(np.argmax(best_match), best_match.shape)
    y_mask, x_mask = ij
    y_mask *= 5
    x_mask *= 5
    y_mask -= H//2
    # In case the mask width is too large, crop it - assumes that the mask is black everywhere where no tape
    if mask_keep.shape[1] > (img.shape[1]) // 2:
        mask = mask_keep[:, :(img.shape[1]) // 2]
    else:
        mask = mask_keep
    h, w = mask.shape
    if h != img.shape[0]: # scale the mask to fit the image height - raise warning
        logging.warning(f"mask height ({mask.shape[0]}) does not coincide with image height ({img.shape[0]})! Mask is being deformed to match shape.")
        mask = cv2.resize(mask, (w, img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # adjust the mask with dilation and slight shift (small oversizing effect)
    mask = np.roll(mask, shift=y_mask, axis=0)
    bool_mask = (mask > 0)
    bool_mask = binary_dilation(bool_mask, iterations=100)
    bool_mask_3d = np.repeat(bool_mask[:, :, np.newaxis], 3, axis=2)
    #bool_mask = np.roll(bool_mask, 50, axis=1)
    #img[:, :x_mask+50+25, :] = 0
    #img[:, x_mask+25:x_mask+w+25, :][bool_mask_3d] = 0

    # Create dilated mask
    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[:, :x_mask+75] = True
    mask[:, x_mask+25:x_mask+w+25] |= bool_mask_3d.any(axis=-1)
    img[mask] = [255, 0, 0] # Blue color for mask for now
    return img


def applyCLAHE(img: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
    """ Apply CLAHE to the L channel of an image in the LAB color space."""
    blue_mask = (
        (img[..., 0] == 255) &      
        (img[..., 1] == 0) &      
        (img[..., 2] == 0)    
    )
    blue_pixels = img.copy()[blue_mask]
    #mask = np.all(img == 0, axis=-1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[...,0] = clahe.apply(lab[...,0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img[blue_mask] = blue_pixels
    return img

def add_reference_pixels(img) -> np.ndarray:
    """ By default places a single green pixel in the middle 1000px from right border"""
    img[img.shape[0]//2, -1000, :] = (0, 255, 0)
    return img

def remove_scannoise(img: np.ndarray) -> np.ndarray:
    """"
    Remove scan noise from an image by removing the column and row mean, neglecting black regions.
    This function might need modification if tape is included in the image.
    Input:
        img (np.array) : image to be processed

    Returns:
        img (np.array) : processed image
    
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype=np.float32)
    blue_mask = (
        (img[..., 0] == 255) &      
        (img[..., 1] == 0) &      
        (img[..., 2] == 0)    
    )
    blue_pixels = img.copy()[blue_mask]
    black_mask = (img_gray < 10) | (img_gray > 200) | blue_mask
    light_mask = (img_gray > 220)
    #black_indices = np.where(mask == 0)
    #exclude_mask = black_indices
    exclude_mask = black_mask # | mask
    #img[black_indices] = np.nan # exclude black regions
    img_copy = img.copy()
    img_copy[exclude_mask] = np.nan # apply mask for mean calc
    mask_img = np.all(img == 0, axis=-1)
    img[mask_img] = np.nan

    
    # remove column mean
    mean = np.nanmean(img_copy, axis=(0, 1), keepdims=True)
    # suppress RuntimeWarning: Mean of empty slice -> due to mask / non-concerning
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore", category=RuntimeWarning)
        colmean = np.nanmean(img_copy, axis=0, keepdims=True)
    img_copy += mean - colmean
    img += mean - colmean 

    # remove row mean
    mean = np.nanmean(img_copy, axis=(0, 1), keepdims=True)
    rowmean = np.nanmean(img_copy, axis=1, keepdims=True)
    img += mean - rowmean 

    # convert back to valid pixel values
    img = np.nan_to_num(img)
    img = np.clip(img, 0, 255)
    #img[light_mask, :] = [255, 0, 0] don't filter bright parts..

    img[blue_mask] = blue_pixels

    return img.astype(np.uint8)


def contrast_brightness_normalization(img: np.ndarray) -> np.ndarray:
    """
    Adjust the contrast and brightness of an image to match a reference value, excluding black regions.
    
    """
    # create mask to exclude black regions
    #mask = np.full((img.shape[0], img.shape[1]), 255, dtype = np.uint8)
    #mask[img[..., 0] == 0] = 0 # mask out the black regions for contrast and brightness calc.

    mask = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
    blue_mask = (img[..., 0] == 255) & (img[..., 1] == 0) & (img[..., 2] == 0)
    blue_pixels = img.copy()[blue_mask]
    mask[blue_mask] = 0


    # reference values
    ref_contrast = 21
    ref_brightness = 90

    # get current brightness and contrast
    _, im_contrast = cv2.meanStdDev(img[..., 0], mask = mask)
    contrast = float(ref_contrast / im_contrast)
    
    img = cv2.addWeighted(img, contrast, img, 0, 0)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_brightness, _ = cv2.meanStdDev(img_grey, mask = mask)

    brightness = float(ref_brightness - im_brightness)
    img = cv2.addWeighted(img, 1, img, 0, brightness)

    img[blue_mask] = blue_pixels

    # new mask?
    #img[mask == 0] = 0


    return img