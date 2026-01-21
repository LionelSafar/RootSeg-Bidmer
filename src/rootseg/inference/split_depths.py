"""
Splits segmented images into different depth profiles based on provided depth masks. 
Further creates a .csv file containing soil area, masked area (blackened out regions) and fraction of each image.
"""

import os
import glob

import numpy as np
from typing import List, Tuple, Dict
import cv2
from PIL import Image
import multiprocessing as mp
import pandas as pd
import argparse
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)


def get_tube_mask(filename: str, masklist: List[str]) -> str:
    """Get depth-mask for the correct tube based on input filename"""
    # Ensure basepath has structure {name}_{tube}_{level}_{date}_...
    filename = os.path.basename(filename)
    tube_nr = filename.split("_")[1][1:]
    orientation = filename.split("_")[0][-1]
    for mask_path in masklist:
        mask_name = os.path.basename(mask_path)
        if tube_nr == mask_name.split("_")[1] and orientation == mask_name.split("_")[0][-1]:
            return mask_path
    raise ValueError(f'No tube characteristic mask found for {filename}!')


def get_image_stack(path: str) -> List[Tuple[str, str, str]]:
    """
    Return a list of tuples (processed_paths, segmented_paths, mask_paths) for a project folder
    
    Args:
        path (str): project path, which should include preprocessed images, segmentations, depth masks in separate folders

    Returns:
        List[Tuple[str, str, str]]: Tuple of corresponding images with
            - (preprocessed image, segmentation, depthmask)
    """
    processed_path = os.path.join(path, 'preprocessed')
    segmented_path = os.path.join(path, 'segmented')
    mask_path = os.path.abspath(os.path.join(path, '..', 'depth_masks'))

    processed_paths = glob.glob(os.path.join(processed_path, '**', '*.tiff'), recursive=True)
    processed_paths.extend(glob.glob(os.path.join(processed_path, '**', '*.png'), recursive=True))
    mask_paths = glob.glob(os.path.join(mask_path, '*.tiff'))
    mask_paths.extend(glob.glob(os.path.join(mask_path, '*.png')))

    segmented_paths = glob.glob(os.path.join(segmented_path, '**', '*.tiff'), recursive=True)
    segmented_paths.extend(glob.glob(os.path.join(segmented_path, '**', '*.png'), recursive=True))

    processed_paths.sort(key=os.path.basename)
    segmented_paths.sort(key=os.path.basename)

    image_stack = []
    assert len(processed_paths) == len(segmented_paths), f"{len(processed_paths)} and {len(segmented_paths)}"
    for p_p, p_s in zip(processed_paths, segmented_paths):
        base_proc = os.path.splitext(os.path.basename(p_p))[0]
        base_seg = os.path.basename(p_s).replace("_segmented.png", "") # such that segmentation name == processed name
        base_seg = os.path.splitext(os.path.basename(p_s))[0]

        if base_proc != base_seg:
            print(f"[Mismatch] Processed: {p_p} | Segmented: {p_s}")

        image_stack.append((p_p, p_s, get_tube_mask(p_p, mask_paths)))
    print(len(image_stack))
    return image_stack


def worker_process(path: str) -> Dict:
    """
    Main process for each worker:
    Reads original image, segmentation and depth mask, 
    split segmentation according to depth mask and extract masked area (e.g. tape) from original image.
    Save new segmentations and return entry to dataframe with image name and area information.
    """
    img = np.array(Image.open(path[0][0]))
    mask = np.array(Image.open(path[0][2]))
    seg = cv2.imread(path[0][1], cv2.IMREAD_GRAYSCALE)
    seg = np.array(seg)
    outpath = path[1]

    parts = os.path.basename(path[0][0]).split("_")
    n1 = '0_10cm'
    n2 = '10_Xcm'

    parts1 = parts.copy()
    parts2 = parts.copy()

    # Insert depth information into copies
    parts1.insert(2, n1)
    parts2.insert(2, n2)

    out1 = os.path.join(outpath, "_".join(parts1))
    out2 = os.path.join(outpath, "_".join(parts2))

    H, W = img.shape[:2]
    # Error case - size missmatch between segmentation and preprocessed image - return NaN's for area
    if seg.shape[:2] != (H, W):
        print(f'WARNING DIFFERENT SHAPE OF segmented image {path[0][1]} and im {path[0][0]} -- SKIPPED!')
        print(f'shapes are {img.shape} for img and {seg.shape} -- return NaNs')
        return [{
            'filename': os.path.basename(out1),
            'image_area_px2': np.nan,
            'excluded_area_px2': np.nan,
            'excluded_area_fraction': np.nan
            },
            {
            'filename': os.path.basename(out2),
            'image_area_px2': np.nan,
            'excluded_area_px2': np.nan,
            'excluded_area_fraction': np.nan
            }]

    # If mask is smaller than the image, extend image with values 200 (=depth 20+cm)
    if mask.shape[1] < W:
        expanded_mask = np.ones((mask.shape[0], W), dtype=mask.dtype) * 200
        expanded_mask[:, :mask.shape[1]] = mask
        mask = expanded_mask
    else:
        mask = mask[:, :W]
    black_mask = np.zeros_like(seg)

    # 10cm mask + inverted for 10-Xcm segmentation
    mask_0_10 = (mask < 100)

    seg_0_10 = np.where(mask_0_10, seg, black_mask)
    seg_10_X = np.where(~mask_0_10, seg, black_mask)

    # Get image area per depth and masked (black) area per depth
    black_0_10 = np.sum(np.all(img[mask_0_10, :] == [0, 0, 0], axis=1))
    black_10_X = np.sum(np.all(img[~mask_0_10, :] == [0, 0, 0], axis=1))
    area_0_10 = np.sum(mask_0_10)
    area_10_X = np.sum(~mask_0_10)

    img_0_10 = Image.fromarray(seg_0_10.astype(np.uint8))
    img_10_X = Image.fromarray(seg_10_X.astype(np.uint8))

    img_0_10.save(out1, compression="tiff_deflate")
    img_10_X.save(out2, compression="tiff_deflate")


    results =  [{
            'filename': os.path.basename(out1),
            'image_area_px2': area_0_10,
            'excluded_area_px2': black_0_10,
            'excluded_area_fraction': black_0_10 / area_0_10
            },
            {
            'filename': os.path.basename(out2),
            'image_area_px2': area_10_X,
            'excluded_area_px2': black_10_X,
            'excluded_area_fraction': black_10_X / area_10_X
            }
    ]
    
    return results


def get_checkpoints(args, image_stack):
    filtered_stack = []
    count = 0
    outpath = args.outpath
    for proc_path, seg_path, mask_path in image_stack:
        parts = os.path.basename(proc_path).split("_")
        n1 = '0_10cm'
        n2 = '10_Xcm'
        parts1 = parts.copy()
        parts2 = parts.copy()
        parts1.insert(2, n1)
        parts2.insert(2, n2)
        out1 = os.path.join(outpath, "_".join(parts1))
        out2 = os.path.join(outpath, "_".join(parts2))
        
        # Keep only if at least one split is missing
        if not (os.path.exists(out1) and os.path.exists(out2)):
            filtered_stack.append((proc_path, seg_path, mask_path))
        else:
            count += 1
    return filtered_stack, count



def main(args):
    """Main function"""
    img_stack = get_image_stack(args.path)
    N = len(img_stack)
    args.outpath = os.path.join(args.path, 'split_depths')
    os.makedirs(args.outpath, exist_ok=True)

    # Initialise results, load previous checkpoint
    outfile = os.path.join(args.outpath, "metrics.csv")
    if os.path.exists(outfile):
        df = pd.read_csv(outfile)
        df = df.dropna(how="all")
        img_stack, count = get_checkpoints(args, img_stack)
    else:
        # unclean way, but works
        df = pd.DataFrame(
            columns=['filename', 'image_area_px2', 'excluded_area_px2', 'excluded_area_fraction']
        )
        count = 0

    worker_args = [(paths, args.outpath) for paths in img_stack]
    num_workers =  5

    # Progressbar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green] Splitting Images:"), 
        BarColumn(), 
        MofNCompleteColumn(), 
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task(
            "[green]Splitting images...", 
            total=N,
            completed=count
            )

        # Asynchronously run worker processes
        with mp.Pool(processes=num_workers) as pool:
            # Consume Iterator -> note that the for loop repeatedly calls next(iterator) -> blocking
            for rows in pool.imap_unordered(worker_process, worker_args):
                for row in rows:
                    filename = row['filename']
                    if filename in df['filename'].values: # overwrite row
                        df.loc[df['filename'] == filename, :] = pd.DataFrame([row])
                    else: # concatenate row to the df
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                progress.update(task, advance=1)

    # export metrics
    df.to_csv(outfile, index=False)
    print(f"Metrics saved to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", 
        type=str, 
        help="path to the data basefolder containing subfolders with segmented images, processed images etc."
    )
    Image.MAX_IMAGE_PIXELS = 200000000 # Avoid DecompressionBombWarning from PIL
    args = parser.parse_args()

    main(args)