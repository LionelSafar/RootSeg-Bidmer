"""
A modified version of split_depths.py in case no splitting is performed - also tracks image size as .csv
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

def get_image_stack(path: str) -> List[Tuple[str, str, str]]:
    """Return a list of tuples (processed_paths, segmented_paths) for a project folder"""
    processed_path = os.path.join(path, 'preprocessed')
    segmented_path = os.path.join(path, 'segmented')

    processed_paths = glob.glob(os.path.join(processed_path, '**', '*.tiff'), recursive=True)
    processed_paths.extend(glob.glob(os.path.join(processed_path, '**', '*.png'), recursive=True))

    segmented_paths = glob.glob(os.path.join(segmented_path, '**', '*.tiff'), recursive=True)
    segmented_paths.extend(glob.glob(os.path.join(segmented_path, '**', '*.png'), recursive=True))

    processed_paths.sort(key=os.path.basename)
    segmented_paths.sort(key=os.path.basename)

    image_stack = []
    assert len(processed_paths) == len(segmented_paths), f"{len(processed_paths)} and {len(segmented_paths)}"
    for p_p, p_s in zip(processed_paths, segmented_paths):
        image_stack.append((p_p, p_s))
    return image_stack


def worker_process(path) -> Dict:
    """
    Main process for each worker.
    Read original image, segmentation and depth mask, 
    split segmentation according to depth mask and extract masked area (e.g. tape) from original image.
    Save new segmentations and return entry to dataframe with image name and area information.
    """
    img = np.array(Image.open(path[0][0]))
    seg = cv2.imread(path[0][1], cv2.IMREAD_GRAYSCALE)
    seg = np.array(seg)
    outpath = path[1]

    out = os.path.join(outpath, os.path.basename(path[0][0]))

    H, W = img.shape[:2]
    if seg.shape[:2] != (H, W):
        print(f'WARNING DIFFERENT SHAPE OF segmented image {path[0][1]} and im {path[0][0]} -- SKIPPED!')
        print(f'shapes are {img.shape} for img and {seg.shape} -- return NaNs')
        return [{
            'filename': os.path.basename(out),
            'image_area_px2': np.nan,
            'excluded_area_px2': np.nan,
            'excluded_area_fraction': np.nan
            }]

    # Get image area per depth and masked (black) area per depth
    if '_T1_' in os.path.basename(out): 
        #NOTE: T1 we neglect the area loss due to a masked out Agrostis root, as it should barely 
        # affects any other root but the mask includes a significant area of soil
        cutoff = 3 * W // 4
        img_section = img[:, cutoff:, :]
        black_area = np.sum(np.all(img_section == [0, 0, 0], axis=2))
    else:
        black_area = np.sum(np.all(img == [0, 0, 0], axis=2))
    area_img = H * W
    img_new = Image.fromarray(seg.astype(np.uint8))
    img_new.save(out, compression="tiff_deflate")

    results =  [{
            'filename': os.path.basename(out),
            'image_area_px2': area_img,
            'excluded_area_px2': black_area,
            'excluded_area_fraction': black_area / area_img
            }]
    
    return results


def get_checkpoints(args, image_stack):
    filtered_stack = []
    count = 0
    outpath = args.outpath
    for proc_path, seg_path in image_stack:
        out = os.path.join(outpath, os.path.basename(proc_path))

        # Keep only if at least one split is missing
        if not os.path.exists(out):
            filtered_stack.append((proc_path, seg_path))
        else:
            count += 1
    return filtered_stack, count


def main(args):
    """Main file"""
    img_stack = get_image_stack(args.path)
    N = len(img_stack)
    print('N', N)
    args.outpath = os.path.join(args.path, 'segmented_flattened')
    os.makedirs(args.outpath, exist_ok=True)

    # Initialise results, load previous checkpoint
    outfile = os.path.join(args.outpath, "metrics.csv")
    if os.path.exists(outfile):
        df = pd.read_csv(outfile)
        df = df.dropna(how="all")
        img_stack, count = get_checkpoints(args, img_stack)
    else:
        df = pd.DataFrame(
            columns=['filename', 'image_area_px2', 'excluded_area_px2', 'excluded_area_fraction']
            )
        count = 0

    worker_args = [(paths, args.outpath) for paths in img_stack]
    num_workers =  5

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
                    if filename in df['filename'].values:
                        df.loc[df['filename'] == filename, :] = pd.DataFrame([row])
                    else:
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                progress.update(task, advance=1)

    # Create results dataframe and save metrics
    df.to_csv(outfile, index=False)
    print(f"Metrics saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to the data basefolder")

    Image.MAX_IMAGE_PIXELS = 200000000 # Avoid DecompressionBombWarning from PIL

    args = parser.parse_args()

    main(args)
