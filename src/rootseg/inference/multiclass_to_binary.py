"""
This script splits multiclass images into binary images of each class. --requires the RGB image path -- 
Currently lacks the option to continue where left off, if not complete
Needs to adjust rich progress bar to match the others..

"""

import os

import numpy as np
import cv2
import argparse
from PIL import Image

from rootseg.inference.segment import get_image_paths
from rootseg.inference.evaluate import reformat_class_label

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

def main(args):
    path_info = get_image_paths(args.path)
    basepath_save = os.path.abspath(os.path.join(args.path, ".."))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        # Task 1: Main progress bar tracking image files
        image_task = progress.add_task(
            "[green]Processing images...", total=len(path_info)
        )
        for p in path_info:
            apath, rpath = p["full"], p["relative"]
            classes = ["Carex", "Graminoids", "Herbs"]
            img = np.array(Image.open(apath))
            #print('loaded image is', img.dtype, type(img), img.shape) #expect (H, W, C)
            if img.shape[2] != 3:
                raise ValueError(
                    f"Expected image with 3 channels, got shape {img.shape}"
                )
            for i in range(1, 4):
                img = reformat_class_label(img)
                c_img = (img == i).astype(np.uint8) * 255
                saving_path = os.path.join(basepath_save, f"binary_{classes[i-1]}")
                os.makedirs(saving_path, exist_ok=True)
                saving_path = os.path.join(saving_path, rpath)
                os.makedirs(os.path.dirname(saving_path), exist_ok=True)

                img_new = Image.fromarray(c_img)
                img_new.save(saving_path, compression="tiff_deflate")

            progress.advance(image_task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="splitting class images into binaries")

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="basepath to possible subfolders containing segmented multiclass images with" \
             "['Carex', 'Graminoid', 'Herbs']"
    )

    args = parser.parse_args()

    Image.MAX_IMAGE_PIXELS = 200000000 # Avoid DecompressionBombWarning from PIL

    main(args)

            









