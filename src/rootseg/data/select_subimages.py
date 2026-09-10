"""
Script to randomly select images from tubes to select random located subimages for 
creating training/ validation/ testing data for neural networks.
"""

import os
import random
import re
from typing import Tuple

import argparse
import numpy as np
import cv2

def select_subimages(image: np.ndarray, 
                     subimage_size: Tuple[int, int], 
                     n_tiles_per_img: int = 1,
                     coords: Tuple[int, int] = None
) -> np.ndarray:
    """
    Select n_tiles_per_img subimages with subimage_size as random crop or specified location.

    Args:
        image (np.ndarray): The input image.
        subimage_size (Tuple[int, int]): The size of the subimages to select.
        
        n_tiles_per_img (int): number of images selected from the input image
        coords (Tuple[int, int]): Location of the subimage, if None provided uses random crop

    Returns:
        np.ndarray: The selected subimages.
    """
    # select coordinates for each subimage, randomly selected if not explicitly provided
    height, width = image.shape[:2]
    sub_height, sub_width = subimage_size
    if coords is None:
        x_range = width - sub_width
        y_range = height - sub_height
        x_coords = np.random.randint(0, x_range, n_tiles_per_img)
        y_coords = np.random.randint(0, y_range, n_tiles_per_img)
    else:
        x_coords = list(coords[0])
        y_coords = list(coords[1])

    # Get list of subimages from the selected coords
    subimages = []
    for i in range(n_tiles_per_img):
        x = x_coords[i]
        y = y_coords[i]

        subimage = image[y:y+sub_height, x:x+sub_width]
        subimages.append(subimage)

    return subimages


def main(args):
    # Get shuffled list of tubes
    if args.tubes_list is None:
        tubes = os.listdir(args.folder)
    else:
        tubes is [t.upper() for t in args.tubes_list]
    random.shuffle(tubes) # random tube length
    save_path = os.path.abspath(os.path.join(args.folder, "..", "subimage_proposals"))
    os.makedirs(save_path, exist_ok=True)

    # Exclude tubes if specified
    if args.exclude_tubes is None:
        excl_tubes = []
    else:
        excl_tubes = [t.upper() for t in excl_tubes]

    # Iterate through tubes
    count = 0 # count saved images
    for tube in tubes:
        if tube in excl_tubes:
            print("skip tube", tube)
            continue

        # Select random images
        tube_folder = os.path.join(os.path.abspath(args.folder), tube)
        image_names = os.listdir(tube_folder)
        filtered_names = [name for name in image_names if name.endswith(".tiff")]
        random_names = np.random.choice(filtered_names, size=args.n_imgs_per_tube, replace=False)

        # Case of keeping fixed coordinates for the tube
        if args.fix_coords:
            height, width = (8784, 10200)
            sub_height, sub_width = args.subimage_size
            x_range = width - sub_width
            y_range = height - sub_height
            x_coords = np.random.randint(0, x_range, args.n_tiles_per_img)
            y_coords = np.random.randint(0, y_range, args.n_tiles_per_img)
            coords = (x_coords, y_coords)
        else:
            coords = None

        # Go through images
        for name in random_names:
            path = os.path.join(tube_folder, name)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
                
            subimages = select_subimages(img, args.subimage_size, n_tiles_per_img=args.n_tiles_per_img, coords=coords)
            for i, subimage in enumerate(subimages):
                save_name = name.replace(".tiff", "_subimg.png")
                if len(subimages) > 1:
                    save_name = save_name.replace(".png", f"_{i}.png")
                save_name = os.path.join(save_path, save_name)
                cv2.imwrite(save_name, subimage)
                count +=1
                if count >= args.total_images: # exit if reached total_images
                    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Select subimages from the raw data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--folder", 
        type=str,
        help="The folder containing the tubefolders to process."
    )
    argparser.add_argument(
        "--tubes_list",
        type=str, 
        nargs="+", 
        default=None,
    )
    argparser.add_argument(
        "--exclude_tubes",
        type=str,
        nargs="+",
        default=None,
    )
    argparser.add_argument(
        "--subimage_size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(2196, 2550),
        help="The size of the subimages to select (height width)"
    )
    argparser.add_argument(
        "--n_tiles_per_img", 
        type=int, 
        default=1,
        help="The number of subimages to select from each tube"
    )
    argparser.add_argument(
        "--n_imgs_per_tube", 
        type=int, 
        default=1,
        help="The number of subimages to select from each tube"
    )
    argparser.add_argument(
        "--total_images",
        type=int,
        default=20,
        help="Upper bound for the amount of images to be selected"
    )
    argparser.add_argument(
        "--fix_coords", 
        action="store_true", 
        default=False,
        help="If selected, takes the same coordinates for each image selected from a tube"
    )
    args = argparser.parse_args()

    main(args)