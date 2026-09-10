"""
Cross-evaluation between segmentation series. 
Compares a set of provided folders with images of the same name (comparison only considers alphabetical 
sorting of each image stack) and evaluates F1 score for each class and Cohen's kappa for the total.

To run, the scripts requires 'path' within which folders with annotations of the same image (same name)
are required. The script opens each folder and only sorts the images by name and then iterates through each
image and cross-evaluates through all folders.
"""

import os
import glob

import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score


def get_F1(TP, FP, FN):
    """Returns F1, based on TP, FP and FN"""
    return 2 * TP / (2 * TP + FN + FP)


def reformat_class_label(img, imgpath: str = None):
    """
    If label image is RGB, convert to grayscale with class integers
    """

    # If image has alpha channel overwrite the current image directly
    # Alpha channel won't significantly change the channel value, hence it works for only pure R, G, B colours.
    if np.ndim(img) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
        img = np.where(img > 200, 255, 0).astype(np.uint8)
        if imgpath is not None:
            img_new = Image.fromarray(img)
            img_new.save(saving_path, compression="tiff_deflate")

    if np.ndim(img) == 3:
        if img.shape[2] == 3:
            colors = {
                0: (0, 0, 0), # background
                1: (0, 0, 255), # blue = cares
                2: (255, 0, 0), # red = graminoids
                3: (0, 255, 0), # green = herbs
            }
            rgb_to_class = {v: k for k, v in colors.items()} 

            # Initialise and write class-map
            label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for rgb_val, c in rgb_to_class.items():
                mask = np.all(img == rgb_val, axis=-1)
                label[mask] = c

            return label

        elif img.shape[2] == 1: # If single-channel, squeeze channel dim (H, W, 1) -> (H, W)
            label = np.squeeze(img, 2)
            return label
    else:
        return img


def main(args):
    folders = [
        f for f in os.listdir(args.path)
        if os.path.isdir(os.path.join(args.path, f))
    ]
    N = len(folders)
    n = len(os.listdir(os.path.join(args.path, folders[0])))

    if args.mode == "binary":
        classes = ["fg"]   
        label_values = [1]
    else:
        classes = ["carex", "gram", "herb"]
        label_values = [1, 2, 3]

    # Gather all columns
    colnames = []

    # Initialise data array
    rownames = []
    rowlen = int(N*(N-1)/2)
    num_classes = len(label_values)
    data_array = np.zeros((rowlen, (n+1)*num_classes))
    rownr = 0
    kappas = []

    # Iterate through all folders
    for i in range(N):
        paths1 = sorted(glob.glob(os.path.join(args.path, folders[i], "*.png")))
        if i == 0: # get column names from imagenames
            basenames = [
                "_".join(os.path.basename(path).split("_")[:2])
                for path in paths1
            ]
            colnames.extend([f"{c}_{basenames[idx]}" for idx in range(n) for c in classes])
        # Cross-comparison of all folders
        for j in range(i+1, N):
            paths2 = sorted(glob.glob(os.path.join(args.path, folders[j], "*.png")))
            rownames.append(f"{folders[i]}_{folders[j]}")
            colnr = 0

            # Gather total stats over all images
            TPs = {str(l): 0 for l in label_values}
            FPs = {str(l): 0 for l in label_values}
            FNs = {str(l): 0 for l in label_values}
            imgs1 = []
            imgs2 = []

            for k in range(n): # iterate over all images
                img1 = np.array(Image.open(paths1[k]))
                img2 = np.array(Image.open(paths2[k]))

                img1 = reformat_class_label(img1, paths1[k])
                img2 = reformat_class_label(img2, paths2[k])

                if args.mode == "binary":
                    img1 = (img1 > 0).astype(np.uint8)
                    img2 = (img2 > 0).astype(np.uint8)

                imgs1.extend(img1.ravel())
                imgs2.extend(img2.ravel())

                for l in label_values: # Iterate over channels of each image
                    y1 = (img1 == l)
                    y2 = (img2 == l)

                    # NOTE: that F1 is symmetrical w.r.t order of img1 and img2
                    # As TP = A n B, FP = A\B, FN = B\A -- if images are switched, FP <-> FN which leaves F1 unchanged
                    TP = np.logical_and(y1, y2).sum()
                    FP = np.logical_and(y1, np.logical_not(y2)).sum()
                    FN = np.logical_and(np.logical_not(y1), y2).sum()

                    TPs[f"{l}"] += TP
                    FPs[f"{l}"] += FP
                    FNs[f"{l}"] += FN

                    f1_score = get_F1(TP, FP, FN)
                    data_array[rownr, colnr] = f1_score
                    colnr += 1
            for l in label_values:
                f1_tot = get_F1(TPs[f"{l}"], FPs[f"{l}"], FNs[f"{l}"])
                data_array[rownr, colnr] = f1_tot
                colnr += 1
            rownr += 1
            # Get Kappa score of the total batch
            if args.mode == "binary":
                kappa = cohen_kappa_score(imgs1, imgs2, labels=[0, 1])
            else:
                kappa = cohen_kappa_score(imgs1, imgs2, labels=[1, 2, 3])
            kappas.append(kappa)
    
    # create dataframe from data array
    colnames.extend([f"{c}_tot" for c in classes])
    df = pd.DataFrame(data=data_array, index=rownames, columns=colnames)
    df["kappa"] = kappas
    if args.mode == "multi":
        df["macro_f1"] = (df["carex_tot"] + df["gram_tot"] + df["herb_tot"]) / 3
    #df2 = df.transpose()
    print(df)
    df.to_excel(f"{args.path}/results.xlsx", sheet_name="Evaluation", index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--path", 
        type=str, 
        required=True, 
        help="Path to image folders to be analysed"
    )
    parser.add_argument(
        "--mode", 
        choices=["multi", "binary"], 
        default="multi"
    )
    args = parser.parse_args()

    main(args)





