"""
This script allows to generate artificial overlays from single species image to create a training base for a
multiclass detection network to further classify roots. 
The script consists of two models merged into a single script:
    1) select_overlay_images:
        pre-selects image crops in a random order and pre-defined distribution within species for given categories,
        segments and filters them directly based on root occurences. Already distinguishes between 
        train/val/test from a provided excel sheet which needs to clarify which tube is used for which.
    2) generate_overlays:
        creates artificial overlays based on alpha blending on the pre-selected images.

Manual post-filtering is strongly advised, as the automated filtering only accounts for abundance 
of root but not overall root quality! 
"""

import os
import sys
import random
import glob
import shutil

import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
from datetime import datetime
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from rootseg.inference.segment import run_segmentation
from rootseg.preprocess.utils import get_date

class SpeciesMapper:
    """
    Mapper class to map ID <-> Species <-> (train/test) Tube(s)

    requires an excel file with columns:
        'species': each species as column
        'ID': integer ID that uniquely describes each class/ species
        'tubes': all tubes of each species per row
        'train_tubes': tube selection for training of each species
        'test_tube': test tube selection for validation/ testing of each species
    """
    def __init__(self, map_path):
        df = pd.read_excel(map_path)
        for col in ["tubes", "train_tubes"]:
            df[col] = df[col].apply(lambda x: [item.strip() for item in x.split(",") if item.strip()])
        self.df = df
    
    def tubes_from_species(self, species):
        return self.df[self.df.species == species]["tubes"]
    
    def species_from_tube(self, tube):
        for _, row in self.df.iterrows():
            if tube in row["tubes"]:
                return row["species"]
        
    def get_train_tubes(self, species):
        return self.df[self.df.species == species]["train_tubes"].item()
    
    def get_test_tube(self, species):
        """returns as list"""
        return self.df[self.df.species == species]["test_tube"].tolist()
    
    def get_unique_species(self):
        return self.df.species.unique()
    
    def species_from_id(self, species_id):
        return self.df[self.df.ID == species_id]["species"].item()
    
    def id_from_species(self, species_name):
        return self.df[self.df.species == species_name]["ID"].item()



def reset_folder(dirpath):
    pass
    for item in os.listdir(dirpath):
        item_path = os.path.join(dirpath, item)
        # Check if the item is a directory
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)


def get_images_of_tubes(base_dirs, tube_list):
    """Return a list of all image paths from the given tube subfolders."""
    image_paths = []
    for base_dir in base_dirs:
        for tube in tube_list:
            try:
                tube_folder = os.path.join(base_dir, tube)
                all_imgs = glob.glob(os.path.join(tube_folder, "*.tiff"))
                for img in all_imgs:
                    if get_date(img) >= datetime(2022, 1, 1):
                        image_paths.append(img)
                image_paths.extend(glob.glob(os.path.join(tube_folder, "*.tiff")))
            except Exception as e:
                continue
    random.shuffle(image_paths)
    return image_paths

def select_overlay_images(args):
    """
    main script to create a folder system with image selection to generate overlays with 
    'generate_artificial_overlays.py'.

    """

    # Select mode - only carex or functional groups are allowed
    if args.categories == "carex":
        working_path = os.path.join(args.savepath, "carex_only")
    elif args.categories == "classes":
        working_path = os.path.join(args.savepath, "classes")
    else:
        working_path = args.savepath
    os.makedirs(working_path, exist_ok=True)
    reset_folder(working_path)

    species_list = speciesmapper.get_unique_species()
    train_imgs = defaultdict(list)
    val_imgs = defaultdict(list)

    if args.categories == "species":
        species_list = species_list[:6] # don't allow non-replicates for species selection..

    # assort a testing tube for each species and put the rest for training
    for species in species_list:
        train_imgs[species] = get_images_of_tubes(args.basedirs, speciesmapper.get_train_tubes(species))
        val_imgs[species] = get_images_of_tubes(args.basedirs, speciesmapper.get_test_tube(species))
        print(50*"-")
        print(f"training images for {species}: {len(train_imgs[species])}")
        print(f"validation/testing  images for {species}: {len(val_imgs[species])}")

    species_idx = np.arange(1, 7)
    probs = np.array([1, 2, 3, 3, 2, 1]) / 12 # probabilities of selecting n images to overlay for the final image

    # Main loop
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Generating Image Samples..."), 
        BarColumn(), 
        MofNCompleteColumn(), 
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task("[green]Generating Image Samples...", total=args.N_images)
        # Create N images
        k, l, m = 0, 0, 0
        test_count = np.ceil(args.N_images * args.test_fraction) # max amount of test images
        for i in range(args.N_images):
            # Select train/val/test image --> 2/3 val 1/3 test of the total testing images
            # E.g. test_fraction of 30% results in 7:2:1 train-val-test split
            if i < test_count and i % 3 != 0: # validation image
                k += 1
                path = os.path.join(working_path, "val_imgs", f"stack_{k}")
                region = "upper"
            elif i < test_count and i % 3 == 0: # test image
                m += 1
                path = os.path.join(working_path, "test_imgs", f"stack_{m}")
                region = "lower"
            else: # train image
                l += 1
                path = os.path.join(working_path, "train_imgs", f"stack_{l}")
                region = "total"
            os.makedirs(path, exist_ok=True)

            # Image selection based on category choice
            if args.categories == "species":
                # sRandom Shuffle to select random species selection
                random.shuffle(species_list) 
                N_species = np.random.choice(species_idx, p=probs)
                selection_list = species_list[:N_species]
            elif args.categories == "carex": 
                # Ensure Carex is in each Selection and shuffle randomly between other species
                other_species = [s for s in species_list if s != "Carex"]
                random.shuffle(other_species)
                N_species = np.random.choice(np.arange(1, 4), p=np.array([3, 2, 1])/6)
                additional_species = other_species[:N_species]
                selection_list = ["Carex"] + additional_species
                random.shuffle(selection_list)
            elif args.categories == "classes": 
                # For train images, include non-replicates as well in a 1:6 proportion
                if args.non_replicates and "train_imgs" in path: 
                    herbs = ["Geum", "Potentilla", "Leontodon", "Ligusticum", "Sibbaldia", "Trifolium"]
                    herb_weights = [5, 5, 5, 1, 1, 1]
                    grams = ["Anthox", "Helictotrichon", "Agrostis", "Nardus", "Poa"]
                    gram_weights = [5, 5, 1, 1, 1]
                    selection_list = ["Carex"]
                    num_herbs = np.random.choice(np.arange(1,4), p=np.array([0.7, 0.25, 0.05])) # select 1-3 herbs
                    num_grams = np.random.choice(np.arange(1,3), p=np.array([0.7, 0.3])) #select 1-2 graminoids

                    selected_herbs = random.choices(herbs, weights=herb_weights, k=num_herbs)
                    selected_grams = random.choices(grams, weights=gram_weights, k=num_grams)

                    selection_list.extend(selected_herbs)
                    selection_list.extend(selected_grams)
                    random.shuffle(selection_list)

                else: # Only select replicated species for val/test
                    herbs = ["Geum", "Potentilla", "Leontodon"]
                    grams = ["Anthox", "Helictotrichon"]
                    selection_list = ["Carex"]
                    num_herbs = np.random.choice(np.arange(1,4), p=np.array([0.7, 0.25, 0.05]))
                    num_grams = np.random.choice(np.arange(1,3), p=np.array([0.7, 0.3]))
                    selected_herbs = random.sample(herbs, num_herbs)
                    selected_grams = random.sample(grams, num_grams)

                    selection_list.extend(selected_herbs)
                    selection_list.extend(selected_grams)
                    random.shuffle(selection_list)    

            # iterate through the randomly shuffled selected species to randomly select image crops for each species
            # for overlays
            for j, current_species in enumerate(selection_list):
                if "train_imgs" in path:
                    # Select random train img of given species
                    img_path = train_imgs[current_species].pop()
                    img_name = f"{j}_{os.path.basename(img_path)}".replace(".tiff", ".png")
                    img = np.array(Image.open(img_path))

                    # Random crop the selected image
                    x = np.random.randint(0, img.shape[1]-2550)
                    y = np.random.randint(0, img.shape[0]-2196)
                    subimg = Image.fromarray(img[y:y+2196, x:x+2550, :])
                    subimg.save(os.path.join(path, img_name))
                else:
                    # Select random val image of given species
                    img_path = val_imgs[current_species].pop()
                    img_name = f"{j}_{os.path.basename(img_path)}".replace(".tiff", ".png")
                    img = np.array(Image.open(img_path))

                    # Random crop from 0-180° for validation image and 180-360° of the image for train image
                    x = np.random.randint(0, img.shape[1]-2550)
                    if region == "upper": # validation image
                        y = np.random.randint(0, img.shape[0]//2)
                    elif region == "lower": # testing image
                        y = np.random.randint(img.shape[0]//2, img.shape[0]-2196)
                    subimg = Image.fromarray(img[y:y+2196, x:x+2550, :])
                    subimg.save(os.path.join(path, img_name))
            progress.update(task, advance=1)

    # In case of a provided model, run the segmentation directly as well
    if args.segmentation_model:
        run_segmentation(
            data_path=working_path, 
            model_path=args.segmentation_model, 
            segmentation_path=os.path.join(working_path, "segmented"),
            class_selection="roots",
            batch_size=4,
            num_workers=2,
            filter_components=True
        )
        # filter "bad images" based on how many roots occur in each
        filter_bad_imgs(working_path)


def filter_bad_imgs(working_path):
    """
    Iterates through all folders of selected images for overlays and discards the whole overlay folder
    if an image within it has a root fractio <0.2% 
    (empirically this corresponds to no significant roots/ mostly noise)
    """
    count = 0
    for dirpath, _, filenames in os.walk(os.path.abspath(working_path)):
        if "segmented" in dirpath.split(os.sep):
            continue
        relative_path = os.path.relpath(dirpath, working_path)
        segmented_path = os.path.join(working_path, "segmented", relative_path)
        img_path = os.path.join(working_path, relative_path)
        for file in filenames:
            if not file.endswith(".png"):
                continue
            seg = cv2.imread(os.path.join(segmented_path, file), cv2.IMREAD_GRAYSCALE)
            # decide on a pixel threshold below which the image won't be taken for train/test
            if np.sum(seg>0)/np.sum(seg==0) < 0.002: # manually set fraction of roots required to count
                count += 1
                if os.path.exists(segmented_path):
                    shutil.rmtree(segmented_path)
                if os.path.exists(img_path):
                    shutil.rmtree(img_path)
                break # abort this folder
    train_path = os.path.join(working_path, "train_imgs")
    val_path = os.path.join(working_path, "val_imgs")
    test_path = os.path.join(working_path, "test_imgs")
    print(50*"-")
    print("Image selection filter summary:")
    print(f"Removed {count} folders in total due to missing roots in random cropped images")
    print(f"Remaining amount of train images: {len(os.listdir(train_path))}")
    print(f"Remaining amount of val images: {len(os.listdir(val_path))}")
    print(f"Remaining amount of test images: {len(os.listdir(test_path))}")    
    print(50*"-")


def plot_multiclass_segmentation(seg_map: np.ndarray, save_path: str, categories):
    """
    Transforms a single-channel segmentation map (0-6) into a colored image,
    plots it with a legend, and saves the figure.

    Args:
        seg_map (np.ndarray): The 2D array of class indices (H, W), with values 0 to 6.
        save_path (str): The full path to save the output image (e.g., 'colored_seg.png').
        class_names (list, optional): List of class names corresponding to indices 0-6.
                                      Defaults to ['Class 0', ..., 'Class 6'].
    """

    # Define default class names if none are provided
    if categories == "carex":
        class_names = [
            "Soil",
            "Carex",
            "Non-Carex"
        ]
        class_colors = ["tab:blue", "tab:brown", "tab:green"]
        cmap = plt.matplotlib.colors.ListedColormap(class_colors)

    elif categories == "species":
        class_names = [
            "Soil",
            "Anthox",
            "Geum",
            "Carex",
            "Leontodon",
            "Potentilla",
            "Helictotrichon"
        ]
        cmap = plt.get_cmap("tab10", 7)
    else:
        class_names = [
            "Soil",
            "Carex",
            "Graminoids",
            "Herbs"
        ]
        class_colors = ["tab:blue", "tab:brown", "tab:orange", "tab:green"]
        cmap = plt.matplotlib.colors.ListedColormap(class_colors)
    num_classes = len(class_names)
    
    # Ensure class_names matches the number of classes
    if len(class_names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, but got {len(class_names)}.")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    img_plot = ax.imshow(seg_map, cmap=cmap, vmin=-0.5, vmax=num_classes - 0.5, interpolation="none")
    
    ax.set_title("Multiclass Segmentation Map")
    ax.axis("off")

    # Add custom legend
    patches = []
    for i in range(num_classes):
        color = cmap(i)
        patches.append(mpatches.Patch(color=color, label=class_names[i]))
    ax.legend(handles=patches, 
              title="Classes", 
              loc="center left", 
              bbox_to_anchor=(1.05, 0.5),
              frameon=False)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def get_tube(filename:str) -> str:
    """
    Extracts tube from filename
    requires the following structure: basepath -> structure {number}_{name}_{tube}_{level}_{date}_...
    """
    filename = os.path.basename(filename)
    tube = filename.split("_")[2]
    return tube

def alpha_blending(img_base: np.ndarray, img_overlay: np.ndarray, overlay_mask: np.ndarray) -> np.ndarray:
    """
    Perform Alpha blending to overlay img_overlay onto img_base with overlay mask with smoothing

    Args:
        img_base (np.ndarray): background image
        img_overlay (np.ndarray): image to overlay
        overlay_mask (np.ndarray): binary root mask for overlay positions

    Returns:
        np.ndarray: RGB image with img_base and blended roots from img_overlay  
    """
    overlay_mask = overlay_mask.copy()

    # Ensure binary uint8 mask with values in {0, 255}
    if overlay_mask.dtype != np.uint8:
        overlay_mask = overlay_mask.astype(np.uint8) * 255

    # Use gaussian kernel to smoothen the transition and overlay
    overlay_mask = (overlay_mask.astype(float) - np.min(overlay_mask)) / (np.max(overlay_mask) - np.min(overlay_mask))
    overlay_mask = cv2.GaussianBlur(overlay_mask, (5, 5), 4)
    img_base = img_base.astype(float)
    img_overlay = img_overlay.astype(float)

    overlay_mask = np.stack([overlay_mask] * 3, axis=-1)
    foreground = cv2.multiply(overlay_mask, img_overlay)
    background = cv2.multiply(1.0 - overlay_mask, img_base)

    out_image = cv2.add(foreground, background)
    return out_image

def create_overlay(imgpath: str, segpath: str, speciesmapper: SpeciesMapper, categories:bool = "species"):
    """
    Takes a stack folder of .png images and creates an overlay based on Alpha blending
    
    Args:
        imgpath (str): folderpath to the folder with all overlay images
        segpath (str): folderpath to the folder with all segmentations of the overlay images
        speciesmapper (SpeciesMapper): Instance of the mapper class with species and tube information
        categories (bool): operating mode - either 'species', 'carex' or 'classes'
    """
    # Get list of .png file paths ordered
    imglist = sorted([f for f in os.listdir(imgpath) if f.endswith(".png")])
    seglist = sorted([f for f in os.listdir(segpath) if f.endswith(".png")])

    # Iterate through the folder and create the overlay
    for i, (imgname, segname) in enumerate(zip(imglist, seglist)):
        # Extract species and id information for the current image
        species = speciesmapper.species_from_tube(get_tube(imgname))
        sp_id = speciesmapper.id_from_species(species)

        # Select mapping based on operational mode
        if categories.lower() == "carex": # Carex only case
            classes_map = {
                3: 1, # Carex -> 1
                1: 2, 4: 2, 5: 2, 6: 2 # Non-Carex -> 2
            }
        elif categories.lower() == "classes":
            classes_map = {
                3: 1, # Carex -> 1
                1: 2, 6: 2, 8: 2, 9: 2, 12: 2, # Graminoids -> 2
                2: 3, 4: 3, 5: 3, 7: 3, 10: 3, 11: 3 # Herbs -> 3
            }
        else:
            classes_map = {
                1: 1, # identity map for species level
                2: 2, 
                3: 3, 
                4: 4, 
                5: 5,
                6: 6 
            }
        sp_id = classes_map[sp_id]
        if i == 0: # create overlay base -- first image
            overlay = cv2.imread(os.path.join(imgpath, imgname), cv2.IMREAD_COLOR)
            seg = cv2.imread(os.path.join(segpath, segname), cv2.IMREAD_GRAYSCALE)
            mask = (seg > 0).astype(np.int8)
            seg_overlay = mask * sp_id
            black_mask_3d = (overlay == [0, 0, 0])
            black_mask = np.all(black_mask_3d, axis=2)

        else: # overlay all images i > 0 to the base image
            img = cv2.imread(os.path.join(imgpath, imgname), cv2.IMREAD_COLOR)
            seg = cv2.imread(os.path.join(segpath, segname), cv2.IMREAD_GRAYSCALE)
            mask = (seg > 0)
            seg_sp = mask.astype(np.int8) * sp_id
            overlay = alpha_blending(overlay, img, mask)
            seg_overlay[mask] = seg_sp[mask]

    overlay[black_mask] = [0, 0, 0]
    seg_overlay[black_mask] = 0

    return overlay, seg_overlay




def generate_overlays(args):
    """
    Based on the folder system generated by 'select_overlay_images' use alpha blending to overlay
    all images within an overlay folder and create the corresponding label image
    """
    # select operational mode
    if args.categories == "carex":
        working_path = os.path.join(args.savepath, "carex_only")
    elif args.categories == "classes":
        working_path = os.path.join(args.savepath, "classes")
    else:
        working_path = args.savepath

    # Load all stacks of train/val/test folders
    train_stacks = sorted(os.listdir(os.path.join(working_path, "train_imgs")))
    val_stacks =  sorted(os.listdir(os.path.join(working_path, "val_imgs")))
    test_stacks =  sorted(os.listdir(os.path.join(working_path, "test_imgs")))
    N = len(train_stacks) + len(val_stacks) + len(test_stacks)
    speciesmapper = SpeciesMapper(os.path.join(args.savepath, "speciesmap.xlsx"))

    # Create all output directories
    outdir1_tr = os.path.join(os.path.join(working_path), "train_n", "images")
    outdir2_tr = os.path.join(os.path.join(working_path), "train_n", "annotations")
    vis_dir_tr = os.path.join(os.path.join(working_path), "train_n", "visual")

    outdir1_v = os.path.join(os.path.join(working_path), "val_n", "images")
    outdir2_v = os.path.join(os.path.join(working_path), "val_n", "annotations")
    vis_dir_v = os.path.join(os.path.join(working_path), "val_n", "visual")

    outdir1_te = os.path.join(os.path.join(working_path), "test_n", "images")
    outdir2_te = os.path.join(os.path.join(working_path), "test_n", "annotations")
    vis_dir_te = os.path.join(os.path.join(working_path), "test_n", "visual")

    if args.root_map: # If provided also generates a binary rootmap for each overlay
        seg_path_tr = os.path.join(os.path.join(working_path), "train_n", "binary")
        seg_path_v = os.path.join(os.path.join(working_path), "val_n", "binary")
        seg_path_te = os.path.join(os.path.join(working_path), "test_n", "binary")
        os.makedirs(seg_path_tr, exist_ok=True)
        os.makedirs(seg_path_te, exist_ok=True)
        os.makedirs(seg_path_v, exist_ok=True)
    
    os.makedirs(outdir1_tr, exist_ok=True)
    os.makedirs(outdir2_tr, exist_ok=True)
    os.makedirs(vis_dir_tr, exist_ok=True)

    os.makedirs(outdir1_te, exist_ok=True)
    os.makedirs(outdir2_te, exist_ok=True)
    os.makedirs(vis_dir_te, exist_ok=True)

    os.makedirs(outdir1_v, exist_ok=True)
    os.makedirs(outdir2_v, exist_ok=True)
    os.makedirs(vis_dir_v, exist_ok=True)
    
    # main generation loop
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold red]Generating artificial overlays"), 
        BarColumn(), 
        MofNCompleteColumn(), 
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task("[red]Generating artificial overlays", total=N)
        for stack in train_stacks:
            stack_id = stack.split("_")[-1]
            trainpath = os.path.join(working_path, "train_imgs", stack)
            trainpath_seg = os.path.join(working_path, "segmented", "train_imgs", stack)

            overlay, seg_overlay = create_overlay(
                trainpath, trainpath_seg, speciesmapper, categories=args.categories
            )

            cv2.imwrite(os.path.join(outdir1_tr, f"{stack_id}_stack.png"), overlay)
            cv2.imwrite(os.path.join(outdir2_tr, f"{stack_id}_stack.png"), seg_overlay)
            vis_path = os.path.join(vis_dir_tr, f"{stack_id}_stack.png")
            plot_multiclass_segmentation(seg_overlay, vis_path, args.categories)
            if args.root_map:
                seg = (seg_overlay > 0).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(seg_path_tr, f"{stack_id}_stack.png"), seg)

            progress.update(task, advance=1)

        for stack in val_stacks:
            stack_id = stack.split("_")[-1]
            valpath = os.path.join(working_path, "val_imgs", stack)
            valpath_seg = os.path.join(working_path, "segmented", "val_imgs", stack)
            overlay, seg_overlay = create_overlay(valpath, valpath_seg, speciesmapper, args.categories)
            cv2.imwrite(os.path.join(outdir1_v, f"{stack_id}_stack.png"), overlay)
            cv2.imwrite(os.path.join(outdir2_v, f"{stack_id}_stack.png"), seg_overlay)
            vis_path = os.path.join(vis_dir_v, f"{stack_id}_stack.png")
            plot_multiclass_segmentation(seg_overlay, vis_path, args.categories)
            if args.root_map:
                seg = (seg_overlay > 0).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(seg_path_v, f"{stack_id}_stack.png"), seg)
            progress.update(task, advance=1)
            
        for stack in test_stacks:
            stack_id = stack.split("_")[-1]
            testpath = os.path.join(working_path, "test_imgs", stack)
            testpath_seg = os.path.join(working_path, "segmented", "test_imgs", stack)
            overlay, seg_overlay = create_overlay(testpath, testpath_seg, speciesmapper, args.categories)
            cv2.imwrite(os.path.join(outdir1_te, f"{stack_id}_stack.png"), overlay)
            cv2.imwrite(os.path.join(outdir2_te, f"{stack_id}_stack.png"), seg_overlay)
            vis_path = os.path.join(vis_dir_te, f"{stack_id}_stack.png")
            plot_multiclass_segmentation(seg_overlay, vis_path, args.categories)
            if args.root_map:
                seg = (seg_overlay > 0).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(seg_path_te, f"{stack_id}_stack.png"), seg)
            progress.update(task, advance=1)
        

    


        
    
if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = 200000000 # Avoid DecompressionBombWarning from PIL

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--basedirs",
        nargs="+",
        required=True,
        help="List of base directories from which random images will be selected for overlay creation" \
             "subfolders will be considered for each base directory"
    )
    parser.add_argument(
        "--N_images",
        type=int,
        default=250,
        help="upper limit of how many artificial images in total to consider" \
             "NOTE: the resulting number will usually be less, as this is the limit considered before filtering"
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=.3,
        help="fraction of val/test data compared to train data. val/test will always be subdivided in 2:1 ratio"
    )
    parser.add_argument(
        "--non_replicates",
        action="store_true",
        help="if selected, included non-replicate species to training pool as well"
    )
    parser.add_argument(
        "--savepath",
        type=str,
        help="Saving path"
    )
    parser.add_argument(
        "--categories",
        choices=["species", "classes", "carex"],
        required=True,
        help="Select between species, classes, and carex"
    )
    parser.add_argument(
        "--root_map",
        action="store_true",
        default=None,
        help="if selected, stores binary rootmap of the final image as well"
    )
    args = parser.parse_args()

    speciesmapper = SpeciesMapper(os.path.join(args.savepath, "speciesmap.xlsx"))

    # step 1) creates foldersystem with selected images for overlays inc. cleaning
    select_overlay_images(args)

    # step 2) generates overlays using alpha blending based on the created foldersystem and pre-selection.
    generate_overlays(args)







    
