"""
Image segmentation pipeline. Allows binary and multiclass segmentation.

"""
import os
import math
import queue
import threading
import gc

import argparse
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Any, List, Iterator
import itertools
import torch
from torch.utils.data import DataLoader, IterableDataset
from skimage import measure
from torchvision.transforms import ToTensor

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from rootseg.training.models import UNet, SwinT_UNet, SwinB_UNet
from rootseg.inference.visualisations import visualize_multiclass, vis_seg_rgb


def get_image_paths(base_path: str) -> List[Dict[str, str]]:
    """
    Loads all images in tiff or png formate from a nested folder structure.

    Args:
        base_path (str): Base folder that contains all potential subfolders with image files

    Returns:
        List[Dict[str, str]]: List of Dicts with full and relative path of each image
    """
    paths = []
    for dirpath, _, filenames in os.walk(base_path):
        for filename in filenames:
            if filename.lower().endswith(('.tif', '.tiff', '.png')):
                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_path, base_path)
                paths.append({'full': full_path, 'relative': relative_path})
    return paths


class TiledImageDataset(IterableDataset):
    """
    Iterable torch dataset to load large images and yields image tiles at a given stride to segment and 
    reassemble later on.

    Args:
        imagepaths (list): list of dict with full and relative image paths (as from get_image_paths()) 
        segmentation_paths (list): list of dict with full and relative image paths of binary segmented images
            - this is used for multiclass segmentation, if not provided defaults to binary mode.
        savepath (str): path where segmentations will be saved
        tilesize (int): size of subtiles for the segmentation network
        stepsize (int): strides over the large image, requires at least the size of the NN output
        pad (int): padding for the NN output. output size should be tilesize - 2*pad
    """
    def __init__(
        self, 
        imagepaths: List[Dict[str, str]], 
        segmentation_paths: List[Dict[str, str]], 
        savepath: str, 
        tilesize: int=512, 
        stepsize: int=388, 
        pad: int=None
    ):
        super().__init__()
        self.imagepaths = imagepaths
        self.tilesize = tilesize
        self.pad = pad
        self.outsize = tilesize - (2 * pad)
        self.savepath = savepath
        self.stepsize=stepsize
        self.seg = segmentation_paths

    def _preprocess_tile(self, tile: np.ndarray) -> torch.Tensor:
        """Convert np array to torch format and normalize to [0, 1]"""
        tile = torch.from_numpy(tile).to(torch.float32)
        img_min = tile.min()
        img_max = tile.max()
        normalized_image = (tile - img_min) / (img_max - img_min)
        return normalized_image.permute(2, 0, 1)

    def _image_exist(self, path_info: Dict[str, str]) -> bool:
        """Check if a segmented image already exists to an input image"""
        savename = path_info['relative'].replace('.tiff', '_segmented.png')
        if os.path.exists(os.path.join(self.savepath, savename)):
            return True
        else:
            return False

    def _generate_tiles(self) -> Iterator[Dict[str, Any]]:
        """
        Generator that loads large images, pads them and yields tiles.
        Returns a Dict containing:
            - tile
            - coords (tile coordinates in large images)
            - image_info (Dict with:)
                -- relative_path (relative path of the large image)
                -- original_shape (shape of large image to restore later)
                -- total_tiles (# tiles of the image to track reconstruction)
        """
        for path_info in self.imagepaths:
            # skip image if segmentation exists already
            if self._image_exist(path_info):
                continue
            try:
                with Image.open(path_info["full"]) as img:
                    img = np.array(img)
            except (OSError, ValueError) as e:
                raise e
            
            # pad large image to account for borders during segmentation
            orig_h, orig_w = img.shape[0], img.shape[1]
            h_tiles = math.ceil(orig_h / self.stepsize)
            w_tiles = math.ceil(orig_w / self.stepsize)

            # padding is calculated by putting n_tiles-1 images with stepsize n_step
            # but considering that the img might have a different outsize - replace last stepsize with actual imagesize
            # and take the difference w.r.t. to the current image size
            pad_h = (h_tiles-1) * self.stepsize + self.outsize - orig_h
            pad_w = (w_tiles-1) * self.stepsize + self.outsize - orig_w
            img_padded = np.pad(
                img,
                ((self.pad, self.pad + pad_h), (self.pad, self.pad + pad_w), (0, 0)),
                mode="reflect"
            )
            total_tiles = h_tiles * w_tiles
            
            # yield tiles with coords and image information for reassembly
            for y in range(0, h_tiles * self.stepsize, self.stepsize):
                for x in range(0, w_tiles * self.stepsize, self.stepsize):
                    tile = img_padded[y:y+self.tilesize, x:x+self.tilesize, :]
                    yield {
                        'tile': self._preprocess_tile(tile),
                        'coords': torch.tensor([x, y]),
                        'image_info': {
                            'relative_path': path_info['relative'],
                            'original_shape': torch.tensor([orig_h, orig_w]),
                            'total_tiles': total_tiles,
                        }
                    }

    def _generate_tiles_with_seg(self) -> Iterator[Dict[str, Any]]:
        """
        modified generator that includes segmentation path - Required for multiclass segmentation
        in the Reconstructor
        """
        for path_info, seg_info in zip(self.imagepaths, self.seg):
            # Skip image if segmentation exists already
            if self._image_exist(path_info):
                continue
            try:
                with Image.open(path_info["full"]) as img:
                    img = np.array(img)
                segpath = seg_info["full"]
            except (OSError, ValueError) as e:
                raise e
            
            # Pad large image to account for borders during segmentation
            orig_h, orig_w = img.shape[0], img.shape[1]
            h_tiles = math.ceil(orig_h / self.outsize)
            w_tiles = math.ceil(orig_w / self.outsize)

            pad_h = h_tiles * self.outsize - orig_h
            pad_w = w_tiles * self.outsize - orig_w
            img_padded = np.pad(
                img,
                ((self.pad, self.pad + pad_h), (self.pad, self.pad + pad_w), (0, 0)),
                mode="reflect"
            )
            total_tiles = h_tiles * w_tiles
            
            # Yield tiles with coords and image information for reassembly
            for y in range(0, h_tiles * self.outsize, self.outsize):
                for x in range(0, w_tiles * self.outsize, self.outsize):
                    tile = img_padded[y:y+self.tilesize, x:x+self.tilesize, :]
                    yield {
                        'tile': self._preprocess_tile(tile),
                        'coords': torch.tensor([x, y]),
                        'image_info': {
                            'relative_path': path_info['relative'],
                            'original_shape': torch.tensor([orig_h, orig_w]),
                            'total_tiles': total_tiles,
                            'segmentation_path': segpath,
                        }
                    }


    def __iter__(self):
        """
        Allow partaging tile stream to multiple workers.
        For N workers, each worker i in N processes tiles j where j % N = i
        """
        worker_info = torch.utils.data.get_worker_info()
        if self.seg is None:
            tile_generator = self._generate_tiles()
        else:
            tile_generator = self._generate_tiles_with_seg()
        if worker_info is None: # Single-process
            return tile_generator
        else: # Multi-process
            return itertools.islice(tile_generator, worker_info.id, None, worker_info.num_workers)
        

class Reconstructor(threading.Thread):
    """
    CPU thread that reconstructs large segmented images from a result queue, containing the segmented
    tiles and reassembly information.

    Args:
        result_queue (queue.Queue[tuple]): Thread-safe queue containing segmentation results
        save_dir (str): saving path
        pad (int): padding between input tile and model output
        stepsize (int): stepsize used during the large image tiling step
        imgsize (int): tile resolution
        progress (rich.progress.Progress): Progress manager to update progress bar
        task_id (int): progress task id from rich
        class_names (list): List of class-names for (multiclass)-segmentation
        filter_components (bool): If true, filters connected components <250px - slow for large images!
    """
    def __init__(
        self, 
        result_queue: queue.Queue[tuple], 
        save_dir: str, 
        pad: int=(512-388)//2, 
        stepsize: int=388, 
        imgsize: int=512, 
        progress: rich.progress.Progress=None, 
        task_id: int=None, 
        class_names: list=None,
        filter_components: bool=False
    ):
        super().__init__()
        self.queue = result_queue
        self.save_dir = save_dir
        self.class_names = class_names
        if self.class_names:
            self.visdir = os.path.join(save_dir, "..", "visualization")
            os.makedirs(self.visdir, exist_ok=True)
        self.imgsize = imgsize
        self.pad = pad
        self.stepsize = stepsize
        self.canvases = {}
        self.segmentations = {}
        self.to_tensor = ToTensor()
        self.overlap = (self.imgsize - 2*self.pad != self.stepsize)
        if self.overlap: # overlapping images case
            self.count_canvases = {}
        self.filter = filter_components

        # Track progress
        self.progress = progress
        self.task_id = task_id

    def run(self):
        """
        Runs the thread indefinitely, blocking when queue is empty. The thread stops only when 
        item = None is manually put to the queue.
        """
        while True:
            item = self.queue.get(block=True) 
            if item is None: 
                break

            # Unpack item on CPU
            pred_tile, coords, info = item
            pred_tile_cpu = pred_tile.to('cpu')
            rel_path = info['relative_path']
            orig_shape = info['original_shape']
            total_tiles = info['total_tiles']

            # If this is the first tile for this image, create a new canvas
            if rel_path not in self.canvases:
                canvas_h = orig_shape[0] + self.pad * 2 + self.imgsize
                canvas_w = orig_shape[1] + self.pad * 2 + self.imgsize
                if self.overlap: # If overlap - count-canvas will track which pixels occur twice to average later
                    self.count_canvases[rel_path] = torch.zeros((1, canvas_h, canvas_w), dtype=torch.float32)
                    self.canvases[rel_path] = {
                        'canvas': torch.zeros((pred_tile.shape[0], canvas_h, canvas_w), dtype=torch.float32), 
                        'count': 0 # count towards finishing the canvas
                    }
                    with Image.open(info["segmentation_path"]) as seg:
                        seg = self.to_tensor(seg)
                        self.segmentations[rel_path] = seg
                else:
                    self.canvases[rel_path] = {
                        'canvas': torch.zeros((1, canvas_h, canvas_w), dtype=torch.float32),
                        'count': 0 # count == total_tiles indicates full reconstruction
                    }
            # Get size and position of the tile and insert to canvas
            out_size = pred_tile_cpu.shape[-1]
            x, y = coords[0], coords[1]
            try:
                self.canvases[rel_path]['canvas'][:, y:y+out_size, x:x+out_size] = pred_tile_cpu
                if self.overlap:
                    self.count_canvases[rel_path][:, y:y+out_size, x:x+out_size] += 1
            except Exception as e:
                raise e
            self.canvases[rel_path]['count'] += 1 # increase counter of inserted tiles

            if self.canvases[rel_path]['count'] == total_tiles: # finished canvas
                if self.overlap:
                    prob_map = self.canvases[rel_path]['canvas'] / self.count_canvases[rel_path] # average prob
                    segmap = (self.segmentations[rel_path] > 0).to(torch.int8)
                    predicted = torch.argmax(prob_map, dim=0).to(torch.int8).unsqueeze(0)
                    predicted += 1 # reserve 0-class for background
                    predicted = predicted[:, :orig_shape[0], :orig_shape[1]]
                    final_mask = predicted * segmap # set non-roots to 0 - background
                else:
                    predicted = (self.canvases[rel_path]['canvas'] > 0.5)
                    final_mask = predicted.to(torch.uint8) * 255
                    final_mask = final_mask[:, :orig_shape[0], :orig_shape[1]]

                # Remove small components if requested
                if self.filter:
                    final_mask = filter_small_components(final_mask)
                else:
                    final_mask = final_mask.numpy().astype(np.uint8)


                # Construct save path
                save_name = os.path.splitext(os.path.basename(rel_path))[0] + "_segmented.png"
                save_name = rel_path.replace('.tiff', '_segmented.png')
                if self.class_names:
                    predmap_path = os.path.join(self.save_dir, 'predmap')
                    vispath = os.path.join(self.save_dir, 'visual')
                    rgbpath = os.path.join(self.save_dir, 'rgb')
                    save_path = os.path.join(self.save_dir, 'output')

                    vis_save = os.path.join(vispath, save_name)
                    rgb_save = os.path.join(rgbpath, save_name)
                    pred_save = os.path.join(predmap_path, save_name)

                    os.makedirs(save_path, exist_ok=True)
                    os.makedirs(vispath, exist_ok=True)
                    os.makedirs(rgbpath, exist_ok=True)
                    os.makedirs(predmap_path, exist_ok=True)

                    visualize_multiclass(predicted.squeeze(0), self.class_names, savepath = pred_save)
                    visualize_multiclass(final_mask.squeeze(0), self.class_names, savepath = vis_save)
                    vis_seg_rgb(final_mask.squeeze(0), rgb_save)
                else:
                    save_path = os.path.join(self.save_dir)
                    os.makedirs(save_path, exist_ok=True)

                # Save image with PIL
                final_mask_PIL = Image.fromarray(final_mask.squeeze(0))
                final_mask_PIL.save(os.path.join(save_path, save_name))
                if self.progress and self.task_id is not None:
                    self.progress.advance(self.task_id)

                # Free up memory
                del self.canvases[rel_path]
                if self.segmentations:
                    del self.segmentations[rel_path]
                    del self.count_canvases[rel_path]
                gc.collect()
            
            self.queue.task_done()
        self.queue.task_done() # signal finished task if item = None is taken up from queue


def run_segmentation(
        data_path: str, 
        model_path: str, 
        save_path: str,
        segmentation_path: str=None, 
        sizes: Tuple[int, int, int]=(572, 388, 388),
        class_selection: str='roots',
        model_name: str = 'unet',
        batch_size: int=4, 
        num_workers: int=2, 
        filter_components: bool=False,
    ) -> None:
    """
    Main execution logic for the inference pipeline.

    Args:
        data_path (str): path where the images to be segmented are stored
        model_path (str): path to the segmentation model (.pth file)
        save_path (str): path where segmented images will be stored
        segmentation_path (str): path of binary segmented images (rootmaps required for multiclass segmentation)
        sizes (Tuple[int, int, int]): (tilesize, outputsize, stepsize)
            - tilesize: tilesize of the input to the model
            - outputsize: outsize of the output of the model
            - stepsize: strides to be used for the tiling in the large images
        class_selection (str): mode selection of segmentation - options: 'roots', 'carex', 'multispecies', 'classes'
        model_name (str): modeltype for segmentation - options: 'unet', 'swin_t', 'swin_b'
        batch_size (int): batchsize for the neural network processing
        num_workers (int): Number of worker processes spawned for CPU processing
        filter_components (bool): If True, use connected components to filter very small clusters of a specific 
            classification group (noise). Very slow for large images!

    """
    
    # SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Image.MAX_IMAGE_PIXELS = 200000000 

    # Select output
    if class_selection == 'roots':
        class_names = None
        output_size = 1
    elif class_selection == 'carex':
        class_names = ['Carex', 'Non-Carex']
        output_size = 2
    elif class_selection == 'multispecies':
        class_names = ['Anthox', 'Geum', 'Carex', 'Leontodon', 'Potentilla', 'Helictotrichon']
        output_size = 6
    elif class_selection == 'classes':
        class_names = ['Carex', 'Graminoids', 'Herbs']
        output_size = 3

    # Load model
    if model_name.lower() == 'unet':
        model = UNet(64, output_size, 4) 
    elif model_name.lower() == 'swin_t':
        model = SwinT_UNet(output_size, pretrained=False)
    elif model_name.lower() == 'swin_b':
        model = SwinB_UNet(output_size, pretrained=False)
    try:
        model.load_state_dict(torch.load(model_path))
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load model weights — ensure the given model was trained on selected category {args.class_selection}: {e}")
    model.to(device)
    model.eval()

    # Get list of images to process
    image_paths = get_image_paths(data_path)
    if segmentation_path is not None:
        segmentation_paths = get_image_paths(segmentation_path)
    else:
        segmentation_paths = None
    if not image_paths:
        return 

    # Instantiate dataset and dataloader
    dataset = TiledImageDataset(
        image_paths, 
        segmentation_paths, 
        save_path, 
        tilesize=sizes[0], 
        pad=(sizes[0]-sizes[1])//2, 
        stepsize=sizes[2]
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True, 
    )

    # Pipeline execution - initialise progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        
        # Get count of already processed images for progress bar
        processed_count = sum(
            1 for path_info in image_paths
            if os.path.exists(os.path.join(save_path, path_info['relative'].replace('.tiff', '_segmented.png')))
        )
        task_id = progress.add_task(
            "Segmenting images...", 
            total=len(image_paths), 
            completed=processed_count
        ) 

        result_queue = queue.Queue()
        reconstructor = Reconstructor(
            result_queue, 
            save_path, 
            progress=progress, 
            task_id=task_id, 
            class_names=class_names, 
            pad=(sizes[0]-sizes[1])//2, 
            stepsize=sizes[2], 
            imgsize=sizes[0],
            filter_components=False
        )
        reconstructor.start()

        with torch.no_grad():
            for batch in data_loader:
                tiles = batch['tile'].to(device, non_blocking=True) 
                logits = model(tiles)

                if logits.shape[1] > 1: # multiclass
                    predicted_tiles = torch.softmax(logits, dim=1)
                else: # binary
                    predicted_tiles = torch.sigmoid(logits)
                for i in range(tiles.shape[0]):
                    info = {k: v[i] for k, v in batch['image_info'].items()}
                    result_item = (predicted_tiles[i], batch['coords'][i], info)
                    result_queue.put(result_item)
                    
        result_queue.put(None) 
        reconstructor.join()


def filter_small_components(binary_mask: np.ndarray|torch.Tensor, min_size: int=250):
    """
    Removes small connected components from a binary mask.

    Args:
        binary_mask (np.ndarray|torch.Tensor): 2D array with 0/1 or 0/255 values.
        min_size (int): Minimum connected component size to keep (in pixels).

    Returns:
        filtered_mask (np.ndarray): 2D binary mask with small components removed.
    """
    # Ensure binary
    if isinstance(binary_mask, torch.Tensor):
        binary_mask = binary_mask.cpu().numpy()
    mask = (binary_mask > 0).astype(np.uint8)

    # Label connected components
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)

    # Initialize filtered mask
    filtered_mask = np.zeros_like(mask)
    for region in regions:
        if region.area >= min_size:
            filtered_mask[labeled == region.label] = 1
    return filtered_mask



if __name__ == "__main__":

    # Parse input
    parser = argparse.ArgumentParser(description="PyTorch Tiled Inference Pipeline")
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to the folder with large images."
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the trained PyTorch model (.pth file)."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        help='model type'
    )
    parser.add_argument(
        "--segmenter_model", 
        type=str, 
        default=None, 
        help="if provided, multiclass segmentation is assumed and a segmenter" \
             "first detects roots for the masking"
    )
    parser.add_argument(
        "--class_selection", 
        type=str, 
        choices=['roots', 'multispecies', 'classes'], 
        required=True, 
        default='roots', 
        help="Segmentation type, depending on how many classes are predicted - limited to binary ('roots'), " \
             "species-specific ('multispecies') and functional groups ('classes')"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True, 
        help="Directory to save segmented masks."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4, 
        help="Batch size for GPU inference."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=2, 
        help="Number of CPU workers for data loading."
    )
    args = parser.parse_args()
    
    # binary segmentation
    if args.class_selection == 'roots':
        run_segmentation(
            data_path=args.data_path, 
            model_path=args.model_path, 
            save_path=args.save_path, 
            sizes=(572, 388, 388),
            model_name=args.model,
            class_selection=args.class_selection,
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
    # Multiclass segmentation 
    else:
        # Create binary rootmaps
        segpath = os.path.join(args.save_path, 'segmentation')
        run_segmentation(
            data_path=args.data_path, 
            model_path=args.segmenter_model, 
            save_path=segpath, 
            sizes=(572, 388, 388),
            model_name='unet',
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            class_selection='roots'
        )

        # Ensure memory is cleaned for the second run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect() 
        
        # multiclass segmentation with rootmap
        run_segmentation(
            data_path=args.data_path, 
            model_path=args.model_path, 
            save_path=args.save_path,
            sizes=(768, 644, 500),
            model_name=args.model,
            class_selection=args.class_selection,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            segmentation_path=segpath,
        )

    