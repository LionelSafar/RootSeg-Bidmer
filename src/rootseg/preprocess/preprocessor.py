import numpy as np
import argparse
import os
import re
import cv2

from collections import defaultdict
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TaskID,
)
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align

import multiprocessing as mp
import threading
import traceback
import logging
from .utils import *
from functools import partial
from torch.utils.data import Dataset
from itertools import groupby


class Process_Tube:
    """
    Tube processor class
    
    Takes a tube and processes images chronologically and saves them in a new 'processed' folder:
    The following steps are done to each image in the timeseries:
        1) If mask-folder is given, removes the tape on the topmost image (if multiple levels)
        2) fix image size of each image to (8784, 10200) (width will be higher if stitching)
        3) If 'args.stitch_data' -- stitch all images in order to their level L{i} -> L{i+1} -> ..
            based on Phase Correlation.
        4) If 'args.reference_path' will align the image to the last-dated image in the reference path
            else will skip and the consecutive images will be aligned to the first image in the series
            -- image alignment is based on Phase Correlation
        5) Enhance image
            a) Remove scanning artefacts by performing row-wise and column-wise mean-centering
            b) Performs Histogram equalization
            c) Normalizes contrast and brightness to manually set reference values
        -- NOTE: masked regions are saved temporarily as blue pixels and will be marked black before saving

    """
    def __init__(self, tube, args, progress_queue):
        self.progress_queue = progress_queue
        self.tube = tube
        self.args = args

        # initialise to gather alignment summaries
        self.stitch_summary = []
        self.align_summary = []

        # Initialise to save previous image size for stitching later in case of missing images as fallback size
        self.prev_size = None

        # Get Reference image to align the first image later - if provided
        if args.reference_path:
            reference_path = os.path.abspath(args.reference_path)
            ref_tube_path = os.path.join(reference_path, self.tube)
            try:
                ref_files = [f for f in os.listdir(ref_tube_path) if f.lower().endswith(".tiff")]
                ref_files.sort(key=lambda x: get_date(x))
                self.ref_path = os.path.join(ref_tube_path, ref_files[-1])
                self.ref_img_gray = np.float32(cv2.imread(self.ref_path, cv2.IMREAD_GRAYSCALE))
            except Exception as e:
                logging.error(f"{self.tube}: Failed to load reference image from '{ref_tube_path}'. Error: {e}", exc_info=True)
        else:
            self.ref_img_gray = None
        
        # Get tape masks of the tube if they exist
        mask_path = os.path.abspath(os.path.join(args.raw_folder, '..', 'masks'))
        if os.path.exists(mask_path):
            masks = os.listdir(mask_path)
            mask_list = [os.path.join(mask_path, mask) for mask in masks]
            self.tube_masks = [mask for mask in mask_list if re.search(rf"{re.escape(tube)}(?=[._])", mask)]
        else:
            logging.warning(f"No Mask folder detected: {mask_path} - Assuming images without mask!")
            self.tube_masks = None

        # Initialise CLAHE and image information
        self.clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(30, 30))
        self.tube_folder = os.path.join(args.raw_folder, tube)
        self.images = [
            {"filename": fname, "date": get_date(fname), "level": get_level(fname)}
            for fname in os.listdir(self.tube_folder)
        ]
        self.dates = sorted({img["date"] for img in self.images}) # create set of dates
        self.max_level = max(img["level"] for img in self.images) # max level for stitching
        self.images.sort(key=lambda x: x["date"]) # sort by date

        # Output paths
        self.outpath = os.path.abspath(os.path.join(args.raw_folder, '..', 'preprocessed', tube))
        os.makedirs(self.outpath, exist_ok=True)
        self.preview_path = os.path.abspath(os.path.join(args.raw_folder, '..', 'preview', tube))
        os.makedirs(self.preview_path, exist_ok=True)

        self.progress_queue.put((tube, "total images", len(self.dates)))

    def process(self):
        """
        main function of the class that preprocesses a whole tube
        """
        # Chronologically process images
        for date in self.dates:
            if self.args.stitch_data:
                shifts = [] # save all shifts for each level to stitch later on
                imgs = []

                # Iterate through all occuring levels and stitch
                for i in range(1, self.max_level):
                    Ltop = [img for img in self.images if img["date"] == date and img["level"] == i]
                    Lbot = [img for img in self.images if img["date"] == date and img["level"] == i+1]
                    if i == 1 and len(Ltop) != 1:
                        raise ValueError(f"{self.tube}: Multiple or No L1 image registered for {date} - Please manually check!")
                    # In case of top image, remove tape if mask is present
                    else: 
                        path = os.path.join(self.tube_folder, Ltop[0]["filename"])
                        L_top = cv2.imread(path, cv2.IMREAD_COLOR)
                        L_top = rem_tape(L_top, self.tube_masks, Ltop[0]["filename"], self.tube)
                        imgs.append(L_top)
                    if len(Lbot) != 1:
                        logging.error(f"{self.tube}: Multiple or no L{i+1} images found at {date}:\n" +
                            "\n".join(img["filename"] for img in Ltop)
                        )
                    try:
                        L_bot_path = os.path.join(self.tube_folder, Lbot[0]["filename"])
                        L_bot = cv2.imread(L_bot_path, cv2.IMREAD_COLOR)
                        L_bot = fix_image_size(
                            L_bot, 
                            size=(self.args.img_height, 10200),
                            name=os.path.basename(L_bot_path))
                    except Exception as e:
                        logging.warning(f"{self.tube}: Issue reading bottom image: {e}")
                        L_bot = None
                    
                    # Get shift from CV per image level
                    shift = get_stitch_coords(L_top, L_bot, self.tube, self.clahe)
                    new_name = re.sub(r"L\d+", "L0", Ltop[0]["filename"])
                    outpath = os.path.join(self.outpath, new_name)
                    if type(shift[0]) == str:
                        line = f"{Ltop[0]['filename']}\t{shift[0]}\t{shift[1]}\t{shift[2]:.2f}\n"
                    else:
                        line = f"{Ltop[0]['filename']}\t{shift[0]:+d}\t{shift[1]:+d}\t{shift[2]:.2f}\n"
                    self.stitch_summary.append(line)
                    if shift[2] < 0.1:
                        logging.warning(f"{self.tube}: Low correlation ({shift[2]}) while stitching {new_name}! -- See stitching log for details")
                    elif shift[1] == (0, 0):
                        logging.warning(f"{self.tube}: Highest correlation is at shift (0, 0) while stitching {new_name}! -- Chance of identical images!")
                    shifts.append(shift)
                    imgs.append(L_bot)

                # Stitch all levels together
                img, self.prev_size = iterative_stitching(imgs, shifts, self.prev_size, self.tube)

            else: # NON-stitching case -- Tape removal and fixing image size
                imgs = [img for img in self.images if img["date"] == date]
                if len(imgs) != 1:
                    logging.error(f"{self.tube}: No or multiple images found at {date}:\n" +
                        "\n".join(img['filename'] for img in imgs) +
                        "-- image is being skipped withour raising ERROR!")
                    continue
                path = os.path.join(self.tube_folder, imgs[0]["filename"])
                new_name = re.sub(r"L\d+", "L0", imgs[0]["filename"])
                outpath = os.path.join(self.outpath, new_name)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = fix_image_size(img, size=(self.args.img_height, 10200), name=os.path.basename(path))
                # Put black spots in blue
                black_mask = np.all(img == 0, axis=-1)
                blue_col = np.array([255, 0, 0], dtype=img.dtype)
                img = np.where(black_mask[..., None], blue_col, img)

                new_name = re.sub(r"L\d+", "L0", imgs[0]["filename"])
                if self.tube_masks:
                    img = rem_tape(img, self.tube_masks, new_name, self.tube)

            # Align image if reference image is given
            if self.ref_img_gray is not None:
                y_ref, x_ref = self.ref_img_gray.shape
                y, x = img.shape[:2]
                if x != x_ref or y != y_ref: # PC requires same image size!
                    img = fix_image_size(img, size=(y_ref, x_ref), crop_left=True)
                img_gray = np.float32(self.clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

                # Remove black parts from the right for correlation and ensure same width of both images:
                img_gray, r_indx = crop_rightmost_nonblack(img_gray)
                ref_img_gray, ref_r_indx = crop_rightmost_nonblack(self.ref_img_gray)   

                # ensure the cropping occurs in the same region of the image -> in case of large differences in rightmost
                diff_indx = r_indx - ref_r_indx
                if diff_indx > 0: # case ref_img has longer black patch
                    img_gray = img_gray[:, :-diff_indx]
                elif diff_indx < 0:
                    ref_img_gray = ref_img_gray[:, :diff_indx]

                # fix image shape for phase correlation
                min_width = min(x_ref, x, 7000)
                ref_img_gray = ref_img_gray[:, -min_width:]
                img_gray = img_gray[:, -min_width:]
                shift = cv2.phaseCorrelate(ref_img_gray[::5, ::5], img_gray[::5, ::5])
                if shift[1] < 0.1: # In case of low correlation - enhance images first and check for improvement
                    img_gray_normed = norm_grayscale(img_gray)
                    ref_img_gray_normed = norm_grayscale(ref_img_gray)
                    h = img_gray_normed.shape[0]
                    ref_img_gray_normed = np.vstack([ref_img_gray_normed[-h//2:], ref_img_gray_normed, ref_img_gray_normed[:h//2]])
                    img_gray_normed = np.vstack([img_gray_normed[-h//2:], img_gray_normed, img_gray_normed[:h//2]])

                    shift_new = cv2.phaseCorrelate(ref_img_gray_normed[::5, ::5], img_gray_normed[::5, ::5])
                    t1 = os.path.basename(self.ref_path)
                    t2 = os.path.basename(outpath)
                    if shift_new[1] > shift[1]:
                        logging.info(f"{self.tube}: Low initial correlation in time series alignment ({shift[1]:.2f}) between {t1} and {t2} -- enhancement procedure leads to higher corr. : {shift_new[1]:.2f}")
                        shift = shift_new
                        if shift_new[1] < 0.1:
                            logging.warning(f"{self.tube}: Correlation of time series alignment is low ({shift[1]})")
                            
                dx, dy = int(round(shift[0][0])), int(round(shift[0][1])) # shift in pixels
                dx *= 5
                dy *= 5
            
                # apply shift to the image and blacken out the overlapped parts in x-direction, y is 360 scan
                img = np.roll(img, (-dy, -dx), axis=(0, 1))
                blue = np.array([255, 0, 0], dtype=img.dtype)
                if dx < 0:
                    img[:, :-dx, :] = blue  # -> shifted to the left -> zero out the right
                elif dx > 0:
                    img[:, -dx:, :] = blue  # -> shifted to the right -> zero out the left
                line = f"{new_name}\t{-dx:+d}\t{-dy:+d}\t{shift[1]:.2f}\n"
                self.align_summary.append(line)

            # Overwrite reference image for next iteration
            self.ref_img_gray = np.float32(self.clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
            self.ref_path = outpath

            # Enhance image
            img = remove_scannoise(img)

            # save blue mask including bright regions + tape
            blue_mask = (
                (img[..., 0] == 255) &      
                (img[..., 1] == 0) &      
                (img[..., 2] == 0)    
            )

            # Enhance images and save
            img = applyCLAHE(img, self.clahe)
            img = contrast_brightness_normalization(img)
            img[blue_mask] = 0
            img = add_reference_pixels(img)
            if self.preview_path:
                img_prev = img[::10, ::10]
                filename_prev = new_name.replace('.tiff', '_preview.png')
                cv2.imwrite(os.path.join(self.preview_path, filename_prev), img_prev)
            cv2.imwrite(outpath, img)
            self.progress_queue.put((self.tube, "images processed", 1))
        logging.info(
            f"\n\n---- Summary of stitching for {self.tube} ----\n" +
            "\n".join(self.stitch_summary) +
            "-"*50
        )
        logging.info(
            f"\n\n---- Summary of alignment for {self.tube} ----\n" +
            "\n".join(self.align_summary) +
            "-"*50
        )
        self.progress_queue.put((self.tube, "done", "done"))


def process_tube(tube, args, progress_queue):
    """Initialise Tube class and process"""
    pipeline = Process_Tube(tube, args, progress_queue)
    pipeline.process()


def manage_progress(progress_queue, progress_ui, all_done_event):
    """
    Progress Manager, runs as separate Thread to communicate with worker tube
    """
    active_tasks: dict[str, TaskID] = {}

    while not all_done_event.is_set():
        try:
            # Wait for a message from a worker
            message = progress_queue.get(timeout=0.1)
            tube, description, value = message

            if tube not in active_tasks:
                total = value if description == "total images" else 1.0
                task_id = progress_ui.add_task(f"{tube}", tube=tube, total=total)
                active_tasks[tube] = task_id
            else:
                task_id = active_tasks[tube]

            # Process the message based on the 'value'
            if description == "total images":
                # Start of a new stage. Reset the bar to 0% and update the description.
                progress_ui.reset(task_id, description=description, total=value, completed=0)
            elif value == "done":
                # End of stage
                progress_ui.update(task_id, completed=progress_ui.tasks[task_id].total)
            else:
                # Advance the bar by the given step.
                progress_ui.update(task_id, advance=value)

        except (mp.queues.Empty, ValueError):
            # Queue is empty or message is not a 3-part tuple - ignore and continue
            continue

def run_job_wrapper(tube, args, progress_queue):
    """Wrapper around tube processing with error callback"""
    try:
        process_tube(tube, args, progress_queue)
        return tube, "Success"
    except Exception as e:
        traceback.print_exc() 
        return tube, f"Error: {e}"

def error_callback(error, tube_name: str = None):
    #print(f"\n[ERROR] An exception occurred in a worker process:")
    #traceback.print_exception(type(error), error, error.__traceback__, file=sys.stderr)
    prefix = f"{tube_name}: " if tube_name else ""
    separator = "\n" + "="*80 + "\n"
    logging.error("%s%sEXCEPTION IN WORKER%s\n%s%s",
                  separator,
                  prefix,
                  separator,
                  "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                  separator)

def parse_tube_list(tube_args, available_tubes):
    """Get all selected tubes from input"""
    tubes = []

    for entry in tube_args:
        # In case of tube range "a-b"
        range_match = re.match(r"^(\d+)-(\d+)$", entry)
        if range_match:
            start, end = map(int, range_match.groups())
            for i in range(start, end + 1):
                # Find all matches like T1, T1N, T1S in available_tubes
                matches = [tube for tube in available_tubes if re.match(rf"T{i}($|N$|S$)", tube)]
                tubes.extend(matches)
        else:
            # Direct name (e.g., T3N or T5)
            if entry in available_tubes:
                tubes.append(entry)
            else:
                # Try matching with suffixes
                matches = [tube for tube in available_tubes if tube.startswith(entry)]
                tubes.extend(matches)

    return sorted(set(tubes))


def processor(args):
    """ main function to run the pipeline"""

    # Configure logger
    log_path = os.path.join(args.raw_folder, '..')
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_path}/pipeline.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print(75*"-")
    print(20*" ", "Starting preprocessing pipeline")
    print(75*"-")
    
    # Get tubes
    tubes = sorted(os.listdir(args.raw_folder))

    if args.tubes_list:
        tubes = parse_tube_list(args.tubes_list, tubes)
    num_tubes = len(tubes)

    #cpu_count = mp.cpu_count()
    #num_workers = min(num_tubes, cpu_count - 1 if cpu_count > 1 else 1)
    num_workers = 5 # Limit to 5 workers - threshold for us is disk reading, increasing mp does not accelerate the process

    # Initialise Multiprocessing and worker UI
    manager = mp.Manager()
    progress_queue = manager.Queue()
    worker_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.fields[tube]}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    
    # Define overall progress UI and Layout
    overall_progress = Progress(
        TextColumn("Overall Tubes processed:"), 
        BarColumn(), 
        MofNCompleteColumn(), 
        TimeElapsedColumn()
    )
    overall_task = overall_progress.add_task("Processing...", total=num_tubes)
    layout = Layout()
    layout.split(
        Layout(Panel(Align.center(overall_progress), title="Total"), name="main", size=3),
        Layout(Panel(Align.center(worker_progress), title=f"Active Workers ({num_workers})"), name="workers")
    )


    with Live(layout, refresh_per_second=5):
        # Create a seperate Thread for progress display
        all_done_event = threading.Event()
        progress_thread = threading.Thread(
            target=manage_progress,
            args=(progress_queue, worker_progress, all_done_event),
        )
        progress_thread.start()

        # Progress tubes asynchronously with MP
        with mp.Pool(processes=num_workers) as pool:
            worker_func = partial(run_job_wrapper, args=args, progress_queue=progress_queue)

            for finished_tube, status in pool.imap_unordered(worker_func, tubes):
                if "Error" in status:
                    logging.error(f"Failed to process {finished_tube}: {status}")
                overall_progress.update(overall_task, advance=1)

        all_done_event.set()
        progress_thread.join()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Preprocessing pipeline for root segmentation')
    argparser.add_argument(
        '--raw_folder', 
        type=str,
        help='The folder containing the files to process, sorted to their tubes.'
    )
    argparser.add_argument(
        '--stitch_data', 
        action='store_true', 
        default=False,
        help='Whether to stitch L1 and L2 images together.' \
             'In case of "Einzelarten" data, the images do not need stitching'
    )
    argparser.add_argument(
        '--reference_path', 
        default=None,
        help='Reference base path containing all tubes of e.g. the previous year for timeseries alignment, ' \
             'if not specified, will use the first image of each tube as reference by default'
    )
    argparser.add_argument(
        '--tubes_list', 
        type=str, 
        nargs='+', 
        default=None,
        help='List of tubes to process. Input can be a list of tubes, e.g. "T1" "T10" or a range' \
             'of tubes "1-10" - will automatically check for N/S orientation as well if range is given.' \
             'If None, all tubes in the folder are processed.'
    )
    argparser.add_argument(
        '--img_height', 
        type=int, 
        default = 8784,
        help='Height of the images -- will be fixed for all to this value'
    )
    args = argparser.parse_args()
    processor(args)