"""
This script is context specific for the Einzelarten and Bidmer files.
It is used to rename all files from a folder to a specific format.
The saving format is: "Einzelarten_T{i}_L{j}_DD.MM.YY_HHMMSS_NR_ID.tiff" or "Bidmer_...T{i}{N/S}..."
The images are assorted to the subfolders indicating their tube.

The script explicitly works by using "T" as identifier for tube and "L" as identifier for scan level,
and expects L{j}_ to be followed by the structure {Date}_{Time}_{Number}_{ID}.tiff
For other usecases this has to be adjusted / a new script has to be written.

NOTE:
- L0 is changed to L1 by default for the Bidmer files, Einzelarten ignores L0 files -> no stitching is done
- Tubes for Bidmer will be names T{i}{N/S} for North and South tubes of a plot
- The script will ask for the orientation of the tube if it is not found in the filename
- The script will ask for the date if it is not found in the filename

NOTE Requirements:
- The script requires the following information on the initial filename

"""

import os
import re
import argparse
from typing import Dict
import shutil

def format_date(filename: str) -> str:
    """
    Format the date of a filename to the format DD.MM.YY

    Args:
        filename (str): filename string with the date in one of the following formats:
            - DD.MM.YYYY
            - YYYY.MM.DD
            - DD.MM.YY
    
    Returns:
        str: filename with adjusted date format to YYYY.MM.DD

    """
    # Try to match all possible allowed date formats and change to output format
    match_DDMMYYYY = re.search(r"_(\d{2})[.\-/](\d{2})[.\-/](\d{4})[_\.]", filename)
    match_YYYYMMDD = re.search(r"_(\d{4})[.\-/](\d{2})[.\-/](\d{2})[_\.]", filename)
    match_DDMMYY = re.search(r"_(\d{2})[.\-/](\d{2})[.\-/](\d{2})[_\.]", filename)
    if match_DDMMYYYY:
        day, month, year = match_DDMMYYYY.groups()
        if len(year) == 2:
            year = f'20{year}'
        new_date = f"_{year}.{month}.{day}_"
    elif match_YYYYMMDD:
        year, month, day = match_YYYYMMDD.groups()
        if len(year) == 2:
            year = f'20{year}'
        new_date = f"_{year}.{month}.{day}_"
    elif match_DDMMYY:
        day, month, year = match_DDMMYY.groups()
        year = f'20{year}'
        new_date = f"_{year}.{month}.{day}_"
    else:
        print(f"Date format in'{filename}' not recognized.")
        new_date = input("Please enter the date in the format DD.MM.YY: ")
        raise ValueError(f"Date format not recognized.")
    
    return re.sub(r"_(\d{2,4})[.\-/](\d{2})[.\-/](\d{2,4})", new_date, filename)


def retrieve_info(filename: str, change_level_name: bool=False) -> Dict[str, str]:
    """
    Retrieve image information from the filename. Require that format_date is 
    called first / Date is in the format DD.MM.YY

    Args:
        filename (str): the filename to retrieve the information from.
        change_level_name (bool): If 'True' changes L0 naming to L1 (common mistake when images were saved)

    Returns:
        dict: a dictionary containing the information of the filename.
            - T: the tube number
            - O: Orientation (N/S in case of Bidmer images)
            - L: the scan level
            - D: the date
            - R: the auxiliary information, including the time, number and ID and file extension.

    """
    date_pattern = r"(\d{4})\.(\d{2})\.(\d{2})"
    date_match = re.search(date_pattern, filename)

    if date_match:
        date = date_match.group(0)
        start_idx = date_match.start()
        aux_indx = date_match.end()
        aux = filename[aux_indx:]
    else:
        raise ValueError(f"Date not found in the filename: {filename}")
    
    # Retrieve tube and level information, only looking before the date to avoid conflicts with ID
    subname = filename[:start_idx] 

    tube = re.search(r"T(\d+)", subname).group(1)
    level = re.search(r"L(\d+)", subname).group(1)

    # In case of bidmer, get North or South tube as well
    try:
        orientation = re.search(r"(N|S)_", subname).group(1)
    except AttributeError:
        orientation = '' 
        #orientation = 'S'

    if level == '0' and change_level_name == True:
        print("Changing L0 -> L1")
        level = '1'

    if not tube or not level:
        raise ValueError("Tube or level not found in the filename.")

    return {"T": tube, "O": orientation, "L": level, "D": date, "R": aux}


def rename_file(experiment: str, file_dict: Dict[str, str]) -> str:
    """ Rename the file to the specific format. {experiment}_T{i}_L{j}_DD.MM.YY_{AUX}.tiff"""
    return f"{experiment}{file_dict['O']}_T{file_dict['T']}_L{file_dict['L']}_{file_dict['D']}{file_dict['R']}"


def rename_and_order(args):
    """
    Rename all files in the base folder to the specific format and move them to the saving path.
    """
    if args.dry_run:
        print("Running in dry run mode. Files will not be moved. but printed.")

    files = []
    for root, _, filenames in os.walk(args.folder):
        for name in filenames:
            if name.endswith(".tiff"):
                files.append(os.path.join(root, name))

    # Extract year:
    data = retrieve_info(format_date(files[0]))
    year = data['D'][2:4]

    # Get saving path and create it if it does not exist
    if args.savepath:
        current_dir = args.savepath
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    saving_path = os.path.join(current_dir, 'Data', args.experiment+year, 'Raw_images')
    os.makedirs(saving_path, exist_ok=True)

    # Rename all files and move them to the saving path - Consider subfolders as well
    for file in files:
        if not file.endswith(".tiff"):
            print(f"Skipping {file}, not a tiff file.")
            continue

        # Get source
        source_file = os.path.join(args.folder, file)
        
        # Rename file
        formatted_file = format_date(file)
        data = retrieve_info(formatted_file, change_level_name=args.rename_L0)
        T = data["T"]
        O = data["O"]
        if data["L"] == "0":
            print('Skipping L0 file')
            continue
        new_name = rename_file(args.experiment, data)

        # Get target and move file
        target_path = os.path.join(saving_path, f"T{T}{O}")
        os.makedirs(target_path, exist_ok=True)

        # Move file
        target_file = os.path.join(target_path, new_name)
        if not args.dry_run:
            if args.copy:
                shutil.copy2(source_file, target_file)
            else:
                shutil.move(source_file, target_file)
        else:
            print(50*"-")
            print(f'moved {source_file} to {target_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename files in a folder to a specific format.')
    parser.add_argument('--folder', type=str,
                        help='The folder containing the files to rename.'
                        )
    parser.add_argument('--savepath', type=str, default=None,
                        help='If given, will save at specified location')
    parser.add_argument('--experiment', type=str,
                        help='The format to rename the files to. - Einzelarten or Bidmer'
                        )
    parser.add_argument('--orientation', action='store_true', default=False,
                        help='Whether to include N/S orientation to the tube'
                        )
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Whether to run the script without actually moving the files.'
                        )
    parser.add_argument('--copy', action='store_true', default=False,
                        help='Whether to make a copy in the target folder instead of moving'
                        )
    parser.add_argument('--rename_L0', action='store_true', default=False,
                        help='Wrongly saved L0 images are renamed to L1, if not, L0 labelled images will be ignored')
                        
    args = parser.parse_args()
    rename_and_order(args)

