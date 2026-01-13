from rootseg.preprocess.rename_and_order import rename_and_order
import argparse

def main():
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

if __name__ == "__main__":
    main()