from rootseg.preprocess.preprocessor import preprocess
import argparse

def main():
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
        help='Whether to stitch L1 and L2 images together. ' \
             'In case of "Einzelarten" data, the images do not need stitching'
                        )
    argparser.add_argument(
        '--reference_path', 
        default=None,
        help='Reference base path containing all tubes of e.g. the previous year for timeseries alignment, '
             'if not specified, will use the first image of each tube as reference by default')
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


if __name__ == '__main__':
    main()