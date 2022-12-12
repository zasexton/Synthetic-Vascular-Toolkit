#!/usr/bin/env python

try:
    import sv
except ImportError:
    raise ImportError('Run with sv --python -- this_script.py')

import argparse
import glob
import os


def main(p):
    sv.PathLegacyIO.Convert(file_name=p.path_in, output_dir=p.path_out)
    for seg in glob.glob(os.path.join(p.seg_in, '*')):
        if '.tcl' not in seg:
            sv.SegmentationLegacyIO.Convert(file_name=seg, path_file=p.path_in, output_dir=p.seg_out)


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser(description='convert legacy path in SimVascular')
    parser.add_argument('path_in', help='input path folder')
    parser.add_argument('path_out', help='output path folder')
    parser.add_argument('seg_in', help='input segmentation folder')
    parser.add_argument('seg_out', help='output segmentation folder')

    # run script
    main(parser.parse_args())
