# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:01:42 2021

@author: minoc

# =============================================================================
# Create mask of the data using the T1 files and considering as background 
# the voxel with same value of the median of T1 data
# =============================================================================

"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import (
    create_mask,
    create_path_from_filename,
    get_nifty_list, 
    load_niftyimg_from_file, 
    save_nifty_file,
    )
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        help='Directory of data.',
                        default="D:/Datasets/valdo_norm/Task3")
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
        
    file_list = get_nifty_list(args.data_dir, "_T1")
    for i, file in enumerate(file_list):
        break
        print(f"{i+1} / {len(file_list)}", end="\r\n")
        mask_path = create_path_from_filename(file, "mask.nii.gz")
        data = load_niftyimg_from_file(file)
        mask = create_mask(data.get_fdata())
        save_nifty_file(mask, mask_path, affine=data.affine)