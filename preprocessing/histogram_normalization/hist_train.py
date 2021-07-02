# -*- coding: utf-8 -*-

"""

@original uthor: pmeht
@changes: minoc

# =============================================================================
# This function extracts histogram landmarks from a dataset
# Requires NiftyReg installed
# =============================================================================

"""

import os
import nibabel as nib
import numpy as np
import niftynet.utilities.histogram_standardisation as hs
from niftynet.layer.binary_masking import BinaryMaskingLayer
import argparse
from .hist_std_with_mapping import standardisation
import glob
import sys

def train(image_folder, mapping_file, img_type="_T1", masking_function=None):
    image_ext = ".nii.gz"
    cutoff=[0.01,0.99]
    percentiles_database = []
    mapping_dict = {}
    file_list = glob.glob(image_folder + "/**/*", recursive=True)
    file_list = [file for file in file_list if image_ext in file and img_type in file]
    num_data = len(file_list)

    for idx in range(num_data):
        image_path = os.path.join(image_folder,file_list[idx])
        print(f"{idx} / {num_data} : {image_path}\r")
        image_load = nib.load(image_path)
        data = np.array(image_load.get_fdata(),dtype=float)

        if masking_function is not None:
            mask = masking_function(data)
        else:
            mask = np.ones_like(data, dtype=np.bool)

        percentiles = hs.__compute_percentiles(data, mask, cutoff)
        percentiles_database.append(percentiles)

    percentiles_database = np.vstack(percentiles_database)
    s1, s2 = hs.create_standard_range()
    mapping_dict['Data'] = tuple(hs.__averaged_mapping(percentiles_database, s1, s2))
    hs.write_all_mod_mapping(mapping_file, mapping_dict)

    return percentiles_database

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, help='Input directory',
                        default="D:/Datasets/valdo_registration/Task3")
    parser.add_argument("--output_folder", type=str, help='Output directory for histnorm data',
                        default="D:/Datasets/valdo_norm/Task3")
    parser.add_argument('--norm_type', type=str, 
                        default="percentile", choices=['percentile'])
    parser.add_argument('--normalize', type=bool, default=False)
    return parser.parse_args(argv)


if __name__ == '__main__':
    
    args = parse_arguments(sys.argv[1:])
    post_whitening=False
    norm_type="percentile"
    
    img_types = ["_T1", '_T2,', '_FLAIR']
    
    for img_type in img_types:
        mapping_file=args.input_folder+"/hist_ref" + img_type + ".txt" 
        os.makedirs(args.output_folder, exist_ok=True)
        masking_function=BinaryMaskingLayer(type_str='otsu_plus',multimod_fusion='and')
        train(args.input_folder, mapping_file, img_type=img_type, masking_function=masking_function)
        if args.normalize:
            standardisation(args.image_folder, args.output_folder, mapping_file, args.norm_type, masking_function=masking_function, post_whitening=post_whitening)

