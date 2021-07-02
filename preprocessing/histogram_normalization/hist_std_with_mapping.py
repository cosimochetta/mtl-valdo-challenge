# -*- coding: utf-8 -*-

"""

@original author: pmeht
@updates: minoc

# =============================================================================
# This function applies histogram standardisation to images 
# in a folder using a landmark mapping file input
# =============================================================================
"""


import niftynet.utilities.histogram_standardisation as hs
from niftynet.layer.binary_masking import BinaryMaskingLayer
import os
import nibabel as nib
import numpy as np
import numpy.ma as ma
import argparse
import glob
import sys

def standardisation(image_folder, output_folder, mapping_file, norm_type, img_type="_T1", masking_function=None, post_whitening=False):
    image_ext = '.nii.gz'
    #histogram standardisation setting
    cutoff = [0.01, 0.99]
    #import histogram landmarks
    mapping_file = os.path.abspath(mapping_file)
    mapping = hs.read_mapping_file(mapping_file)
    mapping = mapping['Data']
    
    #standardise images in img_folder
    file_list = glob.glob(image_folder + "/**/*", recursive=True)
    file_list = [file for file in file_list if image_ext in file and img_type in file]
    num_file = len(file_list)
    
    for idx in range(num_file):
        image = nib.load(os.path.join(image_folder,file_list[idx]))
        data = np.array(image.get_fdata(), dtype=float)
        print(image.header)
        
        if masking_function is not None:
            mask = masking_function(data)
        else:
            mask = np.ones_like(data, dtype=bool)
            
        #perform standardisation
        std_img = hs.transform_by_mapping(data, mask, mapping, cutoff, type_hist=norm_type)
        #perform whitening
        if post_whitening == True:
            masked_img = ma.masked_array(std_img, np.logical_not(mask))
            std_img = (std_img - masked_img.mean()) / max(masked_img.std(), 1e-5)
        # image.header[datatype] = FLOAT32
        # image.header['scl_inter'] = 0
        # print(image.header)
        std_img[std_img<0]=0
        write_image = nib.Nifti1Image(std_img, image.affine, header=image.header)        
        basename = os.path.basename(file_list[idx])
        os.makedirs(os.path.join(output_folder, basename[0:7]), exist_ok=True)

        print(f"saving file: {os.path.join(output_folder, basename[0:7], basename)}")
        nib.save(write_image, os.path.join(output_folder, basename[0:7], basename))        
    return None


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, help='Input directory',
                        default="D:/Datasets/valdo_registration/Task3")
    parser.add_argument("--output_folder", type=str, help='Output directory for histnorm data',
                        default="D:/Datasets/valdo_norm/Task3")
    parser.add_argument('--norm_type', type=str, 
                        default="percentile", choices=['percentile'])
    return parser.parse_args(argv)


if __name__=="__main__":
    args = parse_arguments(sys.argv[1:])
    post_whitening=False
    
    img_types = ["_T1", "_T2", "_FLAIR"]
    for img_type in img_types:
        mapping_file=args.input_folder+"/hist_ref" + img_type + ".txt"
        masking_function=BinaryMaskingLayer(type_str='otsu_plus',multimod_fusion='and') 
        post_whitening=False 
        os.makedirs(args.output_folder, exist_ok=True)  
        standardisation(args.input_folder, args.output_folder, mapping_file, args.norm_type, img_type=img_type, masking_function=masking_function, post_whitening=post_whitening)