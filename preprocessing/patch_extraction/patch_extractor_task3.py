"""
File name: seg_patch_extraction.py
Author: ngocviendang
Date created: July 13, 2020
This file extracts the patches of brain volumes.
Reference https://github.com/prediction2020/unet-vessel-segmentation

# =============================================================================
# Extract all patches centered on voxel with Lacunes with horizontal flip for
# augmentation.
# Then extracts random patches in the same number.
# =============================================================================
"""

import os
import numpy as np
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import (
    adjust_images_task3,
    create_mask,
    get_file_list_task3, 
    load_nifty_from_file, 
    merge_task3_label,
    )

def main(args):
    patch_sizes = [64]    # different quadratic patch sizes n x n
    
    input_dir = os.path.expanduser(args.input_dir)
    patch_extraction_dir = os.path.expanduser(args.patch_extraction_dir)
    if not os.path.exists(patch_extraction_dir):
        os.makedirs(patch_extraction_dir)
        
    for size in patch_sizes:
        if not os.path.exists(os.path.join(patch_extraction_dir, str(size))):
            os.makedirs(os.path.join(patch_extraction_dir, str(size)))
            
    # List filenames of data after the skull stripping process
    t1_list, t2_list, flair_list, r1_list, r2_list = get_file_list_task3(input_dir)
    
    ##############################
    # EXTRACT ALL POSITIVE PATCHES
    ##############################
    tot_lacunes_patches = 0
    for idx in range(len(t1_list)):
        # load image and label stacks as matrices
        print('> Loading images...')
        t1 = load_nifty_from_file(t1_list[idx])
        t2 = load_nifty_from_file(t2_list[idx])
        flair = load_nifty_from_file(flair_list[idx])
        
        print('> Loading label...')
        r1 = load_nifty_from_file(r1_list[idx])  
        r2 = load_nifty_from_file(r2_list[idx])  
        mask = create_mask(t1)

        t1, t2, flair = adjust_images_task3(t1, t2, flair, mask)
        label = merge_task3_label(r1, r2)
        
        print("> Extracting patches...")        
        current_nr_extracted_patches = 0  # counts already extracted patches
        t1_patches = {}  # dictionary to save t1 patches
        t2_patches = {}  # dictionary to save t2 patches
        flair_patches = {}  # dictionary to save flair patches
        r1_patches = {} # dictionaty to save rater 1 patches
        r2_patches = {} # dictionary to save rater 2 patchs
        
        # make lists in dictionaries for each extracted patch size
        for size in patch_sizes:
            t1_patches[str(size)] = []
            t2_patches[str(size)] = []
            flair_patches[str(size)] = []
            r1_patches[str(size)] = []
            r2_patches[str(size)] = []

        # variables with sizes and ranges for searchable areas
        max_patch_size = max(patch_sizes)
        half_max_size = max_patch_size // 2
        max_row = label.shape[0] - max_patch_size // 2
        max_col = label.shape[1] - max_patch_size // 2
        
        # -----------------------------------------------------------------
        # EXTRACT POSITIVE PATCHES WITH LACUNES IN THE CENTER OF EACH PATCH
        # -----------------------------------------------------------------
        
        # cut off half of the biggest patch on the edges to create the searchable area -> to ensure that there will be
        # enough space for getting the patch
        searchable_label_area = label[half_max_size: max_row, half_max_size: max_col, :]
        
        # find all lacunes voxel indices in searchable area
        lacunes_inds = np.asarray(np.where(searchable_label_area == 1))
        patches_number = len(lacunes_inds[0])
        tot_lacunes_patches += patches_number*2
        print(f"NUMBER OF LACUNES: {patches_number}")
        
        # Extract all patches
        for i in range(patches_number):

            # get the coordinates of the lacunes around which the patch will be extracted
            x = lacunes_inds[0][i] + half_max_size
            y = lacunes_inds[1][i] + half_max_size
            z = lacunes_inds[2][i]

            # extract patches of different quadratic sizes with the lacunes voxel in the center of each patch
            # and duplicate them with an horizontal flip
            for size in patch_sizes:
                half_size = size // 2
                odd = size % 2
                
                random_t1_patch = t1[x - half_size:x + half_size+odd, y - half_size:y + half_size+odd, z]
                random_t2_patch = t2[x - half_size:x + half_size+odd, y - half_size:y + half_size+odd, z]
                random_flair_patch = flair[x - half_size:x + half_size+odd, y - half_size:y + half_size+odd, z]
                random_r1_patch = r1[x - half_size:x + half_size+odd, y - half_size:y + half_size+odd, z]
                random_r2_patch = r2[x - half_size:x + half_size+odd, y - half_size:y + half_size+odd, z]
                
                random_t1_patch_fl = np.flip(random_t1_patch, axis=1)
                random_t2_patch_fl = np.flip(random_t2_patch, axis=1)
                random_flair_patch_fl = np.flip(random_flair_patch, axis=1)
                random_r1_patch_fl = np.flip(random_r1_patch, axis=1)
                random_r2_patch_fl = np.flip(random_r2_patch, axis=1)

                # just sanity check if the patch is already in the list
                if any((random_t1_patch == x).all() for x in t1_patches[str(size)]):
                    print('Skip patch because already extracted. size:', size)
                    break
                else:
                    # append the extracted patches to the dictionaries
                    t1_patches[str(size)].append(random_t1_patch)
                    t2_patches[str(size)].append(random_t2_patch)
                    flair_patches[str(size)].append(random_flair_patch)
                    r1_patches[str(size)].append(random_r1_patch)
                    r2_patches[str(size)].append(random_r2_patch)
                    t1_patches[str(size)].append(random_t1_patch_fl)
                    t2_patches[str(size)].append(random_t2_patch_fl)
                    flair_patches[str(size)].append(random_flair_patch_fl)                    
                    r1_patches[str(size)].append(random_r1_patch_fl)
                    r2_patches[str(size)].append(random_r2_patch_fl)
                    
        print(f"Extracted {patches_number*2} patches, total patches = {tot_lacunes_patches}")
        label = np.ones([patches_number*2, 1])
        if patches_number > 0:
            for size in patch_sizes:
                directory = patch_extraction_dir
                file_name = t1_list[idx].split(os.sep)[-1].split('_')[0]
                np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'pos_t1', np.asarray(t1_patches[str(size)]))
                np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'pos_t2', np.asarray(t2_patches[str(size)]))
                np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'pos_flair', np.asarray(flair_patches[str(size)]))
                np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'pos_r1', np.asarray(r1_patches[str(size)]))
                np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'pos_r2', np.asarray(r2_patches[str(size)]))
                np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'pos_label', label)
                print('Positive Patches saved to', directory + "/" + str(size) + "/" + file_name + "_" + str(size) + '_***.npy')
        
        
    #####################################################
    # EXTRACT NEGATIVE PATCHES IN SAME NUMBER OF POSITIVE
    #####################################################
    patches_per_brain = tot_lacunes_patches // len(t1_list)
    tot_negative_patches = 0
    for idx in range(len(t1_list)):
        # load image and label stacks as matrices
        print('> Loading images...')
        t1 = load_nifty_from_file(t1_list[idx])
        t2 = load_nifty_from_file(t2_list[idx])
        flair = load_nifty_from_file(flair_list[idx])
        
        print('> Loading label...')
        r1 = load_nifty_from_file(r1_list[idx])  
        r2 = load_nifty_from_file(r2_list[idx])  
        mask = create_mask(t1)

        t1, t2, flair = adjust_images_task3(t1, t2, flair, mask)
        label = merge_task3_label(r1, r2)

        print("> Extracting patches...")
        current_nr_extracted_patches = 0  # counts already extracted patches
        t1_patches = {}  # dictionary to save t1 patches
        t2_patches = {}  # dictionary to save t2 patches
        flair_patches = {}  # dictionary to save flair patches
        r1_patches = {}
        r2_patches = {}
        # make lists in dictionaries for each extracted patch size
        for size in patch_sizes:
            t1_patches[str(size)] = []
            t2_patches[str(size)] = []
            flair_patches[str(size)] = []
            r1_patches[str(size)] = []
            r2_patches[str(size)] = []

        # variables with sizes and ranges for searchable areas
        max_patch_size = max(patch_sizes)
        half_max_size = max_patch_size // 2
        max_row = label.shape[0] - max_patch_size // 2
        max_col = label.shape[1] - max_patch_size // 2
	  
        # -----------------------------------------------------------
        # EXTRACT RANDOM EMPTY PATCHES
        # -----------------------------------------------------------
        # cut off half of the biggest patch on the edges to create the searchable area -> to ensure that there will be
        # enough space for getting the patch
        searchable_mask_area = mask[half_max_size: max_row, half_max_size: max_col, :]
        # find all brain voxel indices
        searchable_label_area = label[half_max_size: max_row, half_max_size: max_col, :]
        # find all vessel lacunes indices in searchable area
        brain_inds = np.asarray(np.where(searchable_mask_area == 1))               
        # remove lacunes coordinates from brain_inds
        brain_inds = np.asarray(np.where(np.logical_and(searchable_label_area == 0, searchable_mask_area==1)))
        
        # select random brain indexes 
        random_brain_inds = brain_inds[:, np.random.choice(brain_inds.shape[1], patches_per_brain, replace=False)]
        
        # find given number of random vessel indices
        for i in range(patches_per_brain):
            # get the coordinates of the random center around which the patch will be extracted
            x = random_brain_inds[0][i] + half_max_size
            y = random_brain_inds[1][i] + half_max_size
            z = random_brain_inds[2][i]

            # extract patches of different quadratic sizes
            for size in patch_sizes:
                half_size = size // 2
                odd = size % 2
                random_t1_patch = t1[x - half_size:x + half_size+odd, y - half_size:y + half_size+odd, z]
                random_t2_patch = t2[x - half_size:x + half_size+odd, y - half_size:y + half_size+odd, z]
                random_flair_patch = flair[x - half_size:x + half_size+odd, y - half_size:y + half_size+odd, z]
                random_r1_patch = r1[x - half_size:x + half_size+odd, y - half_size:y + half_size+odd, z]
                random_r2_patch = r2[x - half_size:x + half_size+odd, y - half_size:y + half_size+odd, z]

                # just sanity check if the patch is already in the list
                if any((random_t1_patch == x).all() for x in t1_patches[str(size)]):
                    print('Skip patch because already extracted. size:', size)
                    break
                else:
                    # append the extracted patches to the dictionaries
                    t1_patches[str(size)].append(random_t1_patch)
                    t2_patches[str(size)].append(random_t2_patch)
                    flair_patches[str(size)].append(random_flair_patch)
                    r1_patches[str(size)].append(random_r1_patch)
                    r2_patches[str(size)].append(random_r2_patch)
                    tot_negative_patches += 1
                    
        print(f"Extracted {patches_per_brain} patches, total patches = {tot_negative_patches}")
        label = np.zeros([patches_per_brain, 1])
        for size in patch_sizes:
            directory = patch_extraction_dir
            file_name = t1_list[idx].split(os.sep)[-1].split('_')[0]
            np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'neg_t1', np.asarray(t1_patches[str(size)]))
            np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'neg_t2', np.asarray(t2_patches[str(size)]))
            np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'neg_flair', np.asarray(flair_patches[str(size)]))
            np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'neg_r1', np.asarray(r1_patches[str(size)]))
            np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'neg_r2', np.asarray(r2_patches[str(size)]))
            np.save(directory + "/" + str(size) + "/" + file_name + "_" + str(size) + 'neg_label', label)
            print('Negative Patches saved to', directory + "/" + str(size) + "/" + file_name + "_" + str(size) + '_***.npy')
        print()
    
    print(f"TOTAL PATCHES WITH LACUNES: {tot_lacunes_patches}")
    print('DONE')
    
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="D:/Datasets/valdo/Task3/Val", type=str, 
                        help='Directory containing images after normalization and registration')
    parser.add_argument('--patch_extraction_dir', default="D:/Datasets/valdo_patches/Task3/Val", type=str, 
                        help='Directory for saving the images after the patch extraction process.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))