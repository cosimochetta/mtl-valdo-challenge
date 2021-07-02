import logging
import os
import argparse
import sys
import tempfile
import json
from glob import glob
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.ndimage import label

# %%
parser = argparse.ArgumentParser(description='Task 3 rater count')


parser.add_argument('--directory', default="D:/Datasets/valdo_norm_update_mask/Task3/", type=str, help='dataset root')
parser.add_argument('--csv_path', default="./csv_files/task3_stats_reg.csv", type=str, help='index name')
opt = parser.parse_args()


df = pd.DataFrame(columns=['subject',
                            'r1_pixel',
                            'r2_pixel',
                            'r1_object',
                            'r2_object',
                            'concording_pixel',
                            'union_pixel',
                            'concording_pixel_perc'
                            ],
                  dtype='object')


rater1_paths = glob(opt.directory + "**/*Rater1*.nii.gz", recursive=True) + glob(opt.directory + "**/*Rater3*.nii.gz", recursive=True)
rater2_paths = glob(opt.directory + "**/*Rater2*.nii.gz", recursive=True)+ glob(opt.directory + "**/*Rater4*.nii.gz", recursive=True)

# %%
for r1_path, r2_path in zip(rater1_paths, rater2_paths):
    subject = os.path.basename(os.path.dirname(r1_path))
    rater1 = nib.load(r1_path).get_fdata()
    rater2 = nib.load(r2_path).get_fdata()
    rater1 = np.round(rater1)
    rater2 = np.round(rater2)
    r1_pixel_count = np.sum(rater1)
    r2_pixel_count = np.sum(rater2)
    
    _, r1_num_objects = label(rater1)
    _, r2_num_objects = label(rater2)
    
    concording_pixel = np.sum(np.logical_and(rater2, rater1), dtype=int)
    union_pixel = np.sum(np.logical_or(rater2, rater1), dtype=int)
    concording_pixel_perc = concording_pixel / union_pixel if union_pixel != 0 else None
    
    print(f"Subject {subject}")
    print(f"\t r1p: {r1_pixel_count} \t r2p: {r2_pixel_count} \t r1o: {r1_num_objects} \t r2o: {r2_num_objects}")
    print(f"\t c_pix: {concording_pixel} \t u_pix: {union_pixel} \t c_pix_p: {concording_pixel_perc}")
    
    df = df.append(pd.DataFrame([[subject, 
                                  r1_pixel_count,
                                  r2_pixel_count,
                                  r1_num_objects, 
                                  r2_num_objects,
                                  concording_pixel,
                                  union_pixel,
                                  concording_pixel_perc]],
                                columns=df.columns))
    
df.to_csv(opt.csv_path, index=False)

