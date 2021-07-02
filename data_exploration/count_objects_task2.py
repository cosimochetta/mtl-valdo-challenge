import os
import argparse
from glob import glob
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.ndimage import label

# %%
parser = argparse.ArgumentParser(description='Task 2 stats')


parser.add_argument('--directory', default="D:/Datasets/valdo_registration/Task2/", type=str, help='dataset root')
parser.add_argument('--csv_path', default="./csv_files/task2_stats_norm.csv", type=str, help='index name')
opt = parser.parse_args()


df = pd.DataFrame(columns=['subject',
                            'pixel',
                            'object',
                            ],
                  dtype='object')


paths = glob(opt.directory + "**/*_CMB.nii.gz", recursive=True)

# %%
for path in paths:
    subject = os.path.basename(os.path.dirname(path))
    cmb = nib.load(path).get_fdata()
    
    cmb = np.round(cmb)

    cmb_pixel_count = np.sum(cmb)
    
    _, cmb_num_objects = label(cmb)
        
    print(f"Subject {subject}")
    print(f"\t cmb_p: {cmb_pixel_count} \t cmb_o: {cmb_num_objects}")
    
    df = df.append(pd.DataFrame([[subject, 
                                  cmb_pixel_count,
                                  cmb_num_objects, 
                                  ]],
                                columns=df.columns))
    
df.to_csv(opt.csv_path, index=False)
    