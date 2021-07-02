# -*- coding: utf-8 -*-
"""

@author: minoc

# =============================================================================
# Register T1 Images to target file the applies same affine registration to
# T2, FLAIR and label.
# T1, T2, FLAIR interpolation: trilinear
# Labels interpolation: nearest
# =============================================================================
"""

import os
import argparse
from glob import glob
import subprocess

parser = argparse.ArgumentParser(description='Task 3 training')

parser.add_argument('--input_folder', default="/cluster/project0/MAS/cosimo/valdo/Task3/", type=str, help='dataset root')
parser.add_argument('--output_folder', default="/cluster/project0/MAS/cosimo/valdo_registration/", type=str, help='dataset root')
parser.add_argument('--target_image', default="sub-201/sub-201_space-T1_desc-masked_T1.nii.gz", type=str, help='index name')
parser.add_argument('--niftyreg', default="/cluster/project0/MAS/cosimo/niftyreg")
opt = parser.parse_args()

t1_paths = glob(opt.input_folder + "**/*_T1.nii.gz", recursive=True)
target_path = os.path.join(opt.input_folder, opt.target_image)


reg_aladin = os.path.join(opt.niftyreg, "reg_aladin")
reg_resample = os.path.join(opt.niftyreg, "reg_resample")

print(f"target file path: {target_path}")
print(f"reg_aladin path: {reg_aladin}")
print(f"reg_resample path: {reg_resample}")

for input_path in t1_paths:
    print(f"Processing image {input_path}")
    input_folder = os.path.dirname(input_path)
    task_sub_directory = input_folder[-13:]
    output_folder = os.path.join(opt.output_folder, task_sub_directory)
    
    affine_transform_path = os.path.join(output_folder, "affine_transform.txt")
    output_path = os.path.join(output_folder, os.path.basename(input_path))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print("Registration...")
    result = subprocess.run([reg_aladin, 
                    '-ref', target_path, 
                    '-flo', input_path, 
                    '-aff', affine_transform_path,
                    '-res', output_path])
    
    print(f"Registration complete, return code: {result.returncode}")
    result.check_returncode()
    if result.stderr:
        print(result.stderr)
    
    print(f"Apply affine matrix to other files in {input_folder}")
    
    for nii_file_path in glob(input_folder + "/*.nii.gz"):
        output_path = os.path.join(output_folder, os.path.basename(nii_file_path))
        interpolation = '3'
        if "_CMB" in nii_file_path or '_Lacunes' in nii_file_path:
            interpolation = '0'
        
        
        result = subprocess.run([reg_resample, 
                    '-ref', target_path, 
                    '-flo', nii_file_path, 
                    '-inter', interpolation,
                    '-trans', affine_transform_path,
                    '-res', output_path])
        
        print(f"Resampling complete, return code: {result.returncode}")
        result.check_returncode()
        if result.stderr:
            print(result.stderr)
        
# #$ -S /bin/sh
# #$ -l tmem=3.9G,h_vmem=3.9G
# #$ -wd /cluster/project0/MAS/cosimo/valdo_normalized
# #$ -o /cluster/project0/MAS/cosimo/outputs
# #$ -e /cluster/project0/MAS/cosimo/outputs
# source activate /cluster/project0/MAS/cosimo/env
# echo "running registration script"
# /cluster/project0/MAS/cosimo/env/bin/python3.6 -u apply_registration.py