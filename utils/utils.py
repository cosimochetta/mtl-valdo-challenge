# -*- coding: utf-8 -*-
"""
Created on Sun May 16 22:01:34 2021

@author: minoc
"""
from typing import Union
from glob import glob
import nibabel as nib
import numpy as np
import os

# =============================================================================
# File name getters
# =============================================================================

def get_file_list_task2(root_dir, filt=None):
    file_list = glob(root_dir + "/**/*", recursive=True)
    file_list = [file for file in file_list if '.nii.gz' in file]
    if filt:
        file_list = [file for file in file_list if filt in file]

    fl_T1 = sorted([file for file in file_list if '_T1.nii' in file])
    fl_T2 = sorted([file for file in file_list if '_T2.nii' in file])
    fl_FLAIR = sorted([file for file in file_list if '_T2S.nii' in file])
    fl_CMB = sorted([file for file in file_list if '_CMB.nii' in file])
    
    assert len(fl_T1) == len(fl_T2) == len(fl_FLAIR) == len(fl_CMB)
    return fl_T1, fl_T2, fl_FLAIR, fl_CMB

def get_file_list_task3(root_dir):
    file_list = glob(root_dir + "/**/*.nii.gz", recursive=True)
 
    fl_T1 = sorted([file for file in file_list if '_T1' in file])
    fl_T2 = sorted([file for file in file_list if '_T2' in file])
    fl_FLAIR = sorted([file for file in file_list if '_FLAIR' in file])
    fl_R1 = sorted([file for file in file_list if 'Rater1' in file or 'Rater3' in file])
    fl_R2 = sorted([file for file in file_list if 'Rater2' in file or 'Rater4' in file])
    
    assert len(fl_T1) == len(fl_T2) == len(fl_FLAIR) == len(fl_R1) == len(fl_R2)
    return fl_T1, fl_T2, fl_FLAIR, fl_R1, fl_R2

def get_nifty_list(root_dir, name=""):
    """
    Get all nifty file in root_dir that contains 'name' in the path
    """
    
    file_list = glob(root_dir + "/**/*.nii.gz", recursive=True)
    file_list = sorted([file for file in file_list if name in file])
    return file_list
    

def create_path_from_filename(file_path, new_name):
    """
    Parameters
    ----------
    file_path : str
        path of type : PATH/TO/FOLDER/sub-xxx/sub-xxx_filename
    new_name : TYPE
        filename of the generated file
        
    Returns
    -------
    str: PATH/TO/FOLDER/sub-xxx/sub-xxx_new_name
        
    """        
    dir_folder = os.path.dirname(file_path)
    basename = os.path.basename(dir_folder)
    return os.path.join(dir_folder, basename + "_" + new_name)
    
# =============================================================================
# Loader/Saver
# =============================================================================

def load_nifty_from_file(path, loader='nib'):
    if loader == 'nib':
        nifti_orig = nib.load(path)
        return nifti_orig.get_data()  # transform the images into np.ndarrays
    
    raise NameError('No loader found')
    

def load_niftyimg_from_file(path, loader='nib'):
    if loader == 'nib':
        nifti_orig = nib.load(path)
        return nifti_orig
    
    raise NameError('No loader found')    


def save_nifty_file(data, path, affine=np.eye(4)):
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)

# =============================================================================
# Extract bounding box slices from mask
# =============================================================================

def bounding_box_slices(data):
    
    coords = np.where(data == 1)
    bounding_box = []
    for coord in coords:
        bounding_box.append(slice(np.min(coord), np.max(coord)))
    return bounding_box

# =============================================================================
# Task 3 function to adjust registration
# =============================================================================

def merge_task3_label(rater1, rater2):
    '''
    Merge the two raters with union operation
    '''
    
    labels = ((rater1 > 0.5) | ( rater2 > 0.5)).astype(float)
    return labels

def create_mask(data, mode: Union[int, float, str]='median', value: Union[float, int]=0):
    """
    Create mask of the data

    Parameters
    ----------
    data : numpy array
    mode : Union(str, int)
        mode to create mask.
        - 'median': consider the median of data as background
        - 'value': consider value as background
    value: Union[float, int]
        the value that will be considered background if mode == 'value', 

    Returns
    -------
    mask : numpy array
        binary mask of data
    """
    
    assert mode in ['median', 'value']
    
    bg_value = value
    if mode == 'median':
        bg_value = np.median(data)

    mask = (data != bg_value).astype(int)    
    return mask

def cut_outliers_task3(data, mask, cut_value):
    '''
    Set to mean values greater that cut_value
    The mean is computed ignoring the value in the mask
    '''
    
    # invert mask for masked array
    mask = mask == 0
    ma = np.ma.array(data, mask=mask)
    mean = ma.mean()
    
    data[data > cut_value] = mean
    return data
    
def rescale_data(data, mask, min_value=None, max_value=None):
    """
    rescale data between a fixed range, ignore mask data

    Parameters
    ----------
    data : array
    mask : array
        mask of the data, must have the same shape.
        has 1 where the data is valid, 0 where background/invalid
    min_value : optional
        minimum target value, if None computed using the min value of masked data.
        The default is None.
    max_value : TYPE, optional
        minimum target value, if None computed using the min value of masked data.      
        The default is None.

    Returns
    -------
    data : array
        rescaled array with range [min_value, max_value].
    """
    
    assert data.shape == mask.shape
    
    # Invert mask for masked array
    mask = mask == 0
    ma = np.ma.array(data, mask=mask)
    
    if not min_value:
        min_value = ma.min()
    if not max_value:
        max_value = ma.max()
    
    ma = (ma - min_value) / (max_value - min_value)
    data = ma.filled(0)
    return data

def adjust_images_task3(t1, t2, flair, mask, max_values=[150, 150, 180]):
    """
    Remove outliers and rescale images of Task3 between [0, 1]
    
    Adjust the images of task 3 after registration and histogram normalization.
    Cut value are used to rescale the images using the same range as SABRE data 
    tends to have a long tail agter the histogram normalization.
    Most of the values removed come from the border of the images and they should
    not affect the final results.
    """
    
    assert t1.shape == t2.shape == flair.shape == mask.shape
    t1 = cut_outliers_task3(t1, mask, cut_value=max_values[0])
    t1 = rescale_data(t1, mask, min_value=0, max_value=max_values[0])
    
    t2 = cut_outliers_task3(t2, mask, cut_value=max_values[1])
    t2 = rescale_data(t2, mask, min_value=0, max_value=max_values[1])
    
    flair = cut_outliers_task3(flair, mask, cut_value=max_values[2])
    flair = rescale_data(flair, mask, min_value=0, max_value=max_values[2])
    
    return t1, t2, flair
