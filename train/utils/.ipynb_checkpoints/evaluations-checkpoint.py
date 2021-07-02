import logging
import os
import argparse
import shutil
import sys
import tempfile
import json
from glob import glob
from datetime import datetime
import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from monai.config import print_config
from monai.data import list_data_collate, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    Resized,
    ScaleIntensityd,
    ToTensord,    
)
from monai.data.utils import compute_importance_map, dense_patch_slices    

def sliding_window_3dimg_2dmodel(data, label, model, roi_size=(96, 96), overlap=0.25, mode='constant'):
    '''
    ------
    INPUTS
    ------
    img: Tensor of shape BMHWD, M=input channels
    label: Tensor of shape BM'HWD, M'=output channels,
    model: model used for inference
    
    ------
    OUTPUTS
    ------
    prediction: Tensor of shape BMHWD with predicted label
    
    '''
    
    prediction = torch.empty(label.shape)
    #print("Processing image")
    with torch.no_grad():
        for b in range(data.shape[0]):
            for i in range(data.shape[-1]):
                #print(f"Slice {i+1} / {data.shape[1]}", end="\r")
                data_slice = data[b,:,:,:,i].unsqueeze(0)
                pred_slice = sliding_window_inference(data_slice, 
                                                        roi_size, 
                                                        128, 
                                                        model,
                                                        mode=mode,
                                                        overlap=overlap,
                                                        )
                prediction[b,:,:,:,i] = pred_slice[0]
    return prediction
    
def sliding_window_interval_inference(inputs, predictor, sw_batch_size=1024, roi_size=(51,51), interval=(10,10), sw_device='cpu', verbose=False, *args, **kwargs):
    batch_size = inputs.shape[0]
    image_size = inputs .shape[-2:]
    slices = dense_patch_slices(image_size, roi_size, interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    if verbose:
        print(f"Total number of slices: {total_slices}")
    
    output_image = torch.tensor(0.0, device=sw_device)
    _initialized = False
    for slice_g in range(0, total_slices, sw_batch_size):
        if verbose:
            print(f"{slice_g} / {total_slices}", end='\r')
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        seg_prob = predictor(window_data, *args, **kwargs).to(sw_device)  # batched patch segmentation

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.ones(output_shape, dtype=torch.float32, device=sw_device) * -100
            _initialized = True

        # store the result in the proper location of the full output. 
        for idx, original_idx in zip(slice_range, unravel_slice):
            x_point = (original_idx[2].start + original_idx[2].stop) // 2
            y_point = (original_idx[3].start + original_idx[3].stop) // 2
            original_idx[2] = slice(x_point, x_point+1)
            original_idx[3] = slice(y_point, y_point+1)            
            output_image[original_idx] = seg_prob[idx - slice_g]   

    return output_image

def sliding_window_point_inference(inputs, predictor, sw_batch_size=1024, roi_size=(51,51), interval=(10,10), sw_device='cpu', verbose=False, *args, **kwargs):
    batch_size = inputs.shape[0]
    image_size = inputs .shape[-2:]
    slices = dense_patch_slices(image_size, roi_size, interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    if verbose:
        print(f"Total number of slices: {total_slices}")
    
    output_image = torch.tensor(0.0, device=sw_device)
    _initialized = False
    for slice_g in range(0, total_slices, sw_batch_size):
        if verbose:
            print(f"{slice_g} / {total_slices}", end='\r')
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        seg_prob = predictor(window_data, *args, **kwargs).to(sw_device)  # batched patch segmentation

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image = torch.ones(output_shape, dtype=torch.float32, device=sw_device) * -100
            _initialized = True

        # store the result in the proper location of the full output. 
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image[original_idx] = seg_prob[idx - slice_g]   

    return output_image



def TEST_sliding_window_3dimg_2dmodel():
    model = UNet(
        dimensions=2,
        in_channels=1,
        out_channels=2,
        channels=(64,128,256,512),
        strides=(2, 2, 2),
        num_res_units=2,
        norm='BATCH',
        #dropout=0.2,
    )
    metric = DiceMetric(include_background=False, reduction="mean")
    post_inference = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    data = torch.rand([1, 192, 512, 512])
    label = torch.rand([2, 192, 512, 512]) >= 0.5
    
    score, prediction = sliding_window_3dimg_2dmodel(data, label, model)
    return score, prediction
    
    
    
    
    
