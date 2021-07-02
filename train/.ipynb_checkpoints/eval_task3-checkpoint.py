import logging
import os
import argparse
import shutil
import sys
import tempfile
import json
from glob import glob
from datetime import datetime
import nibabel as nib
import numpy as np

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
    Resized,
    ScaleIntensityd,
    ToTensord,    
)

from utils import Task3_DiceMetric

parser = argparse.ArgumentParser(description='Task 3 training')
#/cluster/project0/MAS/cosimo/valdo/Task3
#C:/Datasets/valdo/Task3
# data path
parser.add_argument('--directory', default="C:/Datasets/valdo_normalized/Task3", type=str, help='dataset root')
parser.add_argument('--index', default="my_index.json", type=str, help='index name')
# model path
parser.add_argument('--model_path', default="./models/best_model.pth")
parser.add_argument('--store_image_path', default="./validation_images")
opt = parser.parse_args()

# %%
if __name__ == '__main__':
    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    torch.manual_seed(42)
    
    # %%            
    index_path = os.path.join(opt.directory, opt.index)
    with open(index_path) as f:
        files = json.load(f)

    # define transforms for image and segmentation
    test_transforms = Compose(
            [
                LoadImaged(keys=["t1", "labels", "seg1", "seg2"]),
                AddChanneld(keys=["t1", "labels", "seg1", "seg2"]),
                CropForegroundd(keys=["t1", "labels", "seg1", "seg2"], source_key="t1"),
                #NormalizeIntensityd(keys="t1", nonzero=True, channel_wise=True),
                ScaleIntensityd(keys="t1"),
                Resized(keys=["t1", "labels", "seg1", "seg2"], spatial_size=[256, 256, 128]),
                ToTensord(keys=["t1", "labels", "seg1", "seg2"]),
            ]
    )
    
    
    # create a validation data loader
    val_ds = Dataset(data=files, transform=test_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    
    # %%
    # create UNet, DiceLoss and Adam optimizer
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        #channels=(2,4),
        channels=(30, 60, 120, 240, 480),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    model = torch.load(opt.model_path)
    #state_dict = torch.load(opt.model_path, map_location="cpu")
    #model.load_state_dict(state_dict)

    # %%
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    post_val = AsDiscrete(threshold_values=True)
    
    # %%
        
    validation_size = len(val_loader)
    model.eval()
    with torch.no_grad():
        task3_metrics = Task3_DiceMetric(include_background=False)
        val_iterator = iter(val_loader)
        for i in range(validation_size):
            filename = os.path.basename(files[i]['t1'])[0:7]
            val_data = next(val_iterator)
            val_images, val_labels = val_data["t1"].to(device), val_data["labels"].to(device)
            val_seg1, val_seg2 = val_data["seg1"].to(device), val_data["seg2"].to(device)
            roi_size = (128, 128, 64)
            sw_batch_size = 2
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = post_trans(val_outputs)
                                
            metric_label, metric_seg = task3_metrics(post_trans(val_outputs), 
                                                     post_val(val_labels), 
                                                     post_val(val_seg1),
                                                     post_val(val_seg2))
            
            print("file {}: \n\tmerged label dice={:.6f}\n\tsegmentation mean dice={:.6f}"
              .format(filename, metric_label.item(),metric_seg.item()))
            
            # Save images
            img_data = nib.Nifti1Image(np.array(val_images[0][0].cpu()), np.eye(4))
            img_label = nib.Nifti1Image(np.array(val_labels[0][0].cpu()), np.eye(4))
            img_seg1 = nib.Nifti1Image(np.array(val_seg1[0][0].cpu()), np.eye(4))
            img_seg2 = nib.Nifti1Image(np.array(val_seg2[0][0].cpu()), np.eye(4))
            img_pred = nib.Nifti1Image(np.array(val_outputs[0][0].cpu()), np.eye(4))
            nib.save(img_data, os.path.join(opt.store_image_path, filename + "_data.nii.gz"))
            nib.save(img_label, os.path.join(opt.store_image_path, filename + "_label.nii.gz"))
            nib.save(img_seg1, os.path.join(opt.store_image_path, filename + "_seg1.nii.gz"))
            nib.save(img_seg2, os.path.join(opt.store_image_path, filename + "_seg2.nii.gz"))
            nib.save(img_pred, os.path.join(opt.store_image_path, filename + "_prediction.nii.gz"))
