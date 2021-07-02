import os
import json
import numpy as np

from PIL import Image
import nibabel as nib
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from monai.data import list_data_collate, Dataset
from scipy.stats import multivariate_normal
import random
import scipy.ndimage
import glob

from monai.transforms import (
    Activations,
    AddChannel,
    AddChanneld,
    AsDiscrete,
    AsDiscreted,
    Compose,
    CenterSpatialCropd,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    Resized,
    ScaleIntensityd,
    ToTensor,
    ToTensord,    
)
    
from .transforms import RemoveOutliersd, ResetBackgroundd
from .utils import get_file_list_task3_patches



class ValdoDatasetTask3Evaluation(Dataset):
    def __init__(self, root_dir):
        # Define attributes
        self.root_dir = root_dir
        self.data_list = []
        self.loader = Compose([
            LoadImaged(keys=['t1', 't2', 'flair', 'r1', 'r2']),
            AddChanneld(keys=['t1', 't2', 'flair', 'r1', 'r2']),
            ResetBackgroundd(keys=['t1', 't2', 'flair'], mask='t1'),
            #CropForegroundd(keys=['t1', 't2', 'flair', 'r1', 'r2'], source_key='t1'),
            RemoveOutliersd(keys=['t1', 't2', 'flair'], cut_value=[150, 150, 150]),
            ScaleIntensityd(keys=["t1", "t2", "flair"]),
            ])
        self.transform = ToTensord(keys=['image', 't1', 't2', 'flair', 'r1', 'r2'])
               
        # Load data index
        file_list = glob.glob(self.root_dir + "/**/*", recursive=True)
        file_list = [file for file in file_list if '.nii.gz' in file]

        fl_T1 = [file for file in file_list if '_T1' in file][0:1]
        fl_T2 = [file for file in file_list if '_T2' in file][0:1]
        fl_FLAIR = [file for file in file_list if '_FLAIR' in file][0:1]
        fl_R1 = [file for file in file_list if 'Rater1' in file or 'Rater3' in file][0:1]
        fl_R2 = [file for file in file_list if 'Rater2' in file or 'Rater4' in file][0:1]
        l = len(fl_T1)
        i = 1
        
        for (t1, t2, flair, r1, r2) in zip(fl_T1, fl_T2, fl_FLAIR, fl_R1, fl_R2):
            print(f"Loading {i} / {l}", end='\r')
            data = self.loader({'t1': t1, 't2': t2, 'flair': flair, 'r1': r1, 'r2': r2})
            data['image'] = np.concatenate((data['t1'], data['t2'], data['flair']), 0)
            data['r1'] =  np.concatenate((1-data['r1'], data['r1']), 0)
            data['r2'] =  np.concatenate((1-data['r2'], data['r2']), 0)
            self.data_list.append(data)
            i += 1
            
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        out_data = self.transform(data)
        
        return out_data

class ValdoDatasetTask3(Dataset):

    def __init__(self, root_dir, roi_size, margin=0, crop_sample=1, split="train"):
        # Define attributes
        self.root_dir = root_dir
        self.data_list = []
        self.split = split
        if split == "train":
            self.loader = Compose([
                LoadImaged(keys=['t1', 't2', 'flair', 'r1', 'r2']),
                AddChanneld(keys=['t1', 't2', 'flair', 'r1', 'r2']),
                ResetBackgroundd(keys=['t1', 't2', 'flair'], mask='t1'),
                RemoveOutliersd(keys=['t1', 't2', 'flair'], cut_value=[150, 150, 150], mask='t1'),
                #CropForegroundd(keys=['t1', 't2', 'flair', 'r1', 'r2'], 
                #              source_key="t1",
                #              select_fn=lambda x: x > np.median(x),
                #              margin=margin),
                
                ScaleIntensityd(keys=["t1", "t2", "flair"]),
                ])
            
            
            self.transform = Compose([
                RandCropByPosNegLabeld(
                        keys=['image', 'r1', 'r2', 'label', 't1'],
                        label_key="label",
                        spatial_size=roi_size,
                        pos=1,
                        neg=1,
                        num_samples=crop_sample,
                        image_key='t1',
                    ),
                RandFlipd(keys=['image', 'r1', 'r2', 'label', 't1'],
                          prob=0.5, 
                          spatial_axis=0),
                RandAffined(
                    keys=['image', 'label', 'r1', 'r2', 't1'],
                    mode=('bilinear', 'nearest', 'nearest', 'nearest', 'bilinear'),
                    prob=0.5, spatial_size=roi_size,
                    rotate_range=(0, 0, np.pi/15),
                    scale_range=(0.1, 0.1, 0.1)
                ),
                ToTensord(keys=['image', 'r1', 'r2', 'label', 't1']),
                ])
        else: 
            self.loader = Compose([
                LoadImaged(keys=['t1', 't2', 'flair', 'r1', 'r2']),
                AddChanneld(keys=['t1', 't2', 'flair', 'r1', 'r2']),
                ResetBackgroundd(keys=['t1', 't2', 'flair'], mask='t1'),
                RemoveOutliersd(keys=['t1', 't2', 'flair'], cut_value=[150, 150, 150]),
                ScaleIntensityd(keys=["t1", "t2", "flair"]),
                ])
            self.transform = ToTensord(keys=['image', 'r1', 'r2', 'label', 't1'])
               
        # Load data index
        file_list = glob.glob(self.root_dir + "/**/*", recursive=True)
        file_list = [file for file in file_list if '.nii.gz' in file]

        fl_T1 = [file for file in file_list if '_T1' in file]
        fl_T2 = [file for file in file_list if '_T2' in file]
        fl_FLAIR = [file for file in file_list if '_FLAIR' in file]
        fl_R1 = [file for file in file_list if 'Rater1' in file or 'Rater3' in file]
        fl_R2 = [file for file in file_list if 'Rater2' in file or 'Rater4' in file]
        l = len(fl_T1)
        i = 1
        
        for (t1, t2, flair, r1, r2) in zip(fl_T1, fl_T2, fl_FLAIR, fl_R1, fl_R2):
            print(f"Loading {i} / {l}", end='\r')
            data = self.loader({'t1': t1, 't2': t2, 'flair': flair, 'r1': r1, 'r2': r2})
            self.data_list.append(data)
            i += 1
            
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        r_sum = 0.5 * (data['r1'] + data['r2'])
        out_data = {}
        out_data['label'] = np.round(r_sum)
        out_data['image'] = np.concatenate((data['t1'], data['t2'], data['flair']), 0)
        out_data['r1'] = data['r1']
        out_data['r2'] = data['r2']
        out_data['t1'] = data['t1']
        # Set image treshold of RandCropByPosNegLabeld
        if self.split == "train":
            self.transform.transforms[0].image_threshold = np.median(data['t1'])
        
        out_data = self.transform(out_data)
        
        return out_data

class ValdoDatasetTask3Patches(Dataset):

    def __init__(self, root_dir, split="Train", patch_size=64, data_augm=None):
        # Define attributes
        self.root_dir = root_dir
        self.data_list = []
        self.split = split
        
        self.data_augm = data_augm
        self.to_tensor = Compose([
                AddChanneld(keys=['t1','t2','flair','r1','r2']),
                CenterSpatialCropd(keys=['t1','t2','flair','r1','r2'], roi_size=patch_size),
                ToTensord(keys=['t1','t2','flair','r1','r2']),
        ])
        
        # Load data index
        t1_paths, t2_paths, flair_paths, r1_paths, r2_paths, label_paths = get_file_list_task3_patches(root_dir)
        print("Loading data...")
        print(f"{1} / {len(t1_paths)}\n", end="\r")

        self.t1_data = np.load(t1_paths[0])
        self.t2_data = np.load(t2_paths[0])
        self.flair_data = np.load(flair_paths[0])
        self.r1_data = np.load(r1_paths[0])
        self.r2_data = np.load(r2_paths[0])
        self.label_data = np.load(label_paths[0])

        for i in range(1, len(t1_paths)):
            print(f"{i+1} / {len(t1_paths)}\n", end="\r")
            t1 = np.load(t1_paths[i])
            t2 = np.load(t2_paths[i])
            flair = np.load(flair_paths[i])
            r1 = np.load(r1_paths[i])
            r2 = np.load(r2_paths[i])
            label = np.load(label_paths[i])

            self.t1_data = np.concatenate((self.t1_data, t1), axis=0)
            self.t2_data = np.concatenate((self.t2_data, t2), axis=0)
            self.flair_data = np.concatenate((self.flair_data, flair), axis=0)
            self.r1_data = np.concatenate((self.r1_data, r1), axis=0)
            self.r2_data = np.concatenate((self.r2_data, r2), axis=0)
            self.label_data = np.concatenate((self.label_data, label), axis=0)

                
    def get_data_dictionary(self, idx):
        return  {
                't1': self.t1_data[idx],
                't2': self.t2_data[idx],
                'flair': self.flair_data[idx],
                'r1': self.r1_data[idx],
                'r2': self.r2_data[idx],
                'label': self.label_data[idx]
                }
    
    def __len__(self):
        return len(self.t1_data)

    def __getitem__(self, idx):
        data = self.get_data_dictionary(idx)
        
        if self.data_augm:
            data = self.data_augm(data)
        else:
            data = self.to_tensor(data)
        
        data['img'] = torch.cat((data['t1'], data['t2'], data['flair']), dim=0)
        data['r1'] = torch.vstack((1-data['r1'], data['r1']))
        data['r2'] = torch.vstack((1-data['r2'], data['r2']))
        return data
