# -*- coding: utf-8 -*-

import logging
import os
import argparse
import shutil
import sys
import json
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
from monai.data import list_data_collate

from torch.utils.tensorboard import SummaryWriter

from monai.config import print_config
from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric

from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    RandAffined,
    CenterSpatialCropd,
    RandFlipd,
    RandSpatialCropd,
    ToTensord,
    )

from utils import get_optimizer, get_scheduler
from utils import Task3SegmentationEvaluator
from utils import ValdoDatasetTask3Patches, ValdoDatasetTask3Evaluation

from segmentation_net import get_segmentation_net

# %% 
parser = argparse.ArgumentParser(description='Task 3 training')
#/cluster/project0/MAS/cosimo/valdo/Task3
#C:/Datasets/valdo/Task3
# data path
parser.add_argument('--directory', default="D:/Datasets/valdo_patches/Task3", type=str, help='dataset root')
# run settings
parser.add_argument('--batch_size', default=128, type=int, help='batch  size')
parser.add_argument('--val_interval', default=1, type=int, help='validate every n epochs')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--optimizer', default="Adam", type=str, choices=['Adam', 'SGD', 'Novograd'],)
parser.add_argument('--optim_args', default='{}', type=str)
parser.add_argument('--scheduler', default="None", type=str, choices=['None', 'LambdaLR', 'MultiplicativeLR'])
parser.add_argument('--scheduler_args', default="{}", type=str)
parser.add_argument('--original_size',  default=64,  type=int)
parser.add_argument('--patch_size',  default=48,  type=int)
parser.add_argument('--log_folder', default="runs", type=str)
parser.add_argument('--run_info', default="", type=str)
parser.add_argument('--run_model', default="unet", type=str, choices=['unet', 'pnet'])

opt = parser.parse_args()


# %%
if __name__ == '__main__':
    
    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    torch.manual_seed(42)
        
    # Create run log path and info
    dt = datetime.today()
    run_name = f"{dt.strftime('%Y-%m-%d_%H-%M')}_{opt.run_model}_lr_{opt.lr}_s_{opt.patch_size}_{opt.optimizer}_{opt.scheduler}"
    log_path = os.path.join(os.curdir, opt.log_folder, run_name)
    
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    writer = SummaryWriter(log_path)

    with open(os.path.join(log_path, "info.txt"), "w") as info_file:
        info_file.write(str(opt))
    
    # %% Data augmentation
    
    data_augm = Compose([
                AddChanneld(keys=['t1','t2','flair','r1','r2']),
                CenterSpatialCropd(keys=['t1','t2','flair','r1','r2'], roi_size=opt.original_size),
                RandSpatialCropd(keys=['t1','t2','flair','r1','r2'], 
                                 roi_size=opt.patch_size, 
                                 random_center=True,
                                 random_size=False),
               RandAffined(
                   keys=['t1', 't2', 'flair', 'r1', 'r2'],
                   mode=('bilinear', 'bilinear', 'bilinear', 'nearest','nearest'),
                   prob=0.9, spatial_size=(opt.patch_size, opt.patch_size),
                   rotate_range=(0, np.pi/15),
                   scale_range=(0.05, 0.05)
               ),
               ToTensord(keys=['t1','t2','flair','r1','r2']),
        ])
    
    
    # %% Load data
                
    train_path = os.path.join(opt.directory, "Train", "64")
    val_path = os.path.join(opt.directory, "Val", "64")
    val_full_path = os.path.join(opt.directory, "Val_full")
    test_full_path = os.path.join(opt.directory, "Test_full")

    # %%
    
    train_ds = ValdoDatasetTask3Patches(root_dir=train_path, 
                                 patch_size=opt.original_size, 
                                 #data_augm=data_augm,
                                 split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    
    # %%
    
    # create a validation data loader
    val_ds_full = ValdoDatasetTask3Evaluation(root_dir=val_full_path)
    val_loader_full = DataLoader(
        val_ds_full,
        batch_size=1,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    
    
    # create test data loader
    test_ds_full = ValdoDatasetTask3Evaluation(root_dir=test_full_path)
    test_loader_full = DataLoader(
        test_ds_full,
        batch_size=1,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    
    # %%
    
    model = get_segmentation_net(opt.run_model).to(device)
    
    loss_function = GeneralizedDiceLoss(softmax=True)
    
    optimizer = get_optimizer(opt.optimizer, model.parameters(), opt.lr, kwargs=json.loads(opt.optim_args))
    scheduler = get_scheduler(opt.scheduler, optimizer, kwargs=json.loads(opt.scheduler_args))
    
    # Validation metrics
    metric = DiceMetric(include_background=False, reduction="mean")
    post_inference = Compose([Activations(softmax=True), AsDiscrete(threshold_values=True)])
    softmax = Activations(softmax=True)
    test_evaluator = Task3SegmentationEvaluator(test_loader_full, device=device)
    val_evaluator = Task3SegmentationEvaluator(val_loader_full, device=device)
        
    best_score = -1.
    best_score_epoch = -1
        
    epoch_steps = len(train_loader)
    print_step = 50
    # %%
    for epoch in range(opt.epochs):
        
        print("-" * 10)
        print(f"Running epoch {epoch + 1}/{opt.epochs}")
        print("##### TRAIN #####")
        model.train()
        epoch_loss = 0
        step = 0
        

        for batch_data in train_loader:
             step += 1
             inputs, r1, r2 = batch_data["img"].to(device), batch_data["r1"].to(device), batch_data["r2"].to(device)
             optimizer.zero_grad()
             outputs = model(inputs)      
             loss_r1 = loss_function(outputs, r1)
             loss_r2 = loss_function(outputs, r2)
             loss = 0.5 * (loss_r1 + loss_r2)
             loss.backward()
             optimizer.step()
             epoch_loss += loss.detach().item()
             if step % print_step == 0:
                 current_loss = epoch_loss / step
                 print(f"{step}/{epoch_steps}, train_loss: {current_loss:.4f}")
             writer.add_scalar("batch_train_loss", loss.detach().item(), epoch*epoch_steps + step)

        epoch_loss /= step
        writer.add_scalar("train_loss", loss.item(), epoch)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        scheduler.step()
    
        if (epoch + 1) % opt.val_interval == 0:
            model.eval()
            with torch.no_grad():
                print("\n##### EVALUATION ON FULL BRAINS#####")    
                val_metrics = test_evaluator.evaluate(model, roi_size=opt.patch_size, log=False)
                
                if val_metrics[0] > best_score:
                    best_score = val_metrics[0]
                    best_scoreo_epoch = epoch + 1
                    torch.save(model, os.path.join(log_path, "best_model.pth"))
                    print("saved new best metric model")
                
                print(f"Best ROI model at epoch {best_score_epoch}")
                writer.add_scalar("val_dice_roi", val_metrics[0], epoch + 1)
                writer.add_scalar("val_f1_roi", val_metrics[1], epoch + 1)
                writer.add_scalar("val_recall_roi", val_metrics[2], epoch + 1)
                writer.add_scalar("val_precision_roi", val_metrics[3], epoch + 1)
                                
    print(f"train completed, best_metric: {best_score:.4f} at epoch: {best_score_epoch}")
    
    writer.close()

    # %% BEST MODEL FULL BRAIN ROI TEST EVALUATION 
    best_model = torch.load(os.path.join(log_path, "best_model.pth"), map_location=device)
    print("\n##### EVALUATING BEST MODEL FULL BRAIN ON VALIDATION SET")
    test_metrics = test_evaluator.evaluate(best_model, roi_size=opt.patch_size, log=False)
    print("\n##### EVALUATING BEST MODEL FULL BRAIN ROI ON TEST SET")
    test_metrics = test_evaluator.evaluate(best_model, roi_size=opt.patch_size, log=False)
