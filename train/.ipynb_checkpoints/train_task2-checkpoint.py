import logging
import os
import argparse
import shutil
import sys
import tempfile
import json
from glob import glob
from datetime import datetime
#import nibabel as nib
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

from utils import get_optimizer, get_scheduler
from utils import Task3_DiceMetric

parser = argparse.ArgumentParser(description='Task 3 training')
#/cluster/project0/MAS/cosimo/valdo/Task3
#C:/Datasets/valdo/Task3
# data path
parser.add_argument('--directory', default="C:/Datasets/valdo_normalized/Task2", type=str, help='dataset root')
parser.add_argument('--train', default="my_index_train.json", type=str, help='index name')
parser.add_argument('--val', default="my_index_test.json", type=str, help='index name')
# run settings
parser.add_argument('--batch_size', default=2, type=int, help='batch  size')
parser.add_argument('--sample_size', default=4, type=int, help='number of sample per item')
parser.add_argument('--val_interval', default=1, type=int, help='validate every n epochs')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs')
parser.add_argument('--epoch_length', default=8, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--optimizer', default="Novograd", type=str, choices=['Adam', 'SGD', 'Novograd'],)
parser.add_argument('--optim_args', default='{}', type=str)
parser.add_argument('--scheduler', default="None", type=str, choices=['None', 'LambdaLR', 'MultiplicativeLR'])
parser.add_argument('--scheduler_args', default="{}", type=str)
# log info
parser.add_argument('--log_folder', default="runs", type=str)
parser.add_argument('--run_model', default="test", type=str)
parser.add_argument('--run_info', default="", type=str)
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
    run_name = f"{dt.strftime('%Y-%m-%d_%H-%M')}_{opt.run_model}_lr_{opt.lr}_{opt.optimizer}_{opt.scheduler}"
    log_path = os.path.join(os.curdir, opt.log_folder, run_name)
    
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    writer = SummaryWriter(log_path)      

    with open(os.path.join(log_path, "info.txt"), "w") as info_file:
        info_file.write(str(opt))
    
    
    # %%
    # Set fixed random number seed
            
    train_index_path = os.path.join(opt.directory, opt.train)
    with open(index_path) as f:
        train_files = json.load(f)
    val_index_path = os.path.join(opt.directory, opt.val)
    with open(index_path) as f:
        val_files = json.load(f)

    # define transforms for image and segmentation
    train_transforms = Compose(
            [
                LoadImaged(keys=["t1", "seg"], reader="ITKReader"),
                AddChanneld(keys=["t1","seg"]),
                CropForegroundd(keys=["t1", "seg"], source_key="t1"),
                #NormalizeIntensityd(keys="t1", nonzero=True, channel_wise=True),
                ScaleIntensityd(keys="t1"),
                Resized(keys=["t1", "seg"], spatial_size=[256, 256, 128]),
                RandCropByPosNegLabeld(
                    keys=["t1", "seg"],
                    label_key="seg",
                    spatial_size=[128, 128, 64],
                    pos=1,
                    neg=1,
                    num_samples=opt.sample_size,
                ),
                RandFlipd(keys=["t1", "seg"], prob=0.5, spatial_axis=0),
                RandAffined(
                    keys=['t1', 'seg'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(128, 128, 64),
                    rotate_range=(0, 0, np.pi/15),
                    scale_range=(0.1, 0.1, 0.1)
                ),
                ToTensord(keys=["t1", "seg"]),
            ]
        )
    val_transforms = Compose(
            [
                LoadImaged(keys=["t1", "seg"], reader="ITKReader"),
                AddChanneld(keys=["t1", "seg"]),
                CropForegroundd(keys=["t1", "seg"], source_key="t1"),
                #NormalizeIntensityd(keys="t1", nonzero=True, channel_wise=True),
                ScaleIntensityd(keys="t1"),
                Resized(keys=["t1", "seg"], spatial_size=[256, 256, 128]),
                ToTensord(keys=["t1", "seg"]),
            ]
    )
    
    
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    
    # create a validation data loader
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    
    # %%
    # create UNet, DiceLoss and Adam optimizer
    model = UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(30, 60, 120, 240, 480),
        strides=(2, 2, 2, 2),
        num_res_units=2,
	dropout=0.5,
    ).to(device)
    
    loss_function = DiceLoss(sigmoid=True, squared_pred=True)
    optimizer = get_optimizer(opt.optimizer, model.parameters(), opt.lr, kwargs=json.loads(opt.optim_args))
    scheduler = get_scheduler(opt.scheduler, optimizer, kwargs=json.loads(opt.scheduler_args))
    
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    post_val = AsDiscrete(threshold_values=True)
    
    best_metric = -1
    best_metric_epoch = -1
    epoch_steps = (opt.epoch_length-1) // (opt.batch_size * opt.sample_size) +1 
    # %%
    for epoch in range(opt.epochs):
        print("-" * 10)
        print(f"Running epoch {epoch + 1}/{opt.epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        while step < epoch_steps:
            for batch_data in train_loader:
                step += 1
                inputs, labels = batch_data["t1"].to(device), batch_data["seg"].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)      
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
                print(f"{step}/{epoch_steps}, train_loss: {loss.item():.4f}")
                if step >= epoch_steps:
                    break
        
        epoch_loss /= step
        writer.add_scalar("train_loss", loss.item(), epoch)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        scheduler.step()
    
        if (epoch + 1) % opt.val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_label_sum = 0.0
                metric_seg_sum = 0.0
                metric_count = 0
                task3_metrics = Task3_DiceMetric()
                for val_data in val_loader:
                    val_images, val_labels = val_data["t1"].to(device), post_val(val_data["seg"].to(device))
                    roi_size = (128, 128, 64)
                    sw_batch_size = 2
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = post_trans(val_outputs)
                                        
                    metric_label = dice_metric(val_outputs, val_labels)
                    
                    metric_count += len(metric_label)
                    metric_label_sum += metric_label.sum().item()

                metric_label = metric_label_sum / metric_count
                
                if metric_label > best_metric:
                    best_metric = metric_label
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(log_path, "best_model.pth"))
                    torch.save({"epoch": epoch+1, "mean_dice_label": best_metric_epoch, "mean_dice_seg": metric_seg}, 
                               os.path.join(log_path, "best_score.pth"))
                    print("saved new best metric model")
                print("current epoch: {} current label mean dice: {:.4f} --- best mean dice: {:.4f} at epoch {}"
                      .format(epoch + 1, metric_label, best_metric, best_metric_epoch))
                writer.add_scalar("val_mean_dice_label", metric_label, epoch + 1)
               
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()    
