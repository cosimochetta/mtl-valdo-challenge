# -*- coding: utf-8 -*-

import logging
import os
import argparse
import shutil
import sys
import json
from datetime import datetime


import torch
from torch.utils.data import DataLoader
from monai.data import list_data_collate, Dataset

from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss

from monai.config import print_config
from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.networks.nets import UNet
from monai.metrics import DiceMetric

from monai.transforms import Activations

from utils import get_optimizer, get_scheduler
from utils import Task3_DiceMetric
from utils import ValdoDatasetTask3Patches
# %% 
parser = argparse.ArgumentParser(description='Task 3 training')
#/cluster/project0/MAS/cosimo/valdo/Task3
#C:/Datasets/valdo/Task3
# data path
parser.add_argument('--directory', default="D:/Datasets/valdo_patches/Task3", type=str, help='dataset root')
# run settings
parser.add_argument('--batch_size', default=64, type=int, help='batch  size')
parser.add_argument('--val_interval', default=1, type=int, help='validate every n epochs')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--optimizer', default="Adam", type=str, choices=['Adam', 'SGD', 'Novograd'],)
parser.add_argument('--optim_args', default='{}', type=str)
parser.add_argument('--scheduler', default="None", type=str, choices=['None', 'LambdaLR', 'MultiplicativeLR'])
parser.add_argument('--scheduler_args', default="{}", type=str)
parser.add_argument('--patch_size',  default=32,  type=int)
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
    run_name = f"{dt.strftime('%Y-%m-%d_%H-%M')}_{opt.run_model}_lr_{opt.lr}_s_{opt.patch_size}_{opt.optimizer}_{opt.scheduler}"
    log_path = os.path.join(os.curdir, opt.log_folder, run_name)
    
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    writer = SummaryWriter(log_path)

    with open(os.path.join(log_path, "info.txt"), "w") as info_file:
        info_file.write(str(opt))
    
    
    # %% Load data
                
    train_path = os.path.join(opt.directory, "Train")
    val_path = os.path.join(opt.directory, "Val")

    train_ds = ValdoDatasetTask3Patches(root_dir=train_path, 
                                 patch_size=opt.patch_size, 
                                 split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    
    # create a validation data loader
    val_ds = ValdoDatasetTask3Patches(root_dir=val_path, 
                                 patch_size=opt.patch_size, 
                                split="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=opt.batch_size,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    
    # %%
    # create UNet, DiceLoss and Adam optimizer
    model = UNet(
        dimensions=2,
        in_channels=1,
        out_channels=2,
        channels=(64, 128, 256, 512),
        #channels=(30, 60, 120, 240, 480),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm='BATCH',
        #dropout=0.2,
    ).to(device)
    
    loss_function = GeneralizedDiceLoss(softmax=True)
    #ce_weights = torch.tensor([0.1, 0.9])
    #loss_function = CrossEntropyLoss(weight=ce_weights)
    optimizer = get_optimizer(opt.optimizer, model.parameters(), opt.lr, kwargs=json.loads(opt.optim_args))
    scheduler = get_scheduler(opt.scheduler, optimizer, kwargs=json.loads(opt.scheduler_args))
    
    # Validation metrics
    dice_metric = DiceLoss(softmax=False, include_background=False)
    #dice_metric = DiceMetric(include_background=False, reduction="mean")
    softmax = Activations(softmax=True)
    
    best_score = -1
    best_score_epoch = -1
    epoch_steps = len(train_loader)
    print_step = 50
    # %%
    for epoch in range(opt.epochs):
        print("-" * 10)
        print(f"Running epoch {epoch + 1}/{opt.epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["t1"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)      
            loss = loss_function(outputs, labels)
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
                score_sum = 0.0
                score_count = 0
                task3_metrics = Task3_DiceMetric()
                for val_data in val_loader:
                    val_images, val_labels = val_data["t1"].to(device), val_data["label"].to(device)
                    outputs = model(val_images)   
                    loss = dice_metric(softmax(outputs), val_labels)
                    #metric_label, metric_seg = task3_metrics(val_outputs, val_labels, val_seg1, val_seg2)
                    score_count += 1
                    score_sum += 1 - loss.item()

                current_score = score_sum / score_count
                print("epoch: {} current score: {:.4f};  best score: {:.4f} at epoch {}"
                      .format(epoch + 1, current_score, best_score, best_score_epoch))

                if current_score > best_score:
                    best_score = current_score
                    best_score_epoch = epoch + 1
                    torch.save(model, os.path.join(log_path, "best_model.pth"))
                    torch.save(model.state_dict(), os.path.join(log_path, "best_model_state.pth"))
                    torch.save({"epoch": epoch+1, "score": best_score_epoch}, 
                               os.path.join(log_path, "best_score.pth"))
                    print("saved new best metric model")
                writer.add_scalar("val_score", current_score, epoch + 1)

    print(f"train completed, best_metric: {best_score:.4f} at epoch: {best_score_epoch}")
    
    writer.close()    