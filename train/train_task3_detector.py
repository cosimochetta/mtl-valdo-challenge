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
from monai.data import list_data_collate

from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCELoss

from monai.metrics import compute_roc_auc

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    )

from detection_net import get_detector
from utils import get_optimizer, get_scheduler
from utils import ValdoDatasetTask3Patches, ValdoDatasetTask3Evaluation
from utils import Task3DetectorEvaluator

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
parser.add_argument('--patch_size',  default=32,  type=int)
parser.add_argument('--log_folder', default="runs", type=str)
parser.add_argument('--run_model', default="multi_scale", type=str, choices=['resnet', 'multi_scale', 'dilated_network'])
parser.add_argument('--run_info', default="", type=str)
opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
if __name__ == '__main__':
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Set seed
    torch.manual_seed(42)
        
    # Create run log path and info
    dt = datetime.today()
    run_name = f"{dt.strftime('%Y-%m-%d_%H-%M')}_Detector_{opt.run_model}_lr_{opt.lr}_s_{opt.patch_size}_{opt.optimizer}_{opt.scheduler}"
    log_path = os.path.join(os.curdir, opt.log_folder, run_name)
    
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    writer = SummaryWriter(log_path)

    with open(os.path.join(log_path, "info.txt"), "w") as info_file:
        info_file.write(str(opt))
    
    # %% CREATE DATASETS
                
    train_path = os.path.join(opt.directory, "Train", "64")
    val_full_path = os.path.join(opt.directory, "Val_full")
    test_full_path = os.path.join(opt.directory, "Test_full")
    
    # %%
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

    
    #%%
    # create a validation data loader
    val_ds_full = ValdoDatasetTask3Evaluation(root_dir=val_full_path)
    val_loader_full = DataLoader(
        val_ds_full,
        batch_size=1,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    test_ds_full = ValdoDatasetTask3Evaluation(root_dir=test_full_path)
    test_loader_full = DataLoader(
        test_ds_full,
        batch_size=1,
        num_workers=0,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    
    
    
    # %%
    # Create model and losses
    model = get_detector(opt.run_model).to(device)
        
    loss_function = BCELoss()
    optimizer = get_optimizer(opt.optimizer, model.parameters(), opt.lr, kwargs=json.loads(opt.optim_args))
    scheduler = get_scheduler(opt.scheduler, optimizer, kwargs=json.loads(opt.scheduler_args))
    
    # Validation metrics
    metric = BCELoss()
    post_inference = Compose([Activations(softmax=True), AsDiscrete(threshold_values=True)])
    softmax = Activations(softmax=True)
    sigmoid = Activations(sigmoid=True)
    
    val_evaluator = Task3DetectorEvaluator(val_loader_full, device=device)
    test_evaluator = Task3DetectorEvaluator(test_loader_full, device=device)
    
    best_f1_score = -1
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
        
        logits_all = None
        labels_all = None

        for batch_data in train_loader:
             step += 1
             inputs, label = batch_data["img"].to(device), batch_data["label"].float().to(device)
             optimizer.zero_grad()
             logits = sigmoid(model(inputs)).view(-1,1)     
             loss = loss_function(logits, label)
             loss.backward()
             optimizer.step()
             epoch_loss += loss.detach().item()
             
             logits_all = logits.detach() if logits_all == None else torch.vstack([logits_all, logits.detach()])
             labels_all = label if labels_all == None else torch.vstack([labels_all, label])
             
             
             if step % print_step == 0:
                 current_loss = epoch_loss / step
                 print(f"{step}/{epoch_steps}, train_loss: {current_loss:.4f}")
             writer.add_scalar("batch_train_loss", loss.detach().item(), epoch*epoch_steps + step)
             
        auc_score = compute_roc_auc(logits_all, labels_all)
     
        epoch_loss /= step
        writer.add_scalar("train patch loss", loss.item(), epoch+1)
        writer.add_scalar("train patch auc", auc_score, epoch+1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} - auc {auc_score:.4f}")
        scheduler.step()
    
        if (epoch + 1) % opt.val_interval == 0:
            model.eval()
            print("##### EVALUATION #####")
            with torch.no_grad():
                
                metrics, _ = val_evaluator.evaluate(model)
                writer.add_scalar("val cross entropy", metrics[0], epoch + 1)
                writer.add_scalar("val f1", metrics[1], epoch + 1)
                writer.add_scalar("val recall ", metrics[2], epoch + 1)
                writer.add_scalar("val precision ", metrics[3], epoch + 1)
                writer.add_scalar("val auc ", metrics[4], epoch + 1)

                if metrics[1] > best_f1_score:
                    best_f1_score = metrics[1]
                    best_score_epoch = epoch + 1
                    torch.save(model, os.path.join(log_path, "best_model.pth"))
                    torch.save({"epoch": epoch+1, "score": best_f1_score}, 
                               os.path.join(log_path, "best_score.pth"))
                    print("saved new best metric model")
                
    torch.save(model, os.path.join(log_path, "final_model.pth"))
    print(f"train completed, best_metric: {best_f1_score:.4f} at epoch: {best_score_epoch}")
    writer.close()
    
    # %% BEST MODEL TEST EVALUATION
    best_model = torch.load(os.path.join(log_path, "best_model.pth"), map_location=device)
    print("##### EVALUATING BEST MODEL ON VAL SET")
    val_metrics, best_cutoff = val_evaluator.evaluate(best_model)
    print(f"BEST CUTOFF: {best_cutoff}")
    print("##### EVALUATING BEST MODEL ON TEST SET")

    test_metrics, _ = test_evaluator.evaluate(best_model, best_cutoff)
            