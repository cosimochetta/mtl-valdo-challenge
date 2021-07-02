import numpy as np

import torch.nn as nn
import torch
from torch.nn import BCELoss, MaxPool2d
from sklearn.metrics import roc_curve
import pandas as pd
from scipy.ndimage import label
from sklearn.metrics import precision_score, recall_score, f1_score

from monai.metrics import DiceMetric
from monai.metrics import compute_roc_auc
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
)
from .evaluations import sliding_window_3dimg_2dmodel

class Task3SegmentationEvaluator:
    """
    Evaluate detector fully convolutional network
    A fully conv layer produce an output 
    
    """
    
    def __init__(self, 
                 data_loader,
                 device='cpu'):
        
        assert data_loader.batch_size == 1
        self.data_loader = data_loader
        self.metric = DiceMetric(include_background=False, reduction="mean")
        self.activation = Compose([Activations(softmax=True), AsDiscrete(threshold_values=True)])
        self.device = device
          
    def compute_metrics(self, rater, prediction, mask):
        """
        Compute Dice metric, F1 score, Recall and Precision
        of a prediction
        
        Parameters
        ----------
        rater : Array shape B2WDH
            Ground truth, first class contains 
        prediction : Array shape B2WDH
        
        
        Returns
        -------
        metric : TYPE
            DESCRIPTION.
        f1 : TYPE
            DESCRIPTION.
        recall : TYPE
            DESCRIPTION.
        precision : TYPE
            DESCRIPTION.

        """
        
        # Dice Metric
        metric, _ = self.metric(prediction, rater)
        
        # F1, Recall, Precision
        prediction = prediction[:,1,mask==1].view(-1,1)
        rater = rater[:,1,mask==1].view(-1,1)
        if rater.max() == 0:
            f1, recall, precision = [None]*3
        else:
            f1 = f1_score(rater, prediction)
            recall = recall_score(rater, prediction)
            precision = precision_score(rater, prediction)
        return metric, f1, recall, precision
    
    def print_metrics(self, idx, metrics, rater_name):
        print("IDX {} {} - dice metric: {} - f1: {} - recall: {} - precision: {}"
              .format(idx, rater_name, metrics[0], metrics[1], metrics[2], metrics[3]))
    
    def evaluate(self, net, roi_size=48, log=True):
        """

        Parameters
        ----------
        net : callable
            Fully convolutional network, must have get_original_space_params() function.

        Returns
        -------
        metrics: array
            array containing the average of the following metrics:
            [self.metric, f1, recall, precision, auc]
        """
        
        scores = []        
        with torch.no_grad():
            for i, data in enumerate(self.data_loader):
                images = data['img'].to(self.device)
                r1 = data['r1']
                r2 = data['r2']
                mask = data['mask'][0,0]
                
                prediction = self.activation(
                    sliding_window_3dimg_2dmodel(
                        images, 
                        r1, 
                        net, roi_size=(roi_size, roi_size),
                        overlap=0, 
                        mode='constant',
                        )
                    ).cpu()

                # Predict                
                metrics_r1 = self.compute_metrics(r1, prediction, mask)
                metrics_r2 = self.compute_metrics(r2, prediction, mask)
              #  print(metrics_r1)
                if log:
                    self.print_metrics(i, metrics_r1, "Rater1")
                    self.print_metrics(i, metrics_r2, "Rater2")
                scores.append(metrics_r1)
                scores.append(metrics_r2)
                
        avg_res = np.nanmean(np.array(scores, dtype=np.float), axis=0)

        self.print_metrics("AVG", avg_res, "both rater")
        return avg_res


class AbsoluteCountDiff(nn.Module):
    '''
    Compute count difference of object between label and prediction
    
    Returns
    -------
    abs(Label_obj_count - prediction_obj_count), prediction_obj_count, label_obj_count 
    '''
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        _, prediction_count = label(y_pred)
        _, label_count = label(y_true)
        prediction_count = torch.tensor(prediction_count).unsqueeze(-1)
        label_count = torch.tensor(label_count).unsqueeze(-1)
        diff = torch.abs(torch.tensor(label_count - prediction_count))
        return diff, prediction_count, label_count


def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------     
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])[0] 

class Task3DetectorEvaluator:
    """
    Evaluate detector fully convolutional network
    A fully conv layer produce an output 
    
    """
    
    def __init__(self, 
                 data_loader, 
                 metric=BCELoss(), 
                 activation=Activations(sigmoid=True), 
                 device='cpu'):
        
        assert data_loader.batch_size == 1
        self.data_loader = data_loader
        self.metric = metric
        self.activation = activation
        self.device = device
    
    
    def maxpool_reduction(self, data, offset, scale):
        maxpool = MaxPool2d(kernel_size=scale, stride=scale)
        return maxpool(data[:,:,offset:-offset, offset:-offset])
      
    def compute_metrics(self, rater, prediction, cutoff):
        metric = self.metric(prediction, rater)
        if rater.max() == 0:
            f1, recall, precision, auc = [None]*4
        else:
            if cutoff == None:
                cutoff = find_optimal_cutoff(rater, prediction)
            print(f"CUTOFF: {cutoff}")
            f1 = f1_score(rater, prediction >= cutoff)
            recall = recall_score(rater, prediction >= cutoff)
            precision = precision_score(rater, prediction >= cutoff)
            auc = compute_roc_auc(prediction, rater)
        return metric, f1, recall, precision, auc, cutoff
    
    def print_metrics(self, idx, metrics, rater_name):
        print("IDX {} {} - metric: {:.4f} - f1: {} - recall: {} - precision: {} - auc: {}"
              .format(idx, rater_name, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))
    
    def evaluate(self, net, cutoff=None):
        """

        Parameters
        ----------
        net : callable
            Fully convolutional network, must have get_original_space_params() function.

        Returns
        -------
        metrics: array
            array containing the average of the following metrics:
            [self.metric, f1, recall, precision, auc]

        """
        
        offset, scale = net.get_original_space_params()
        logits_all = None
        r_all = None
        
        with torch.no_grad():
            for i, data in enumerate(self.data_loader):
                images = data['img'][0].permute(3, 0, 1, 2).to(self.device)
                r1 = data['r1'][0,1:2].permute(3, 0, 1, 2)
                r2 = data['r2'][0,1:2].permute(3, 0, 1, 2)
                mask = data['mask'][0].permute(3, 0, 1, 2)
                
                # Apply maxpool to match the model output
                r1 = self.maxpool_reduction(r1, offset, scale).view(-1,1)
                r2 = self.maxpool_reduction(r2, offset, scale).view(-1,1)
                mask = self.maxpool_reduction(mask, offset, scale).view(-1,1)
                
                # Predict
                logits = self.activation(net(images)).view(-1,1).cpu()
                logits = logits[mask==1].view(-1,1)
                #print(torch.sum(logits > 0.5))
                r1 = r1[mask==1].view(-1,1)
                r2 = r2[mask==1].view(-1,1)
                metrics_r1 = self.compute_metrics(r1, logits, cutoff)
                self.print_metrics(i, metrics_r1, "Rater1")
                metrics_r2 = self.compute_metrics(r2, logits, cutoff)
                self.print_metrics(i, metrics_r2, "Rater2")
           #     print(logits.shape)
                logits_all = torch.vstack([logits, logits]) if logits_all == None else torch.vstack([logits_all, logits, logits])
            #    print(logits_all.shape)
                r_all = torch.vstack([r1, r2]) if r_all == None else torch.vstack([r_all, r1, r2])                    

             

        avg_res = self.compute_metrics(r_all, logits_all, cutoff)
        self.print_metrics("AVG", avg_res, "both rater")
        return avg_res, avg_res[-1]