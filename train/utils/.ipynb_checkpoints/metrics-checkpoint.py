import torch.nn as nn
import torch

from monai.metrics import DiceMetric

from scipy.ndimage import label


class Task3_Metrics():
    def __init__(self, dice_metric):
        self.dice_metric = dice_metric
        self.absolute_count_diff = AbsoluteCountDiff()
        self.f1_score = F1_Score()
        
    
    def __call__(self, predictions, labels, seg1, seg2):
        dice_label, _ = self.dice_metric(y_pred=predictions,y=labels)
        dice_seg1, _ = self.dice_metric(y_pred=predictions, y=seg1)
        dice_seg2, _ = self.dice_metric(y_pred=predictions, y=seg2)
        dice_seg_avg = (dice_seg1 + dice_seg2 ) / 2
        
        
        acd_label = self.absolute_count_diff(y_pred=predictions,y_true=labels)
        acd_seg1 = self.absolute_count_diff(y_pred=predictions, y_true=seg1)
        acd_seg2 = self.absolute_count_diff(y_pred=predictions, y_true=seg2)
        acd_seg_avg = (acd_seg1 + acd_seg2) / 2
        
        
        f1_label = self.f1_score(y_pred=predictions,y_true=labels)
        f1_seg1 = self.f1_score(y_pred=predictions, y_true=seg1)
        f1_seg2 = self.f1_score(y_pred=predictions, y_true=seg2)
        f1_seg_avg = (f1_seg1 + f1_seg2) / 2
        
        return {
            'dice_l': dice_label, 
            'dice_s': dice_seg_avg,
            'acd_l': acd_label,
            'acd_s': acd_seg_avg,
            'f1_l': f1_label, 
            'f1_s': f1_seg_avg
        }
        


class Task3_DiceMetric():
    def __init__(self, include_background=True):
        self.dice_metric = DiceMetric(include_background=include_background, reduction="mean")

    def __call__(self, predictions, labels, seg1, seg2):
        dice_label, _ = self.dice_metric(y_pred=predictions,y=labels)
        dice_seg1, _ = self.dice_metric(y_pred=predictions, y=seg1)
        dice_seg2, _ = self.dice_metric(y_pred=predictions, y=seg2)
        dice_seg_avg = (dice_seg1 + dice_seg2 ) / 2
        
        return dice_label, dice_seg_avg
    
class F1_Score(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        
        tp = (y_true * y_pred).sum().to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return f1.mean()

class AbsoluteCountDiff(nn.Module):
    '''
    Compute count difference of object
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
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