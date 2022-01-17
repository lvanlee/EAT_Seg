# ! usr/bin/env python3
# -*- coding:utf-8 -*-
# author:lvanlee   time:2021/3/25


import torch
import torch.nn as nn

class PNDiceLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4):
        super(PNDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, logits, targets):
        smooth = 0.0001

        negtive = ((1 - logits)**2) * (1 - targets)  
        
        PDice = (positive.sum(1) + smooth) / ((logits**2).sum(1) + targets.sum(1) + smooth)
        NDice = (negtive.sum(1) + smooth) / (((1 - logits)**2).sum(1) + (1 - targets).sum(1) + smooth)

        PNDiceLoss = 1. - 2*(self.alpha*PDice+self.beta*NDice)
        return PNDiceLoss


#################SoftDiceLoss##################################

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        
        smooth = 0.0001
        intersection = logits * targets

        score = 2. * (intersection.sum(1) + smooth) / (logits.sum(1) + targets.sum(1) + smooth)
        
        score = 1 - score.sum()
        return score



##########################  BCELoss2d   ###########################

class BCELoss2d(nn.Module):
    def __init__(self, weight=None):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, logits, targets):
        return self.bce_loss(logits, targets)




##################################################Combo Loss
ALPHA = 0.5 
BETA = 0.5 
e = 1e-4

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.0001, alpha=ALPHA, beta=BETA):


       
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        inputs = torch.clamp(inputs, e, 1.0 - e)
        out = - (BETA * ((targets * torch.log(inputs)) + ((1 - BETA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (ALPHA * weighted_ce) - ((1 - ALPHA) * dice)

        return combo





######################################IOULoss
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.0001):

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU

#Focal Loss
Focal_ALPHA = 0.8
Focal_GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=Focal_ALPHA, gamma=Focal_GAMMA, smooth=0.0001):

        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        return focal_loss

##############################################Tversky Loss
Tversky_ALPHA = 0.5
Tversky_BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.0001, alpha=Tversky_ALPHA, beta=Tversky_BETA):

        
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

        return 1 - Tversky

######################################Focal Tversky Loss
FT_ALPHA = 0.5
FT_BETA = 0.5
FT_GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.0001, alpha=FT_ALPHA, beta=FT_BETA, gamma=FT_GAMMA):

    
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        FocalTversky = (1 - Tversky)**gamma

        return FocalTversky



#######Dice_BCELoss######################################################


class Dice_BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.5):
        super(Dice_BCELoss, self).__init__()
        self.alpha = alpha
        self.beta = 1 - alpha

    def forward(self, logits, targets):
        smooth = 0.0001
        
        intersection = logits * targets
        Dice_score = 2. * (intersection.sum(1) + smooth) / (logits.sum(1) + targets.sum(1) + smooth)
        
        Dice_score = 1 - Dice_score.sum()

        bce_loss = nn.BCELoss()  
        BCE_score = bce_loss(logits, targets)

        return self.alpha * Dice_score + self.beta * BCE_score



##############focalloss2d#############################################
import torch.autograd as autograd
from torch.nn import functional as F



class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        
        assert len(logits.shape) == len(targets.shape)
        assert logits.size(0) == targets.size(0)
        assert logits.size(1) == targets.size(1)

        logpt = - F.binary_cross_entropy_with_logits(logits, targets, reduction=self.reduction)

        pt = torch.exp(logpt)

        
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss





###############DDCLoss#############################################
class DDCLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=3, beta=2):
        super(DDCLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        smooth = 0.0001
        bceloss = -(targets * torch.log(logits + smooth) + (1 - targets) * torch.log(1 - logits + smooth))
        certern = (1 - abs(logits - 0.5)) ** self.alpha
        distant = (2 * abs(targets - logits)) ** self.beta
        DDCloss = bceloss * certern * distant
        DDCloss = DDCloss.sum(1) / 65536

        return DDCloss



