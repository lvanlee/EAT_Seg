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



