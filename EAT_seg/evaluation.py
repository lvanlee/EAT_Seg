import torch
import os

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR, GT, threshold=0.5):

    GT[GT > threshold] = 1
    GT[GT < threshold] = 0
    SR[SR > threshold] = 1
    SR[SR < threshold] = 0
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR, GT, threshold=0.5):
  
    GT[GT > threshold] = 1
    GT[GT < threshold] = 0
    SR[SR > threshold] = 1
    SR[SR < threshold] = 0

    TP = ((SR == 1) & (GT == 1))
    FN = ((SR == 0)&(GT == 1)) 

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR, GT, threshold=0.5):
    GT[GT > threshold] = 1
    GT[GT < threshold] = 0
    SR[SR > threshold] = 1
    SR[SR < threshold] = 0

    TN = ((SR == 0)&(GT == 0)) 
    FP = ((SR == 1)&(GT == 0)) 

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    GT[GT > threshold] = 1
    GT[GT < threshold] = 0
    SR[SR > threshold] = 1
    SR[SR < threshold] = 0

    TP = ((SR == 1) & (GT == 1))
    FP = ((SR == 1)&(GT == 0)) 

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC


def get_F1(SR,GT,threshold=0.5):

    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR, GT, threshold=0.5):

    SR[SR > threshold] = 1
    SR[SR < threshold] = 0
    GT[GT > threshold] = 1
    GT[GT < threshold] = 0
    Inter = torch.sum((SR+GT) == 2)
    Union = torch.sum((SR+GT) >= 1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR, GT, threshold=0.5):

    SR[SR > threshold] = 1
    SR[SR < threshold] = 0
    GT[GT > threshold] = 1
    GT[GT < threshold] = 0
    Inter = torch.sum((SR+GT) == 2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC


def tensor2img(x):
    img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
    img = img * 255
    return img

