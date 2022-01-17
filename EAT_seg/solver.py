# ! usr/bin/env python3
# -*- coding:utf-8 -*-
# author:lvanlee   time:2021/4/17
import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import *
import csv
from torchvision import transforms
from loss import *
import datetime
import time



class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

       
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch

        self.criterion = SoftDiceLoss()
        self.augmentation_prob = config.augmentation_prob

        
        self.lr = config.lr  
        self.beta1 = config.beta1  
        self.beta2 = config.beta2

       
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

       
        self.log_step = config.log_step
        self.val_step = config.val_step

       
        self.model_path = config.model_path
        self.SR_result_path = config.SR_result_path
        self.EAT_GT_path = config.EAT_GT_path
        self.EAT_images_path = config.EAT_images_path

        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            self.unet = U_Net(img_ch=1, output_ch=1)  
        elif self.model_type == 'IRCU_Net':
            self.unet = IRCU_Net(img_ch=1, output_ch=1)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=1, output_ch=1)
        elif self.model_type == 'IRU_Net':
            self.unet = IRU_Net(img_ch=1, output_ch=1)
        elif self.model_type == 'segnet':
            self.unet = segnet(img_ch=1, output_ch=1)
       
        elif self.model_type == 'PSPNet':
            self.unet = PSPNet(in_class=1, n_classes=1, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=1024,
                               backend='resnet34', pretrained=True)
        elif self.model_type == 'FCN32s':
            self.unet = FCN32s(in_class=1, n_class=1)
       
        elif self.model_type == 'DilaLab6':
            self.unet = DilaLab6(ch_in=1, ch_out=1)
    

        elif self.model_type == 'NestedUNet':
            self.unet = NestedUNet(in_ch=1, out_ch=1)
        
       
        self.optimizer = torch.optim.RMSprop(list(self.unet.parameters()),
                                             self.lr, alpha=0.9)

        self.unet.to(self.device)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))  

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data


    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

  
    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    #############################################################

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        unet_path = os.path.join(self.model_path, '%s-epoch%d-lr%.4f-decay%d-augprob%.4f.pkl' % (
        self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

       
        if os.path.isfile(unet_path):
            
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
        else:
           
            lr = self.lr
            best_unet_score = 0.
            all_DC_train = []
            all_DC_valid = []

            train_loss_list = []
            valid_loss_list = []

            train_acc_list = []
            valid_acc_list = []

            train_PC_list = []
            valid_PC_list = []

            train_SP_list = []
            valid_SP_list = []

            for epoch in range(self.num_epochs):

                self.unet.train(True)
                train_loss = 0

                acc = 0.  
                SE = 0.  
                SP = 0.  
                PC = 0.  
                F1 = 0.  
                JS = 0.  
                DC = 0.  
                length = 0

                for i, (images, GT, filename) in enumerate(self.train_loader):
                   
                    images = images.to(self.device)
                    GT = GT.to(self.device)

                    SR = self.unet(images)

                    SR = torch.sigmoid(SR)  
                    SR_flat = SR.view(SR.size(0), -1)  

                    GT_flat = GT.view(GT.size(0), -1)  

                    loss = self.criterion(SR_flat, GT_flat) 

                   
                    self.reset_grad()
                    loss = loss.requires_grad_()
                    loss.backward(torch.ones_like(loss))
                  
                    self.optimizer.step()

                    train_loss += loss.item()
                    acc += get_accuracy(SR, GT)
                    SE += get_sensitivity(SR, GT)
                    SP += get_specificity(SR, GT)
                    PC += get_precision(SR, GT)
                    F1 += get_F1(SR, GT)
                    JS += get_JS(SR, GT)
                    DC += get_DC(SR, GT)
                    length += images.size(0)
                  
                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length

                train_loss = train_loss / length

                all_DC_train.append(DC)
                train_loss_list.append(train_loss)
                train_acc_list.append(acc)
                train_PC_list.append(PC)
                train_SP_list.append(SP)

                print('##################mean_train_DC:%.4f####################' % np.mean(all_DC_train))

               
                print(
                    'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                        epoch + 1, self.num_epochs, train_loss, acc, SE, SP, PC, F1, JS, DC))

               
                if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                    lr -= (self.lr / float(self.num_epochs_decay))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Decay learning rate to lr: {}.'.format(lr))

                # ===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()
                valid_loss = 0
                acc = 0.  
                SE = 0.  
                SP = 0.  
                PC = 0. 
                F1 = 0.  
                JS = 0.  
                DC = 0. 
                length = 0

                for i, (images, GT, filename) in enumerate(self.valid_loader):
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = torch.sigmoid(self.unet(images)) 
                    SR_flat = SR.view(SR.size(0), -1)  
                    GT_flat = GT.view(GT.size(0), -1)  
                    loss = self.criterion(SR_flat, GT_flat)  
                    valid_loss += loss.item()

                    acc += get_accuracy(SR, GT)
                    SE += get_sensitivity(SR, GT)
                    SP += get_specificity(SR, GT)
                    PC += get_precision(SR, GT)
                    F1 += get_F1(SR, GT)
                    JS += get_JS(SR, GT)
                    DC += get_DC(SR, GT)
                    length += images.size(0)
                   
                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length
                unet_score = DC

                all_DC_valid.append(DC)
                valid_loss_list.append(valid_loss / length)
                valid_acc_list.append(acc)
                valid_PC_list.append(PC)
                valid_SP_list.append(SP)

               
                print('##################mean_valid_DC:%.4f####################' % np.mean(all_DC_valid))

                print(
                    'Epoch [%d/%d], Loss: %.4f, \n[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                    epoch + 1, self.num_epochs, valid_loss / length, acc, SE, SP, PC, F1, JS, DC))

               
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best %s model DC score : %.4f' % (self.model_type, best_unet_score))
                    torch.save(best_unet, unet_path)

            # ===================================== Test ====================================#

            del self.unet
            del best_unet
            self.build_model()
            self.unet.load_state_dict(torch.load(unet_path))

            self.unet.train(False)
            self.unet.eval()

            acc = 0.  
            SE = 0.  
            SP = 0.  
            PC = 0.  
            F1 = 0.  
            JS = 0.  
            DC = 0.  
            length = 0
            test_loss = 0
            test_loss_list = []
            
            start = time.clock()
            
            for i, (images, GT, filename) in enumerate(self.test_loader):
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR = torch.sigmoid(self.unet(images))

                acc += get_accuracy(SR, GT)
                SE += get_sensitivity(SR, GT)
                SP += get_specificity(SR, GT)
                PC += get_precision(SR, GT)
                F1 += get_F1(SR, GT)
                JS += get_JS(SR, GT)
                DC += get_DC(SR, GT)
                length += images.size(0)

                SR_flat = SR.view(SR.size(0), -1)  
                GT_flat = GT.view(GT.size(0), -1)  

                loss = self.criterion(SR_flat, GT_flat)
                test_loss += loss.item()

                torchvision.utils.save_image(SR.data.cpu(),
                                             os.path.join(self.SR_result_path, '%s.bmp' % filename))
               
            acc = acc / length
            SE = SE / length
            SP = SP / length
            PC = PC / length
            F1 = F1 / length
            JS = JS / length
            DC = DC / length

            unet_score = DC
            test_loss_list.append(test_loss / length)

            print('Loss: %.4f, \n[test] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
            test_loss / length, acc, SE, SP, PC, F1, JS, DC))
            
            end = time.clock()
            print(end - start)




