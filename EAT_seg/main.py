# ! usr/bin/env python3
# -*- coding:utf-8 -*-
# author:lvanlee   time:2021/4/17
import argparse
import os

from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

 
def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['DilaLab6','NestedUNet','U_Net', 'PSPNet','FCN32s', 'AttU_Net', 'IRU_Net', 'segnet']:
        print('ERROR!! model_type should be selected in dp_U_Net/IRCd5U_Net/DilaLab/U_Net/R2U_Net/AttU_Net/R2AttU_Net/IRAttU_Net/IRU_Net/segnet')
        print('Your input for model_type was %s' % config.model_type)
        return

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    config.model_path = os.path.join(config.model_path, config.model_type)
    if not os.path.exists(config.SR_result_path):
        os.makedirs(config.SR_result_path)
    config.SR_result_path = os.path.join(config.SR_result_path, config.model_type)
    if not os.path.exists(config.EAT_GT_path):
        os.makedirs(config.EAT_GT_path)
    config.EAT_GT_path = os.path.join(config.EAT_GT_path, config.model_type)
    if not os.path.exists(config.EAT_images_path):
        os.makedirs(config.EAT_images_path)
    config.EAT_images_path = os.path.join(config.EAT_images_path, config.model_type)

    
    lr = 0.0001
    augmentation_prob = 0.5
    epoch = random.choice([50])  
    decay_ratio = random.random()
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)
        
    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=1) 
    parser.add_argument('--num_workers', type=int, default=8) 
    
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        
    parser.add_argument('--beta2', type=float, default=0.999)          
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='DilaLab6', help='IRDdU_Net/DAttU_Net/DilaLab6/IRDdU_Net/IRDU_Net/IRd5U_Net/IRCdAttU_Net/IRCdU_Net/FCDenseNet103/PSPNet/segnet/U_Net/AttU_Net')
    
    parser.add_argument('--train_path', type=str, default='../dataset/train/')
    parser.add_argument('--valid_path', type=str, default='../dataset/valid/')
    parser.add_argument('--test_path', type=str, default='../dataset/test/')
    parser.add_argument('--origin_data_path', type=str, default="../dataset/Training_Input/")
    parser.add_argument('--origin_GT_path', type=str, default="../dataset/Training_Input_GT/")
    
    parser.add_argument('--model_path', type=str, default='../models/')
    parser.add_argument('--SR_result_path', type=str, default='../SR_result/')
    parser.add_argument('--EAT_GT_path', type=str,
                        default='../EAT_GT/')
    parser.add_argument('--EAT_images_path', type=str,
                        default='../EAT_images/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
