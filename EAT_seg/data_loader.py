import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


class ImageFolder(data.Dataset):
	def __init__(self, root, image_size=512, mode='train', augmentation_prob=0):
		
		self.root = root    
		self.GT_paths = root[:-1]+'_GT/'
		self.image_paths = [os.path.join(root,x) for x in os.listdir(root)]   
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0, 90, 180, 270]  
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode, len(self.image_paths)))



	def __getitem__(self, index):
		
		image_path = self.image_paths[index]
		filename = image_path.split('/')[-1][:-len(".bmp")]
		GT_path = self.GT_paths + filename + '.bmp'
		image = Image.open(image_path).convert('L')
		GT = Image.open(GT_path).convert('L')
		


		aspect_ratio = image.size[1]/image.size[0]

		Transform = []
		p_transform = random.random()

		if (self.mode == 'train') and p_transform <= self.augmentation_prob:
			RotationRange = random.randint(-15, 15) 
			Transform.append(T.RandomRotation((RotationRange, RotationRange)))
			
			Transform = T.Compose(Transform)

			image = Transform(image)
			GT = Transform(GT)
			if random.random() < 0.5:
				image = F.hflip(image)
				GT = F.hflip(GT)

			if random.random() < 0.5:
				image = F.vflip(image)
				GT = F.vflip(GT)

			image = Transform(image)

			Transform =[]


		
		Transform.append(T.ToTensor())   
		Transform = T.Compose(Transform)
		
		image = Transform(image)
		GT = Transform(GT)


		return image, GT, filename



	def __len__(self):
		
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.5):
	
	
	dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers, drop_last=True)
	return data_loader




