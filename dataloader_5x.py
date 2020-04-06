import os
import glob
import numpy as np

from PIL import Image
import scipy.ndimage.morphology as morph

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from args import *


class TrainPerson(Dataset):
	def __init__(self):
		super().__init__()
		self.target_trans = transforms.Compose([
			transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=0),
			transforms.ToTensor()
			])
		self.search_trans = transforms.Compose([
			transforms.Resize((SEARCH_SIZE, SEARCH_SIZE), interpolation=0),
			transforms.ToTensor()
			])

	def list_dir(self, object):
		return sorted(os.listdir("../rec_siam_unet/pathignore/data/train/" + object))

	def transform_score_label(self, depth2):
		depth2 = depth2.reshape(1, 1, depth2.size(0), depth2.size(1))
		max_value = depth2.max()
		depth2 = (depth2 == max_value).float()
		score_label = F.max_pool2d(depth2, kernel_size=(16, 16), padding=8, stride=16)
		score_zero = (score_label == 0).float()
		score_label = torch.stack([score_zero, score_label], dim=1).squeeze()
		return score_label

	def get_labels(self, object):
		labels = torch.tensor([])
		depths = torch.tensor([])
		score_labels = torch.tensor([])
		for i in range(object.size(0)):
			label1 = (object[i]==0).float()
			depth1 = torch.tensor(morph.distance_transform_edt(np.asarray(label1[0])))
			label2 = (label1==0).float()
			depth2 = torch.tensor(morph.distance_transform_edt(np.asarray(label2[0])))
			depth = (depth1 + depth2).float().unsqueeze(0)
			label = torch.stack([label1, label2], dim=1)
			labels = torch.cat([labels, label], dim=0)
			depths = torch.cat([depths, depth], dim=0)
			score_label = self.transform_score_label(depth2).unsqueeze(0)
			score_labels = torch.cat([score_labels, score_label], dim=0)
		return labels, depths, score_labels

	def __getitem__(self, idx):
		if idx < len(self.list_dir("imgs/0")) - TIMESTEPS:
			names = self.list_dir("imgs/0")
			target = Image.open("../rec_siam_unet/pathignore/data/train/0.jpg").convert('RGB')
			target = self.target_trans(target)

			searchs = torch.tensor([])
			masks = torch.tensor([])
			for time in range(TIMESTEPS):
				search = Image.open("../rec_siam_unet/pathignore/data/train/imgs/0/" + names[idx]).convert('RGB')
				search = self.search_trans(search).unsqueeze(dim=0)
				searchs = torch.cat([searchs, search], dim=0)
				mask = Image.open("../rec_siam_unet/pathignore/data/train/anno/0/" + names[idx][:5] + ".png").convert('L')
				mask = self.search_trans(mask).unsqueeze(dim=0)
				masks = torch.cat([masks, mask], dim=0)

		else:
			names = self.list_dir("imgs/1")
			target = Image.open("../rec_siam_unet/pathignore/data/train/1.jpg").convert('RGB')
			target = self.target_trans(target)

			searchs = torch.tensor([])
			masks = torch.tensor([])
			for time in range(TIMESTEPS):
				search = Image.open("../rec_siam_unet/pathignore/data/train/imgs/1/" + names[idx-71]).convert('RGB')
				search = self.search_trans(search).unsqueeze(dim=0)
				searchs = torch.cat([searchs, search], dim=0)
				mask = Image.open("../rec_siam_unet/pathignore/data/train/anno/1/" + names[idx-71][:5] + ".png").convert('L')
				mask = self.search_trans(mask).unsqueeze(dim=0)
				masks = torch.cat([masks, mask], dim=0)

		labels, depths, score_labels = self.get_labels(masks)
		return target, searchs, labels, depths, score_labels

	def __len__(self):
		path_0 = len(self.list_dir("imgs/0"))
		path_1 = len(self.list_dir("imgs/1"))
		return path_0 + path_1


train_dataset = TrainPerson()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=1,
                          shuffle=True)



if __name__ == '__main__':
	print('Write number of image in dataset: ')
	inp = int(input())
	target, searchs, labels, depths, score_labels = train_dataset[inp]
	print('target.shape: ', target.shape)
	print('searchs.shape: ', searchs.shape)
	print('labels.shape: ', labels.shape)
	print('depths.shape: ', depths.shape)
	print('score_labels.shape: ', score_labels.shape)
	print('len(train_dataset): ', len(train_dataset))