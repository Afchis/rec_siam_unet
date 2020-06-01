import os
import glob
import numpy as np

from PIL import Image, ImageDraw
import scipy.ndimage.morphology as morph

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from args import *


import json
with open('/storage/ProtopopovI/_data_/COCO/2014/annotations/person_keypoints_train2014.json') as data_file:    
    data_json = json.load(data_file)


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
        self.file_names = sorted(os.listdir("/storage/ProtopopovI/_data_/COCO/2014/train2014/"))
        
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
        
        label1 = (object==0).float()
        depth1 = torch.tensor(morph.distance_transform_edt(np.asarray(label1[0])))
        label2 = (label1==0).float()
        depth2 = torch.tensor(morph.distance_transform_edt(np.asarray(label2[0])))
        depth = (depth1 + depth2).float().unsqueeze(0)
        label = torch.stack([label1, label2], dim=1)
        labels = torch.cat([labels, label], dim=0)
        depths = torch.cat([depths, depth], dim=0)
        score_label = self.transform_score_label(depth2).unsqueeze(0)
        score_labels = torch.cat([score_labels, score_label], dim=0)
        labels = labels.squeeze()
        score_labels = score_labels.squeeze()
        
        return labels, depths, score_labels

    def Choise_feat(self, label, score_label, x=8):
        score_label = score_label[0][1]
        max_value = score_label.max()
        pos = (score_label == max_value).nonzero()#.unsqueeze(0)

        label = label.permute(0, 2, 3, 1)
        i_tensors = torch.tensor([])
        for i in range(label.size(0)):
            i_tensor = label[i][x*pos[i][0]:x*pos[i][0]+x*16, x*pos[i][1]:x*pos[i][1]+x*16, :].unsqueeze(0)
            i_tensors = torch.cat([i_tensors, i_tensor], dim=0)

        label = i_tensors.permute(0, 3, 1, 2)
        return label
    
    def  __getitem__(self, idx):
        file_name = self.file_names[idx]
        bboxs = []
        seg_ids = []
        js = []
        for i in range(len(data_json['images'])):
            if file_name == data_json['images'][i]['file_name']:
                id = data_json['images'][i]['id']
                image_i = i
                for j in range(len(data_json['annotations'])):
                    if id == data_json['annotations'][j]['image_id']:
                        js.append(j)
                        seg_ids.append(data_json['annotations'][j]['id'])
                        bboxs.append(data_json['annotations'][j]['bbox'])
        search = Image.open("/storage/ProtopopovI/_data_/COCO/2014/train2014/" + file_name).convert('RGB')

        box = [bboxs[0][0], bboxs[0][1], bboxs[0][2], bboxs[0][3]]
        target = search.crop([box[0], box[1], box[0]+box[2], box[1]+box[3]])
        
        target = self.target_trans(target)
        search = self.search_trans(search)
        
        mask = Image.new('L', (data_json['images'][image_i]['width'], data_json['images'][image_i]['height']))
        W_H = torch.tensor([box[2]/data_json['images'][image_i]['width'], box[3]/data_json['images'][image_i]['height']])
        # print(W_H)
        idraw = ImageDraw.Draw(mask)
        idraw.polygon(data_json['annotations'][js[0]]['segmentation'][0], fill='white')
        mask = self.search_trans(mask)
        label, depth, score_label = self.get_labels(mask)

        search, label, depth, score_label = search.unsqueeze(0), label.unsqueeze(0), depth.unsqueeze(0), score_label.unsqueeze(0)
        label = self.Choise_feat(label, score_label)
        depth = self.Choise_feat(depth, score_label)
        return target, search, label, depth, score_label

    def __len__(self):
    	return len(os.listdir("/storage/ProtopopovI/_data_/COCO/2014/train2014/"))


train_dataset = TrainPerson()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=1,
                          shuffle=True)


if __name__ == '__main__':
	# print('Write number of image in dataset: ')
	# inp = int(input())
	target, search, label, depth, score_label = train_dataset[9]
	print('target.shape', target.shape)
	print('search.shape', search.shape)
	print('label.shape', label.shape)
	print('depth.shape', depth.shape)
	print('score_label.shape', score_label.shape)
	# print(score_label)