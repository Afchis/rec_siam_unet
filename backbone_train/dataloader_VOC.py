from args import *

import os
import numpy as np

from PIL import Image
import scipy.ndimage.morphology as morph

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


# way to the data folders
FOLDER_DATA = "/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/JPEGImages/"
FOLDER_MASK = "/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/SegmentationClass/"
FOLDER_DATA_VAL = "/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/JPEGImages/"
FOLDER_MASK_VAL = "/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/SegmentationClass/"
FOLDER_DATA_TEST = "/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/JPEGImages/"
FOLDER_MASK_TEST = "/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/SegmentationClass/"

FOLDER_DATA_NAMES = open("/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt", "r").read().split('\n')
FOLDER_MASK_NAMES = open("/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt", "r").read().split('\n')
FOLDER_DATA_VAL_NAMES = open("/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", "r").read().split('\n')
FOLDER_MASK_VAL_NAMES = open("/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", "r").read().split('\n')
FOLDER_DATA_TEST_NAMES = open("/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", "r").read().split('\n')
FOLDER_MASK_TEST_NAMES = open("/storage/ProtopopovI/_data_/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt", "r").read().split('\n')

FOLDER_DATA_NAMES = FOLDER_DATA_NAMES[:-1]
FOLDER_MASK_NAMES = FOLDER_MASK_NAMES[:-1]
FOLDER_DATA_VAL_NAMES = FOLDER_DATA_VAL_NAMES[:-1]
FOLDER_MASK_VAL_NAMES = FOLDER_MASK_VAL_NAMES[:-1]
FOLDER_DATA_TEST_NAMES = FOLDER_DATA_TEST_NAMES[:-1]
FOLDER_MASK_TEST_NAMES = FOLDER_MASK_TEST_NAMES[:-1]

# transforms
transform = transforms.Compose([
                              transforms.Resize((SEARCH_SIZE, SEARCH_SIZE), interpolation = 0),
                              transforms.ToTensor()
                              ])

to_tensor = transforms.ToTensor()

resize = transforms.Resize((SEARCH_SIZE, SEARCH_SIZE),  interpolation = 0)

'''
Dataloader
'''
VOC_COLORS = np.array(
              [[0, 0, 0],[128, 0, 0],[0, 128, 0],[128, 128, 0], [0, 0, 128],
              [128, 0, 128],[0, 128, 128],[128, 128, 128],[64, 0, 0],[192, 0, 0],[64, 128, 0],
              [192, 128, 0],[64, 0, 128],[192, 0, 128],[64, 128, 128],[192, 128, 128],
              [0, 64, 0],[128, 64, 0],[0, 192, 0],[128, 192, 0],[0, 64, 128],[224, 224, 192]]
                          )
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor', 'void']
def get_labels_voc(mask):
    for i, color in enumerate(VOC_COLORS):
        if i == 0:
            labels = torch.tensor(np.all(mask==color, axis=-1), dtype=torch.float).unsqueeze(0)
            depths = morph.distance_transform_edt(labels.numpy())
        else:
            labels = torch.cat((labels, torch.tensor(np.all(mask==color, axis=-1), dtype=torch.float).unsqueeze(0)), 0)
            depths += morph.distance_transform_edt(labels[-1].numpy())
    return labels, torch.from_numpy(depths).squeeze(0)


class TrainVOC2012Data(Dataset):
    def __init__(self):
        super().__init__()
        self.folder_data = FOLDER_DATA
        self.folder_mask = FOLDER_MASK

        self.folder_data_names = FOLDER_DATA_NAMES
        self.folder_mask_names = FOLDER_MASK_NAMES

    def __getitem__(self, idx):
        image = transform(Image.open(self.folder_data + self.folder_data_names[idx] + '.jpg').convert('RGB'))
        mask = np.array(resize(Image.open(self.folder_mask + self.folder_mask_names[idx] + '.png').convert('RGB')))
        label, depth = get_labels_voc(mask)
        label = label[:21]
        return image, label, depth

    def __len__(self):
        return len(self.folder_data_names)


class ValidVOC2012Data(Dataset):
    def __init__(self):
        super().__init__()
        self.folder_data = FOLDER_DATA_VAL
        self.folder_mask = FOLDER_MASK_VAL

        self.folder_data_names = FOLDER_DATA_VAL_NAMES
        self.folder_mask_names = FOLDER_MASK_VAL_NAMES

    def __getitem__(self, idx):
        image = transform(Image.open(self.folder_data + self.folder_data_names[idx] + '.jpg').convert('RGB'))
        mask = np.array(resize(Image.open(self.folder_mask + self.folder_mask_names[idx] + '.png').convert('RGB')))
        label, depth = get_labels_voc(mask)
        label = label[:21]
        return image, label, depth

    def __len__(self):
        return len(self.folder_data_names)


class TestVOC2012Data(Dataset):
    def __init__(self):
        super().__init__()
        self.folder_data = FOLDER_DATA_TEST
        self.folder_mask = FOLDER_MASK_TEST

        self.folder_data_names = FOLDER_DATA_TEST_NAMES
        self.folder_mask_names = FOLDER_MASK_TEST_NAMES

    def __getitem__(self, idx):
        image = transform(Image.open(self.folder_data + self.folder_data_names[idx] + '.jpg').convert('RGB'))
        mask = np.array(resize(Image.open(self.folder_mask + self.folder_mask_names[idx] + '.png').convert('RGB')))
        label, depth = get_labels_voc(mask)
        label = label[:21]
        return image, label, depth

    def __len__(self):
        return len(self.folder_data_names)


train_dataset = TrainVOC2012Data()
valid_dataset = ValidVOC2012Data()
test_dataset = TestVOC2012Data()

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         num_workers=1,
                         shuffle=False)

data_loaders = {
    'train' : train_loader,
    'valid' : valid_loader,
    'test' : test_loader
}

dataset_sizes = {
    'train': len(train_dataset),
    'valid': len(valid_dataset),
    'test': len(test_dataset)
}


if __name__ == '__main__':
    image, masks, depth = train_dataset[0]
    print(image.shape, masks.shape, depth.shape)