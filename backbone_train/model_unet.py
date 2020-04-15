import torch
import torch.nn as nn
import torch.nn.functional as F

from args import *


'''
Unet parts
'''
class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvRelu, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convrelu = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
                                      nn.BatchNorm2d(self.out_channels),
                                      nn.ReLU()
                                      )

    def forward(self, x):
        return self.convrelu(x)


class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.maxpool(x)


class UpAndCat(nn.Module):    
    def __init__(self):
        super(UpAndCat, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x_up, x_cat):
        out = self.up(x_up)
        out = torch.cat([out, x_cat], dim=1)
        return out


class Cat(nn.Module):    
    def __init__(self):
        super(Cat, self).__init__() 

    def forward(self, x_up, x_cat):
        out = torch.cat([x_up, x_cat], dim=1)
        return out


class Unet(nn.Module):
	def __init__(self, input_size=SEARCH_SIZE):
		super(Unet, self).__init__()
		self.input_size = input_size
		self.ch_list = [INPUT_CHANNELS, 128, 256, 512, 1024, 2048]
		self.input_x2 = int(self.input_size / 2)
		self.input_x4 = int(self.input_size / 4)
		self.input_x8 = int(self.input_size / 8)

		##### Down layers #####
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.down1 = nn.Sequential(
			ConvRelu(self.ch_list[0], self.ch_list[1]),
			ConvRelu(self.ch_list[1], self.ch_list[1])
			)
		self.down2 = nn.Sequential(
			ConvRelu(self.ch_list[1], self.ch_list[2]),
			ConvRelu(self.ch_list[2], self.ch_list[2])
			)
		self.down3 = nn.Sequential(
			ConvRelu(self.ch_list[2], self.ch_list[3]),
			ConvRelu(self.ch_list[3], self.ch_list[3])
			)
		self.down4 = nn.Sequential(
			ConvRelu(self.ch_list[3], self.ch_list[4]),
			ConvRelu(self.ch_list[4], self.ch_list[4])
			)
		self.bottom = nn.Sequential(
			ConvRelu(self.ch_list[4], self.ch_list[5]),
			ConvRelu(self.ch_list[5], self.ch_list[5])
			)

		##### Up layers #####
		self.cat_4 = Cat()
		self.up_conv_4 = nn.Sequential(
			ConvRelu(self.ch_list[5]+self.ch_list[4], self.ch_list[4]),
			ConvRelu(self.ch_list[4], self.ch_list[4])
			)
		self.up_cat_3 = UpAndCat()
		self.up_conv_3 = nn.Sequential(
			ConvRelu(self.ch_list[4]+self.ch_list[3], self.ch_list[3]),
			ConvRelu(self.ch_list[3], self.ch_list[3])
			)
		self.up_cat_2 = UpAndCat()
		self.up_conv_2 = nn.Sequential(
			ConvRelu(self.ch_list[3]+self.ch_list[2], self.ch_list[2]),
			ConvRelu(self.ch_list[2], self.ch_list[2])
			)
		self.up_cat_1 = UpAndCat()
		self.up_conv_1 = nn.Sequential(
			ConvRelu(self.ch_list[2]+self.ch_list[1], self.ch_list[1]),
			ConvRelu(self.ch_list[1], self.ch_list[1])
			)
		##### Final layers #####
		self.final = nn.Sequential(
			nn.Conv2d(self.ch_list[1], UNET_NUM_CLASSES, kernel_size=1),
			# nn.Sigmoid(),
			)


	def forward(self, x):
		down1_feat = self.down1(x)
		pool1 = self.pool(down1_feat)
		down2_feat = self.down2(pool1)
		pool2 = self.pool(down2_feat)
		down3_feat = self.down3(pool2)
		pool3 = self.pool(down3_feat)
		down4_feat = self.down4(pool3)

		bottom_feat = self.bottom(down4_feat)

		up_feat4 = self.cat_4(bottom_feat, down4_feat)
		up_feat4 = self.up_conv_4(up_feat4)
		up_feat3 = self.up_cat_3(up_feat4, down3_feat)
		up_feat3 = self.up_conv_3(up_feat3)
		up_feat2 = self.up_cat_2(up_feat3, down2_feat)
		up_feat2 = self.up_conv_2(up_feat2)
		up_feat1 = self.up_cat_1(up_feat2, down1_feat)
		up_feat1 = self.up_conv_1(up_feat1)

		out = F.log_softmax(self.final(up_feat1), 1)

		return out