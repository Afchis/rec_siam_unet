import torch
import torch.nn as nn
import torch.nn.functional as F

from args import *
from model_parts import *


'''
Model head
'''
class ModelDisigner(nn.Module):
	def __init__(self):
		super(ModelDisigner, self).__init__()
		self.target = BackboneUNet(TARGET_SIZE)
		self.search = BackboneUNet(SEARCH_SIZE)
		self.score_branch = ScoreBranch()
		self.mask_branch = MaskBranch()

	def Correlation_func(self, s_f, t_f): # s_f-->search_feat, t_f-->target_feat
		s_f = s_f.reshape(TIMESTEPS, BATCH_SIZE, s_f.size(1), s_f.size(2), s_f.size(3)) # 5, 2, 256, 32, 32
		t_f = t_f.reshape(-1, 1, t_f.size(2), t_f.size(3)) # 512, 1, 32, 32
		outs = torch.tensor([])
		for i in range(TIMESTEPS):
			out = s_f[i].reshape(1, -1, s_f[i].size(2), s_f[i].size(3)) # 1, 512, 32, 32
			out = F.conv2d(out, t_f, groups=out.size(1))
			out = out.reshape(BATCH_SIZE, s_f[i].size(1), out.size(2), out.size(3))
			outs = torch.cat([outs, out], dim=0)
		return outs

	def Chiose_RoW(self, corr_feat, pos_list):
		i_tensors = torch.tensor([])
		for i in range(corr_feat.size(0)):
			j_tensors = torch.tensor([])
			for j in range(corr_feat.size(1)):
				j_tensor = corr_feat[i][j][pos_list[i][j][0]][pos_list[i][j][1]].unsqueeze(0)
				j_tensors = torch.cat([j_tensors, j_tensor], dim=0)
			i_tensor = j_tensors.unsqueeze(0)
			i_tensors = torch.cat([i_tensors, i_tensor], dim=0)
		return i_tensors

	def forward(self, target, searchs):
		_,  target_feat = self.target(target)
		searchs = searchs.permute(1, 0, 2, 3, 4).reshape(TIMESTEPS*BATCH_SIZE, INPUT_CHANNELS, SEARCH_SIZE, SEARCH_SIZE)
		search_cats, searchs_feat = self.search(searchs) 
		corr_feat = self.Correlation_func(searchs_feat, target_feat) # TIMESTEPS*BATCH_SIZE, 256, 17, 17
		# Score Branch
		score, pos_list = self.score_branch(corr_feat) # TIMESTEPS*BATCH_SIZE, 2, 17, 17 # TIMESTEPS*BATCH_SIZE, 2
		score = score.reshape(TIMESTEPS, BATCH_SIZE, NUM_CLASSES, 17, 17)
		score = score.permute(1, 0, 2, 3, 4) # BATCH_SIZE, TIMESTEPS, NUM_CLASSES, 17, 17
		pos_list = pos_list.reshape(TIMESTEPS, BATCH_SIZE, 2)
		pos_list = pos_list.permute(1, 0, 2,) # BATCH_SIZE, TIMESTEPS, 2
		# Mask Branch
		corr_feat = corr_feat.reshape(TIMESTEPS, BATCH_SIZE, 256, 17, 17).permute(1, 0, 3, 4, 2) # BATCH_SIZE, TIMESTEPS, 256, 17, 17
		masks_feat = self.Chiose_RoW(corr_feat, pos_list)
		# print('score.shape: ', score.shape)
		# print('masks_feat.shape: ', masks_feat.shape)
		mask = self.mask_branch(masks_feat)
		# print(mask.shape)
		search_cats_3 = search_cats[3].reshape(TIMESTEPS, BATCH_SIZE, 512, 32, 32).permute(1, 0, 2, 3, 4) # BATCH_SIZE*TIMESTEPS, 512, 32, 32
		print(search_cats[3].shape)
		# search_cats[0] = search_cats[0].reshape(TIMESTEPS, BATCH_SIZE, 256, 17, 17).permute(1, 0, 3, 4, 2)
		# search_cats[1] = search_cats[1].reshape(TIMESTEPS, BATCH_SIZE, 256, 17, 17).permute(1, 0, 3, 4, 2)
		# search_cats[2] = search_cats[2].reshape(TIMESTEPS, BATCH_SIZE, 256, 17, 17).permute(1, 0, 3, 4, 2)
		# 
		return score, masks_feat

if __name__ == '__main__':
	model = ModelDisigner()
	target = torch.rand([BATCH_SIZE, 3, 128, 128])
	searchs = torch.rand([BATCH_SIZE, TIMESTEPS, 3, 256, 256])
	score, pos_list = model(target, searchs)