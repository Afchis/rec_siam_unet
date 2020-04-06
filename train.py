import torch

from args import *
from dataloader import *
from model_head import *
from loss_metric import *

model = ModelDisigner()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

iter = 0
for epoch in range(15):
	print('*'*10, 'epoch: ', epoch, '*'*10)
	for i, data in enumerate(train_loader):
		target, search, label, depth, score_label = data
		# target, search, label, depth, score_label \
		#  = target.to(device), search.to(device), label.to(device), depth.to(device), score_label.to(device)
		pred_score, pred_mask = model(search, target)
		loss = all_losses(pred_mask, label, depth, pred_score, score_label)
		if iter % 20 == 0:
			print(loss.item())
			# print(pred_mask[0][1][128])
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		iter += 1