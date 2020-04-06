import torch

from args import *
from model_head import *
from dataloader import *
from loss_metric import *


model = ModelDisigner()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


iter = 0
for epoch in range(15):
	print('*'*10, 'epoch: ', epoch, '*'*10)
	for i, data in enumerate(train_loader):
		target, searchs, labels, depths, score_labels = data
		target, searchs, labels, depths, score_labels \
		 = target.to(device), searchs.to(device), labels.to(device), depths.to(device), score_labels.to(device)
		try:
			pred_scores, pred_masks = model(target, searchs)		
			loss = all_losses(pred_masks, labels, depths, pred_scores, score_labels)
			if iter % 2 == 0:
				print(loss.item())
				# print(pred_mask[0][1][128])
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			iter += 1
		except RuntimeError:
			pass

torch.save(model.state_dict(), 'pathignore/weights/test.pth')
print('WEIGHTS IS SAVED: pathignore/weights/test.pth')