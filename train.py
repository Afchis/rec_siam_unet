import torch
from torch.utils.tensorboard import SummaryWriter

from args import *
from model_head import *
from dataloader_5x import *
from loss_metric import *


writer = SummaryWriter()

model = ModelDisigner()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('Tensorboard graph name: ')
GRAPH_NAME = input()


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
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			if iter % 2 == 0:
				print(loss.item())
			if iter % 10 == 0:
				 writer.add_scalars('%s_loss' % GRAPH_NAME, {'train' : loss.item()}, epoch)
			iter += 1

		except RuntimeError:
			pass
print('Tensorboard graph name: ', GRAPH_NAME)
writer.close()

torch.save(model.state_dict(), 'pathignore/weights/test.pth')
print('WEIGHTS IS SAVED: pathignore/weights/test.pth')