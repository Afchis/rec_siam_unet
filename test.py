import torch

from args import *
from model_head import *
from dataloader import *
from loss_metric import *


to_pil = transforms.ToPILImage()

model = ModelDisigner()
model = model.to(device)
model.load_state_dict(torch.load('pathignore/weights/test.pth'))

def save_img(object, object2,  j):
	# for batch in range(BATCH_SIZE):
	imgs = object[0][0][1]
	img = (imgs > 0.3).float()
	# print(imgs.shape)
	img = to_pil(img)
	img.save("../rec_siam_unet/test_output/frame%d_mask.png" % i)

	masks = object2[0][0][1]
	mask = (masks > 0.3).float()
	mask = to_pil(mask)
	mask.save("../rec_siam_unet/test_output/frame%d_pred.png" % i)
	print('save!!!', i)


for i, data in enumerate(train_loader):
	if i < 50:
		target, searchs, labels, depths, score_labels = data
		target, searchs, labels, depths, score_labels \
			 = target.to(device), searchs.to(device), labels.to(device), depths.to(device), score_labels.to(device)
		pred_score, pred_mask = model(target, searchs)
		save_img(labels.cpu(), pred_mask.cpu(), i)

