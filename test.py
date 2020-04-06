import torch

from args import *
from model_head import *
from dataloader import *
from loss_metric import *


to_pil = transforms.ToPILImage()

model = ModelDisigner()
model = model.to(device)
model.load_state_dict(torch.load('pathignore/weights/test.pth'))

def save_img(object, j):
	for batch in range(BATCH_SIZE):
		imgs = object[batch][4][1]
		img = (imgs > 0.5).float()
		print(imgs.shape)
		img = to_pil(img)
		img.save("../rec_siam_unet/test_output/frame%d.png" % i)
		print('save!!!', i)


for i, data in enumerate(train_loader):
	target, searchs, labels, depths, score_labels = data
	target, searchs, labels, depths, score_labels \
		 = target.to(device), searchs.to(device), labels.to(device), depths.to(device), score_labels.to(device)
	pred_score, pred_mask = model(target, searchs)
	save_img(pred_mask.cpu(), i)
