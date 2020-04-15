import torch

BATCH_SIZE = 4
TIMESTEPS = 5
INPUT_CHANNELS = 3
NUM_CLASSES = 2
SEARCH_SIZE = 256
TARGET_SIZE = 128
UNET_NUM_CLASSES = 21

MODE = '--'
CELL_MODEL = 'Gru'

UNET_WEIGHTS = "pathignore/weights/weights_voc2012.pth"

DEVICE = "cuda:0"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
device

# tensorboard --logdir=runs