import torch

BATCH_SIZE = 2
TIMESTEPS = 5
INPUT_CHANNELS = 3
NUM_CLASSES = 2
SEARCH_SIZE = 256
TARGET_SIZE = 128

MODE = 'Rec'

UNET_WEIGHTS = "pathignore/weights/weights.pth"

DEVICE = "cuda:1"
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
device

