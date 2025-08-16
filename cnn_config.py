import torch

EMNIST_SAVE_PATH = "Datasets/EMNIST"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train Hyperparameters
LR=1e-5
BS=64
DECAY=1e-12
NUM_EPOCHS = 200