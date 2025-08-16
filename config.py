from pathlib import Path
import torch

SMHD_SAVE_PATH = Path('./Datasets/SMHD') 
IAM_SAVE_PATH = "./datasets/lines"
TEST_SAVE_PATH = "./Datasets/test_lines"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESUME = None
SAVED_FEATURES = True

# Model Hyperparameters
HIDDEN_SIZE=256
NUM_LAYERS=2

# Train Hyperparameters
LR=1e-4
BS=128
DECAY=1e-10
NUM_EPOCHS = 300