# config.py

import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'Resized_dataset', 'train_resized')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
LOG_DIR = os.path.join(PROJECT_ROOT, 'runs', 'MRI_Segmentation')

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 8
LR = 5e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Other
AMP = True  # Enable mixed precision training
NUM_WORKERS = 4
