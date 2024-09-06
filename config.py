from pathlib import Path
import torch
# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Dataset"
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"
CHECKPOINT_PATH = BASE_DIR / "checkpoint.pt"

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Model parameters
NUM_CLASSES = 2

# Data augmentation parameters
RANDOM_ROTATION = 10
COLOR_JITTER = 0.2

# Normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
