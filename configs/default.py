"""
Default configuration for Noise Genome Estimator

These can be overridden via command line arguments.
"""

# Training
NUM_EPOCHS = 300
BATCH_SIZE = 16
MAX_DEVICE_BATCH_SIZE = 16
NUM_WORKERS = 16
LEARNING_RATE = 1.5e-4
WARMUP_EPOCHS = 30
LR_GAMMA = 0.5

# Checkpointing
CHECKPOINT_SAVE_PATH = './checkpoints'
CHECKPOINT_FLASH_FREQUENCY = 5
SAVE_OPTIMIZER = True

# Model
SEED = 3407
MODEL_TYPE = 'vit'  # or 'swin'
CROP_SIZE = 192
PATCH_SIZE = 16
IN_CHANNELS = 3

# Datasets
TRAIN_DATASET_PATH = './datasets/train'
VAL_DATASET_PATH = './datasets/val'
TEST_IMAGE_PATH = './datasets/test'
TEST_PARAM_PATH = './noise_params'

# Visualization
IF_VISDOM_VISUALIZE = True
VISDOM_FLASH_FREQUENCY = 100

# Device
GPU_ID = 1
