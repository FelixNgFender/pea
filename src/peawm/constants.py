import pathlib

# general
DATEFMT_STR_HUMAN = "%Y-%m-%d %H:%M:%S"
DATEFMT_STR = "%Y-%m-%d_%H:%M:%S"

# training settings
USE_ACCELERATOR = True
TRAIN_SPLIT = 0.9
VAL_SPLIT = 1.0 - TRAIN_SPLIT
DATA_DIR = pathlib.Path("data")
TORCH_SEED = 2_147_483_647
SEED = 42
# https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
FP32_MATMUL_PRECISION = "high"  # "highest", "high", "medium"
USE_MIXED_PRECISION = True
# ddp
DDP_RANK = 0
DDP_LOCAL_RANK = 0
DDP_WORLD_SIZE = 1

# hyperparams
BATCH_SIZE = 64  # the number of independent sequences to process at once
LEARNING_RATE = 3e-4
NUM_STEPS = 5000
CONTEXT_SIZE = 256  # the maximum length of predictions

# checkpointing
CKPT_DIR = pathlib.Path("checkpoints")
