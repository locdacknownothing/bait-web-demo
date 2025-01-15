TRAIN_DATASET_DIR = "/mnt/data/src/bait/train/sim/data_sim_seg_30k/train"
VAL_DATASET_DIR = "/mnt/data/src/bait/train/sim/data_sim_seg_30k/val"
NUM_WORKERS = 16

DEVICE_ID = 0
BACKBONE_NAME = "mobilenet_v3_large"
BATCH_SIZE = 128  # 32
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001  # 0.001
MARGIN = 1.0

MODEL_SAVE_NAME = f"model_seg_30k_margin{MARGIN}"
HISTORY_SAVE_NAME = f"history_seg_30k_margin{MARGIN}.pkl"
PRETRAINED = ""  # "/mnt/data/src/dedup/weights/model_160k_25epochs_margin1.0.pth"
TEST_ONLY = False
