TRAIN_DATASET_DIR = "/mnt/data/src/bait/train/sim/data_sim_processed/train"
VAL_DATASET_DIR = "/mnt/data/src/bait/train/sim/data_sim_processed/val"
NUM_WORKERS = 16

DEVICE_ID = 0
BACKBONE_NAME = "mobilenet_v3_large" # "mobilenet_v3_large"
BATCH_SIZE = 128  # 32
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001  # 0.001
MARGIN = 10

MODEL_SAVE_NAME = f"weights_margin{MARGIN}"
HISTORY_SAVE_NAME = f"history_30k_margin{MARGIN}.pkl"
PRETRAINED = "/mnt/data/src/bait/train/sim/weights/mobilenet_v3_large/weights_margin10/best.pth"  # "/mnt/data/src/dedup/weights/model_160k_25epochs_margin1.0.pth"
TEST_ONLY = False
