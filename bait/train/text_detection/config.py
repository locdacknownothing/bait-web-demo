PRETRAINED = False
PRETRAINED_WEIGHTS = "./pths/pretrained/east_vgg16.pth" # ignore this when PRETRAINED=False

TRAIN_IMG_PATH = "/mnt/data1/src/ocr/vinai-vietnamese/train_img"
TRAIN_GT_PATH = "/mnt/data1/src/ocr/vinai-vietnamese/train_gt"
VAL_IMG_PATH = "/mnt/data1/src/ocr/vinai-vietnamese/test_img"
VAL_GT_PATH = "/mnt/data1/src/ocr/vinai-vietnamese/test_gt"

DEFAULT_CKPT_PATH = "./checkpoint"

BATCH_SIZE = 1 # 16
NUM_WORKERS = 4
EPOCH_ITER = 500
LOG_INTERVAL = 10
