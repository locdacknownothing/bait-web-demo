from dataset import get_labels, get_label2id, get_id2label

scheme = "iob2"

LABELS = get_labels(scheme)
LABEL2ID = get_label2id(scheme)
ID2LABEL = get_id2label(scheme)

TRAIN_DATASET = "/mnt/data/data/Data_Ner/2412/train.xlsx"
VAL_DATASET = "/mnt/data/data/Data_Ner/2412/val.xlsx"
TEST_DATASET = "/mnt/data/data/Data_Ner/2412/test.xlsx"
BASE_MODEL = (
    # "/mnt/data/src/ner/models/model_2511"
    "NlpHUST/electra-base-vn"
)
SAVE_CKPT = "./train_original15/checkpoints"
SAVE_MODEL = "./train_original15/model"
CKPT = None
