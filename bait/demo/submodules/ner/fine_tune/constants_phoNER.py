from phoNER_COVID19_dataset import get_tags, get_label2id, get_id2label


LABELS = get_tags()
LABEL2ID = get_label2id()
ID2LABEL = get_id2label()

BASE_MODEL_NAME = "NlpHUST/electra-base-vn"
SAVE_CKPT = "./train_phoNER3/checkpoints"
SAVE_MODEL = "./train_phoNER3/model"
