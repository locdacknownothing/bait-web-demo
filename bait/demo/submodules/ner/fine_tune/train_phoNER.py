import sys

sys.path.append("../")

import warnings

warnings.filterwarnings("ignore")

from functools import partial
import json
from pathlib import Path

import numpy as np
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
)
from transformers import TrainingArguments, Trainer
from datasets import Dataset, load_metric

from phoNER_COVID19_dataset import extract_data
from fine_tune.metrics import compute_metrics
from fine_tune.constants_phoNER import *


def tokenize_and_align_labels(
    data, tokenizer, label2id=LABEL2ID, label_all_tokens=True
):
    """
    ## The below function does 2 jobs

    1. set -100 as the label for special tokens
    2. mask the subword representations after the first subword
    """
    words = [x["words"] for x in data]
    tags = [[label2id[tag] for tag in x["tags"]] for x in data]
    tokenized_input = tokenizer(words, padding=True, is_split_into_words=True)
    labels = []

    for i, label in enumerate(tags):
        word_ids = tokenized_input.word_ids(batch_index=i)
        previous_word_id = None

        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                # set -100 b/c these special characters
                # are ignored by pytorch when training
                label_ids.append(-100)
            elif word_id != previous_word_id:
                # if current word_id != prev when it's the most regular case
                # and add the corresponding token
                label_ids.append(label[word_id])
            else:
                # for sub-word which has the same word_id
                # set -100 as well only if label_all_tokens=False
                if not label_all_tokens:
                    label_ids.append(label[word_id])
                else:
                    label_ids.append(-100)

            previous_word_id = word_id

        labels.append(label_ids)

    tokenized_input["labels"] = labels
    return tokenized_input


def load_model(
    base_model_name=BASE_MODEL_NAME,
    label2id=LABEL2ID,
    id2label=ID2LABEL,
):
    model = AutoModelForTokenClassification.from_pretrained(
        base_model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    for param in model.parameters():
        param.data = param.data.contiguous()

    return model


if __name__ == "__main__":
    train_data = extract_data("../data/phoNER_COVID19/train_syllable.json")
    val_data = extract_data("../data/phoNER_COVID19/dev_syllable.json")

    base_model_name = BASE_MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.model_max_length = 512
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    tokenized_train_data = tokenize_and_align_labels(train_data, tokenizer)
    tokenized_val_data = tokenize_and_align_labels(val_data, tokenizer)

    train_dataset = Dataset.from_dict(tokenized_train_data.data)
    val_dataset = Dataset.from_dict(tokenized_val_data.data)

    model = load_model(base_model_name)
    args = TrainingArguments(
        output_dir=SAVE_CKPT,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        load_best_model_at_end=True,
        # metric_for_best_model="seqeval",
        # greater_is_better=True,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=15,  # 100
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    custom_compute_metrics = partial(compute_metrics, label_list=LABELS, metric=metric)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=custom_compute_metrics,
    )

    trainer.train(
        # resume_from_checkpoint="/mnt/data/src/ner/fine_tune/train_phoNER2/checkpoints/checkpoint-600",
    )
    trainer.save_model(SAVE_MODEL)
