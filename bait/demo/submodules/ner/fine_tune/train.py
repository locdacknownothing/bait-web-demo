from os.path import join, basename, dirname
import sys

sys.path.append(join(dirname(__file__), ".."))

import warnings

warnings.filterwarnings("ignore")

from functools import partial
from time import time

from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
)
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import load_metric

from dataset import clean_data, get_conll_dataset, get_train_dataset
from file import read_df
from fine_tune.metrics import compute_metrics
from fine_tune.constants import *
from export import Exporter
from utils import (
    get_gpu_utilization,
    human_readable_time,
    get_current_date,
)


def load_datasets(
    tokenizer,
    train_data_path=TRAIN_DATASET,
    val_data_path=VAL_DATASET,
    label2id=LABEL2ID,
    labels=LABELS,
):
    train_raw_data = read_df(train_data_path)
    val_raw_data = read_df(val_data_path)

    train_cleaned_data = clean_data(train_raw_data, label2id=label2id, scheme="iob2")
    val_cleaned_data = clean_data(val_raw_data, label2id=label2id)

    # New approach, use one tag
    # from dataset import list2one, one2list, upsample_data

    # train_cleaned_data = list2one(train_cleaned_data)
    # train_cleaned_data = upsample_data(train_cleaned_data)
    # train_cleaned_data = one2list(train_cleaned_data)

    train_conll_dataset = get_conll_dataset(
        train_cleaned_data,
        labels=labels,
        augment_data=False,
    )
    val_conll_dataset = get_conll_dataset(val_cleaned_data, labels=labels)

    train_dataset = get_train_dataset(train_conll_dataset, tokenizer)
    val_dataset = get_train_dataset(val_conll_dataset, tokenizer)

    return train_dataset, val_dataset


def load_model(
    base_model_name=BASE_MODEL,
    label2id=LABEL2ID,
    id2label=ID2LABEL,
):
    model = AutoModelForTokenClassification.from_pretrained(
        base_model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        device_map="cuda",
    )

    for param in model.parameters():
        param.data = param.data.contiguous()

    return model


def load_tokenizer(
    base_model_name=BASE_MODEL,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.model_max_length = 512
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    return tokenizer


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    train_dataset, val_dataset = load_datasets(tokenizer)

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Validation dataset: {len(val_dataset)}")

    model = load_model()
    args = TrainingArguments(
        output_dir=SAVE_CKPT,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=32,
        num_train_epochs=2000,
        learning_rate=1e-5,
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
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=500)],
    )

    # Start measuring train time
    start_train_time = time()
    try:
        trainer.train(resume_from_checkpoint=CKPT if CKPT else None)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught! Saving results...")
    end_train_time = time()
    # End train time

    trainer.save_model(SAVE_MODEL)
    train_time = human_readable_time(end_train_time - start_train_time)

    metrics = trainer.evaluate()
    gpu_usage = get_gpu_utilization()

    train_exporter = Exporter(
        train_id=basename(dirname(SAVE_MODEL)),
        train_set_volume=len(train_dataset),
        val_set_volume=len(val_dataset),
        total_volume=len(train_dataset) + len(val_dataset),
        precision="{:.4f}".format(metrics["eval_precision"]),
        recall="{:.4f}".format(metrics["eval_recall"]),
        f1="{:.4f}".format(metrics["eval_f1"]),
        accuracy="{:.4f}".format(metrics["eval_accuracy"]),
        train_time=train_time,
        gpu_usage=gpu_usage,
        date=get_current_date(),
    )

    train_exporter.to_csv(join(SAVE_MODEL, "../train_results.csv"))
