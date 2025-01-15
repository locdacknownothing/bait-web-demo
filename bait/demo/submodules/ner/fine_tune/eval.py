import argparse
from functools import partial
from os.path import basename, dirname, join
import sys

sys.path.append(join(dirname(__file__), "../"))

import numpy as np
from datasets import Dataset
import evaluate
from evaluate import evaluator
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
)

from fine_tune.train import read_df, load_tokenizer
from fine_tune.constants import *
from fine_tune.metrics import compute_metrics
from dataset import clean_data, get_conll_dataset, get_train_dataset
from export import Exporter
from utils import get_current_date


def get_eval_data(data_path: str, tokenizer: AutoTokenizer):
    raw_data = read_df(data_path)
    cleaned_df = clean_data(raw_data, label2id=LABEL2ID)
    dataset = get_conll_dataset(cleaned_df, labels=LABELS)
    # train_dataset = get_train_dataset(dataset, tokenizer)
    return dataset


# NOTE: not working
def evaluate_trainer(model_path: str, data_path: str):
    metric = evaluate.load("seqeval")
    custom_compute_metrics = partial(compute_metrics, label_list=LABELS, metric=metric)

    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test_data = read_df(data_path)
    test_dataset = get_dataset(test_data, tokenizer, LABEL2ID)

    trainer = Trainer(
        model,
        tokenizer=tokenizer,
        compute_metrics=custom_compute_metrics,
        eval_dataset=test_dataset,
    )

    predictions, labels, _ = trainer.predict(test_dataset)
    metrics = custom_compute_metrics((predictions, labels))
    return metrics


def evaluate_evaluator(model_path: str, test_dataset: Dataset):
    task_evaluator = evaluator("token-classification")
    selected_indices = [i for i in range(len(test_dataset))][::-1]

    # model = AutoModelForTokenClassification.from_pretrained(
    #     model_path,
    #     num_labels=len(LABEL2ID),
    #     id2label=ID2LABEL,
    #     label2id=LABEL2ID,
    # )

    tokenizer = load_tokenizer(base_model_name=model_path)
    tokenizer.model_max_length = 512

    results = task_evaluator.compute(
        model_or_pipeline=model_path,
        tokenizer=tokenizer,
        data=test_dataset.select(selected_indices),
        metric="seqeval",
    )
    return results


def cast_base_python(metrics):
    for k, v in metrics.items():
        if isinstance(v, dict):
            metrics[k] = cast_base_python(v)
        elif isinstance(v, list):
            metrics[k] = cast_base_python(v[0])
        elif isinstance(v, (np.float64, np.int64, np.float32, np.int32)):
            metrics[k] = v.item()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-p", default=SAVE_MODEL, type=str)

    args = parser.parse_args()

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    data_dict = {
        # "val": VAL_DATASET,
        "test": TEST_DATASET
    }
    dataset_dict = {
        key: get_eval_data(value, tokenizer) for key, value in data_dict.items()
    }

    metric_dict = {}

    for split, dataset_ in dataset_dict.items():
        metrics = evaluate_evaluator(model_path, dataset_)
        metrics = cast_base_python(metrics)
        metric_dict[split] = metrics

    test_exporter = Exporter(
        train_id=basename(dirname(model_path)),
        test_set_volume=len(dataset_dict["test"]),
        precision=metric_dict["test"]["overall_precision"],
        recall=metric_dict["test"]["overall_recall"],
        f1=metric_dict["test"]["overall_f1"],
        accuracy=metric_dict["test"]["overall_accuracy"],
        date=get_current_date(),
    )

    test_exporter.to_csv(join(model_path, "../test_results.csv"))
