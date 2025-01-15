from os.path import dirname, join
import sys

sys.path.append(join(dirname(__file__), "../"))

import evaluate
from evaluate import evaluator
from datasets import ClassLabel, Dataset, Features, Sequence, Value

from train_phoNER import *


def format_data(data):
    tokens = [x["words"] for x in data]
    ner_tags = [x["tags"] for x in data]
    ner_tags = [[LABEL2ID[tag] for tag in tags] for tags in ner_tags]

    dataset = Dataset.from_dict(
        mapping={
            "tokens": tokens,
            "ner_tags": ner_tags,
        },
        features=Features(
            {
                "tokens": Sequence(feature=Value(dtype="string")),
                "ner_tags": Sequence(feature=ClassLabel(names=LABELS)),
            }
        ),
    )

    return dataset


def evaluate_evaluator():
    task_evaluator = evaluator("token-classification")
    raw_test_data = extract_data(
        "/mnt/data/src/ner/data/phoNER_COVID19/test_syllable.json"
    )
    test_data = format_data(raw_test_data)

    # BUG: list out of index
    results = task_evaluator.compute(
        model_or_pipeline="/mnt/data/src/ner/fine_tune/train_phoNER2/model4",
        data=test_data,
        metric="seqeval",
    )
    print(results)


def evaluate_trainer():
    metric = evaluate.load("seqeval")
    custom_compute_metrics = partial(compute_metrics, label_list=LABELS, metric=metric)

    model_path = "/mnt/data/src/ner/fine_tune/train_phoNER2/model3"
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test_data = extract_data("/mnt/data/src/ner/data/phoNER_COVID19/test_syllable.json")
    tokenized_test_data = tokenize_and_align_labels(test_data, tokenizer)
    test_dataset = Dataset.from_dict(tokenized_test_data.data)

    trainer = Trainer(
        model,
        tokenizer=tokenizer,
        compute_metrics=custom_compute_metrics,
        eval_dataset=test_dataset,
    )

    predictions, labels, _ = trainer.predict(test_dataset)
    metrics = custom_compute_metrics((predictions, labels))
    print(metrics)

    # results = trainer.evaluate()
    # print(results)


if __name__ == "__main__":
    evaluate_trainer()
