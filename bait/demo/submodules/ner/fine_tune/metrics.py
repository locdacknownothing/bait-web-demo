import numpy as np


def compute_metrics(eval_preds, label_list, metric):
    pred_logits, labels = eval_preds
    pred_logits = np.argmax(pred_logits, axis=2)

    predictions = [
        [
            label_list[eval_preds]
            for prediction, label in zip(pred_logits, labels)
            for (eval_preds, l) in zip(prediction, label)
            if l != -100
        ]
    ]

    true_labels = [
        [
            label_list[l]
            for prediction, label in zip(pred_logits, labels)
            for (eval_preds, l) in zip(prediction, label)
            if l != -100
        ]
    ]

    results = metric.compute(predictions=predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
