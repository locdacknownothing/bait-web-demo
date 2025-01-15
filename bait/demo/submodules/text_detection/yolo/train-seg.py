from os import environ
from os.path import dirname

import sys

sys.path.append(dirname(__file__))

from pathlib import Path
from time import time
import yaml
from ultralytics import YOLO

from export import (
    Exporter,
    get_current_date,
    get_gpu_utilization,
    human_readable_time,
)
from file import get_weights

environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def train(model, config_data_file):
    """
    Train the model.

    Args:
        model (YOLO): The YOLO model object.
        config_data_file (str): The path to the data configuration file.
    """
    results = model.train(
        data=config_data_file,
        project="./runs/segment",
        epochs=500,
        imgsz=640,
        batch=8,
        workers=8,
        patience=100,
        device=0,
        cos_lr=True,
        # resume=True,
    )

    return results


if __name__ == "__main__":
    # model = YOLO("yolo11x-seg.pt")  # base pretrained model
    # best_weight = get_weights(
    #     "/mnt/data/text_detection/yolo_seg/imgsz_640/yolo11x-0111.pt"
    # )
    # model = YOLO(best_weight)
    last_weight = (
        "/home/dgm_bait_02/text_detection/yolo/runs/segment/train6/weights/last.pt"
    )
    model = YOLO(last_weight)

    config_data_file = "./config_data.yaml"
    with open(config_data_file, "r") as f:
        data_config = yaml.safe_load(f)

    train_folder = Path(data_config["path"]) / data_config["train"]
    val_folder = Path(data_config["path"]) / data_config["val"]

    train_dataset = list(train_folder.glob("*.jpg"))
    val_dataset = list(val_folder.glob("*.jpg"))

    # Start measuring train time
    start_train_time = time()

    results = train(model, config_data_file)

    end_train_time = time()
    train_time = human_readable_time(end_train_time - start_train_time)

    metric_dict = {key: f"{value:.4f}" for key, value in results.results_dict.items()}
    gpu_usage = get_gpu_utilization()

    train_exporter = Exporter(
        train_id=results.save_dir.name,
        train_set_volume=len(train_dataset),
        val_set_volume=len(val_dataset),
        total_volume=len(train_dataset) + len(val_dataset),
        train_time=train_time,
        gpu_usage=gpu_usage,
        date=get_current_date(),
        **metric_dict,
    )

    train_exporter.to_csv(results.save_dir / "train_results.csv")
