from pathlib import Path
import yaml

from ultralytics import YOLO

from export import Exporter, get_current_date
from file import get_weights


def val(model, config_data_file, save_dir_name=""):
    """
    Validate the model.

    Parameters
    ----------
    model : ultralytics.YOLO
        The model to be validated.
    config_data_file : str
        The path to the data configuration file.

    Returns
    -------
    result : dict
        A dictionary containing the validation results.
    """
    results = model.val(
        data=config_data_file,
        split="test",
        project="./runs/segment/val",
        name=save_dir_name,
    )
    return results


def get_metrics_result(results):
    try:
        metric_results = {
            "box_precision": results.box.mp,
            "box_recall": results.box.mr,
            "box_map50": results.box.map50,
            "box_map": results.box.map,
            "mask_precision": results.seg.mp,
            "mask_recall": results.seg.mr,
            "mask_map50": results.seg.map50,
            "mask_map": results.seg.map,
        }
    except AttributeError:
        metric_results = {
            "box_precision": results.box.mp,
            "box_recall": results.box.mr,
            "box_map50": results.box.map50,
            "box_map": results.box.map,
            "mask_precision": None,
            "mask_recall": None,
            "mask_map50": None,
            "mask_map": None,
        }
    return metric_results


if __name__ == "__main__":
    name = "yolo11x_1612"
    best_weight = "/mnt/data/src/text_detection/weights/yolo11x_1612.pt"
    model = YOLO(best_weight)

    config_data_file = "./config_data.yaml"
    with open(config_data_file, "r") as f:
        data_config = yaml.safe_load(f)

    test_folder = Path(data_config["path"]) / data_config["test"]
    test_dataset = list(test_folder.glob("*.jpg"))
    results = val(model, config_data_file, name)

    test_exporter = Exporter(
        train_id=results.save_dir.name,
        test_set_volume=len(test_dataset),
        date=get_current_date(),
        **get_metrics_result(results),
    )
    test_exporter.to_csv(results.save_dir / "test_results.csv")
    test_exporter.to_json(results.save_dir / "test_results.json")
