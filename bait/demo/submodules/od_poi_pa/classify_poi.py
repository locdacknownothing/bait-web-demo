from functools import reduce
from pathlib import Path
from utils.files import get_weights, dirwalk, load_json, save_json
from ultralytics import YOLO
from shutil import copy


class POIClassification:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)
        self.id2cls = None
        self.cls2id = None

    def __call__(
        self,
        source,
        return_class_only: bool = False,
        save: bool = False,
        save_dir: str | None = None,
    ) -> list | dict:
        if isinstance(source, list):
            results_list = reduce(
                lambda x, y: x + y, [self.model(image) for image in source]
            )
        else:
            results_list = self.model(source)
        if self.id2cls is None:
            self.id2cls = results_list[0].names
            self.cls2id = {value: key for key, value in self.id2cls.items()}

        if return_class_only:
            results = [
                self.id2cls.get(results.probs.top1, "unknown_class")
                for results in results_list
            ]
        else:
            preds = [
                (results.path, self.id2cls.get(results.probs.top1, "unknown_class"))
                for results in results_list
            ]
            results = dict(preds)

            if save:
                self.save(results, save_dir=save_dir)

        return results

    def save(self, results: dict, save_dir: str | None = None):
        if save_dir is None:
            save_dir = "runs/cls/predict"

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        for image, cls in results.items():
            save_cls_dir = save_dir / str(cls)
            save_cls_dir.mkdir(exist_ok=True, parents=True)
            copy(image, save_cls_dir)


if __name__ == "__main__":
    cls_weights = get_weights("/mnt/data/POI_tiny/RELEASE_0.1.3/1010.pt")
    cls = POIClassification(cls_weights)
    # result = cls("./out_od_Image_Test/crops/POI")
    # save_json(result, "cls_poi.json")
    results = load_json("cls_poi.json")
    cls.save(results)
