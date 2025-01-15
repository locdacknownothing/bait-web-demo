from os import makedirs
from os.path import basename, join
from pathlib import Path

from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results


class CustomYOLO(YOLO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, *args, **kwargs):
        # custom_kwargs = kwargs.pop("custom_kwargs", {})

        save_ = kwargs.get("save", False)
        save_txt_ = kwargs.get("save_txt", False)
        save_crop_ = kwargs.get("save_crop", False)

        # suppress saving to apply filter
        kwargs["save"] = False
        kwargs["save_txt"] = False
        kwargs["save_crop"] = False

        results = super().predict(*args, **kwargs)
        # from time import time

        # runtime = 0

        # if filter:
        #     save_invalid_boxes = custom_kwargs.get("save_invalid_boxes", False)
        #     start_time = time()
        #     results = filter_by_area_ratio(
        #         results,
        #         poi_classifier=poi_classifier,
        #         returns_invalid_boxes=save_invalid_boxes,
        #     )
        #     runtime = time() - start_time

        if save_txt_:
            self.save_txt_(results, *args, **kwargs)

        if save_:
            self.save_(results, *args, **kwargs)

        if save_txt_:
            self.save_txt_(results, *args, **kwargs)

        if save_crop_:
            self.save_crop_(results, *args, **kwargs)

        return results

    def save_(self, results: list[Results], *args, **kwargs):
        for result in results:
            path = Path(result.path)
            save_dir = Path(result.save_dir)

            save_dir = save_dir / "annotations"
            save_dir.mkdir(exist_ok=True, parents=True)

            result.save(
                filename=save_dir / path.name,
                boxes=kwargs["show_boxes"],
                conf=kwargs["show_conf"],
                labels=kwargs["show_labels"],
                line_width=kwargs["line_width"],
            )

        return results

    def save_txt_(self, results: list[Results], *args, **kwargs):
        for result in results:
            image_path = Path(result.path)
            save_dir = Path(result.save_dir)

            if self.save_txt_:
                save_txt_ = save_dir / "labels"
                save_txt_.mkdir(exist_ok=True, parents=True)
                txt_file = save_txt_ / f"{image_path.stem}.txt"
                result.save_txt(txt_file=txt_file, save_conf=kwargs["save_conf"])

    def save_crop_(self, results: list[Results], *args, **kwargs):
        for result in results:
            path = Path(result.path)
            orig_img = Image.open(path)
            save_dir = Path(result.save_dir)

            for i, detection in enumerate(result.boxes):
                label_name = result.names[int(detection.cls)]
                class_dir = save_dir / "crops" / label_name
                class_dir.mkdir(parents=True, exist_ok=True)

                x1, y1, x2, y2 = map(int, detection.xyxy[0])
                crop = orig_img.crop((x1, y1, x2, y2))
                crop_path = class_dir / f"{path.stem}_{i}.jpg"
                crop.save(crop_path)

            # if kwargs.get("save_invalid_boxes", False) and hasattr(results, "invalid_boxes"):
            #     for i, detection in enumerate(results.invalid_boxes):
            #         label_name = "invalid"
            #         class_dir = save_dir / "crops" / label_name
            #         class_dir.mkdir(parents=True, exist_ok=True)

            #         x1, y1, x2, y2 = map(int, detection.xyxy[0])
            #         crop = orig_img.crop((x1, y1, x2, y2))
            #         crop_path = class_dir / f"{path.stem}_{i}.jpg"
            #         crop.save(crop_path)
