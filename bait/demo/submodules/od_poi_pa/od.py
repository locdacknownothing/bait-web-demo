from os import getcwd, makedirs
from os.path import join, exists
import ntpath
from pathlib import Path
from shutil import rmtree

from ultralytics import YOLO
from custom_yolo import CustomYOLO
import torch
from ultralytics.cfg import cfg2dict
from ultralytics.engine.results import Results

from utils import (
    crop_img_from_txt,
    remove_dir,
    get_weights,
    load_object,
    save_object,
)


class OD:
    def __init__(
        self,
        des_dir_path: str,
        config_path: str,
        weight_path: str,
    ) -> None:
        self.config = cfg2dict(config_path)
        weight_detect = weight_path
        self.model_detect = CustomYOLO(weight_detect)
        self._des_dir = des_dir_path

    def detect(self, source: str) -> list:
        if not source:
            return []

        config_dict = self.config

        results = self.model_detect.predict(
            source=source,
            project=self._des_dir,
            name="./",
            exist_ok=True,
            **config_dict,
        )

        return results

    def _detect_batch_(
        self,
        images: list[str | Path],
        batch_size: int = 1,
        *args,
        **kwargs,
    ):
        start_index = 0

        for i in range(start_index, len(images), batch_size):
            batch_results = self.detect(images[i : i + batch_size])
            yield batch_results


class ODDeprecated:
    def __init__(self, weight_detect: str):
        self.model_detect = YOLO(Path(weight_detect))

    def detect(
        self,
        folder_image: str,
        conf: float,
        iou: float,
        path_out=getcwd(),
    ):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        name_in = ntpath.split(folder_image)[-1]

        path_out = join(path_out, join("out_od", name_in))
        remove_dir(path_out)

        self.model_detect.predict(
            folder_image,
            save=True,
            imgsz=1088,  # 1088
            conf=conf,
            save_txt=True,
            iou=iou,
            device=device,
            agnostic_nms=True,
            name=path_out,
            line_width=3,
            show_conf=False,
            classes=[0, 1],
            show_labels=False,
        )

        image_path_out = join(path_out, name_in)
        if exists(image_path_out):
            rmtree(image_path_out)
            makedirs(image_path_out)

        crop_img_from_txt(
            folder_image,
            join(path_out, "labels"),
            join(path_out, name_in),
        )
