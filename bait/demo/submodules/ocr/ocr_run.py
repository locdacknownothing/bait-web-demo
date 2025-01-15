from PIL import Image
from os.path import isdir, abspath, dirname
import sys

sys.path.append(dirname(__file__))

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from ocr_utils.file import dirwalk, get_weights
from recognition_result import RegRes

import warnings

warnings.filterwarnings("ignore")


class OCR:
    def __init__(
        self,
        weight_reg: str,
        config_reg: str,
        weight_detect: str = None,
        img_size: int = None,
        threshold: float = None,
        iou: float = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ## config model detect
        if weight_detect is not None:
            self.model_detect = YOLO(weight_detect)
            self.img_size = img_size
            self.threshold = threshold
            self.iou = iou

        ## config model reg
        self.config = Cfg.load_config_from_file(config_reg)
        self.config["device"] = self.device
        self.config["weights"] = weight_reg
        self.detector = Predictor(self.config)

    def detect(
        self,
        images: str | list,
    ):
        if type(images) not in [str, list]:
            raise TypeError("Invalid images data type, should be str or list.")

        if isinstance(images, str):
            if isdir(images):
                images = dirwalk(images)
            else:
                images = [images]

        result = {}
        for image in images:

            result_detect = self.model_detect(
                image,
                imgsz=self.img_size,
                conf=self.threshold,
                iou=self.iou,
                device=self.device,
            )
            image = abspath(str(image))
            img = Image.open(image)
            result[image] = []

            for box in result_detect[0].boxes:
                assert box.xyxy.shape[0] == 1 and box.conf.shape[0] == 1
                i = box.xyxy[0].tolist()
                crop = img.crop([int(i[0]), int(i[1]), int(i[2]), int(i[3])])
                label = self.detector.predict(crop)
                conf = box.conf[0].item()

                result[image].append(RegRes(i[:4], label, conf))

        return result

    def recognize(self, detection_results: list[Results]):
        if type(detection_results) is not list:
            raise TypeError("Invalid detection_results data type, should be list.")

        for results in detection_results:
            if type(results) is not Results:
                raise TypeError(
                    "Invalid results data type, should be ultralytics.engine.results.Results."
                )

        result_dict = {}

        for results in detection_results:
            path = results.path
            image = Image.open(path)
            result_dict[path] = []

            for box in results.boxes:
                assert box.xyxy.shape[0] == 1 and box.conf.shape[0] == 1
                i = box.xyxy[0].tolist()
                crop = image.crop([int(i[0]), int(i[1]), int(i[2]), int(i[3])])
                label = self.detector.predict(crop)
                conf = box.conf[0].item()

                result_dict[path].append(RegRes(i[:4], label, conf))

        return result_dict


if __name__ == "__main__":
    from ocr_utils.extractor import extract_string
    from ocr_utils.file import (
        load_object,
        save_json,
    )

    ocr = OCR(
        weight_reg=get_weights("/mnt/data/ocr/v3.0.0/transformerocr_2911.pth"),
        config_reg=get_weights("/mnt/data/ocr/v3.0.0/config_transformer.yml"),
    )

    detection_results = load_object(
        "/mnt/data/src/text_detection/data/test_results.pkl"
    )
    recognition_results = ocr.recognize(detection_results)

    string_dict = extract_string(
        recognition_results, is_sorted=True, is_lowercase=True, apply_unidecode=True
    )
    save_json(string_dict, "/mnt/data/src/ocr/data/test_string_dict.json")
