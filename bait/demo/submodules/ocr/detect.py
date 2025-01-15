from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import DeviceObjType
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

from ocr_utils.file import dirwalk


def get_masked_crop(image_path: str | Path | Image.Image, mask) -> Image:
    if isinstance(image_path, Image.Image):
        image = np.array(image_path)[:, :, ::-1].copy()
    else:
        image = cv2.imread(str(image_path))

    h, w = image.shape[:2]
    roi = [[(int(i[0]), int(i[1])) for i in mask.xy[0]]]
    points = np.array(roi)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, points, (255, 255, 255))
    res = cv2.bitwise_and(image, image, mask=mask)

    rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
    crop = res[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
    converted_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(converted_crop)


class Detector:
    def __init__(
        self,
        weight_detect: str,
        device: DeviceObjType,
        saved_folder: str | Path,
        img_size: int = 224,
        threshold: float = 0.3,
        iou: float = 0.2,
        batch_size: int | None = None,
        save: bool = False,
        save_crop: bool = False,
        save_txt: bool = False,
    ):
        self.model_detect = YOLO(weight_detect)
        self.img_size = img_size
        self.threshold = threshold
        self.iou = iou
        self.device = device
        self.batch_size = int(batch_size) if batch_size else 1
        self.saved_folder = Path(saved_folder)
        self.save_ = save
        self.save_crop = save_crop
        self.save_txt = save_txt

    def detect(
        self,
        folder_image: str | list[str],
    ) -> dict:
        if isinstance(folder_image, str):
            images = dirwalk(folder_image)

        result = []

        for i in range(0, len(images), self.batch_size):
            result_detect = self.model_detect.predict(
                images[i : i + self.batch_size],
                imgsz=self.img_size,
                conf=self.threshold,
                iou=self.iou,
                device=self.device,
                project=self.saved_folder,
                save=self.save_,
                save_crop=self.save_crop,
                save_txt=self.save_txt,
                exist_ok=True,
            )
            # result[image] = result_detect[0].boxes.xyxy
            result.extend(result_detect)

        result_dict = {key: value for key, value in zip(images, result)}
        return result_dict

    def save(self, result: dict):
        save_folder = self.saved_folder
        if not save_folder.exists():
            save_folder.mkdir()

        for key, value in result.items():
            file_path = save_folder / Path(key).name
            value.save(
                filename=file_path,
                labels=False,
                conf=False,
                line_width=1,
            )

    def save_masked_crop(self, result: dict):
        save_folder = self.saved_folder / "predict/masked_crops"
        save_folder.mkdir(parents=True, exist_ok=True)

        for image_path, results in result.items():
            image_path = Path(image_path)
            img = cv2.imread(image_path)
            height, width = img.shape[:2]

            if not results or not results.masks:
                # cv2.imwrite(saved_image_path, img)
                continue

            for i, mask in enumerate(results.masks):
                saved_image_path = (
                    save_folder / f"{image_path.stem}_{i+1}{image_path.suffix}"
                )
                if mask.xy[0].size == 0:
                    continue

                roi = [[(int(i[0]), int(i[1])) for i in mask.xy[0]]]

                mask = np.zeros((height, width), dtype=np.uint8)
                points = np.array(roi)
                cv2.fillPoly(mask, points, (255, 255, 255))
                res = cv2.bitwise_and(img, img, mask=mask)

                rect = cv2.boundingRect(points)  # returns (x,y,w,h) of the rect
                crop = res[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
                cv2.imwrite(saved_image_path, crop)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    detector = Detector(
        # weight_detect=get_weights("/mnt/data/ocr/RELEASE_0.1.0/yolov8_text.pt"),
        # weight_detect="/mnt/data/src/text_detection/yolo/runs/obb/train3/weights/best.pt",
        weight_detect="/mnt/data/src/text_detection/yolo/runs/segment/train5/weights/best.pt",
        device=device,
        saved_folder="./yolo/runs/segment/",
        threshold=0.3,
        iou=0.2,
        batch_size=64,
    )

    a = detector.detect("/mnt/data/data/Data_Ve_Chu/YOLO_2608/val")
    detector.save_masked_crop(a)
