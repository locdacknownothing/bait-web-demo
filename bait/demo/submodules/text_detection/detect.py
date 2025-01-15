from os.path import dirname, join
from sys import path

path.append(dirname(__file__))

from pathlib import Path

import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
from ultralytics.cfg import cfg2dict

import warnings

warnings.filterwarnings("ignore")

from utils_td.file import dirwalk, get_weights


class Detector:
    def __init__(
        self,
        weight_detect: str = None,
        img_size: int = None,
        conf: float = None,
        iou: float = None,
        saved_folder: str | Path = None,
    ):
        args = cfg2dict(join(dirname(__file__), "./results/args.yaml"))

        if weight_detect is None:
            weight_server, weight_path = str(args["weight_detect"]).split(":")[:2]
            out_dir = join(dirname(__file__), "./weights")
            weight_detect = get_weights(weight_path, weight_server, out_dir)

        self.model_detect = YOLO(weight_detect)
        self.img_size = img_size if img_size else int(args["img_size"])
        self.conf = conf if conf else float(args["conf"])
        self.iou = iou if iou else float(args["iou"])
        self.saved_folder = Path(saved_folder) if saved_folder else None

    def detect(
        self,
        source: str | list[str],
        batch_size: int = 1,
        save: bool = False,
        save_crop: bool = False,
        save_txt: bool = False,
    ) -> dict:
        images = self._source_to_images_(source)
        results = []

        for i in range(0, len(images), int(batch_size)):
            result_detect = self.model_detect.predict(
                images[i : i + int(batch_size)],
                imgsz=self.img_size,
                conf=self.conf,
                iou=self.iou,
                project=self.saved_folder,
                name="./",
                exist_ok=True,
                save=save,
                save_crop=save_crop,
                save_txt=save_txt,
            )

            results.extend(result_detect)

        return results

    def _source_to_images_(self, source: str | Path | np.ndarray | list):
        images = []

        if isinstance(source, (str, Path)):
            if Path(source).is_file():
                images = [source]
            elif Path(source).is_dir():
                images = dirwalk(source)
            else:
                raise ValueError(f"Invalid source: {source}")
        elif isinstance(source, (np.ndarray, Image.Image)):
            images = [source]
        elif isinstance(source, list):
            images = source
        else:
            raise ValueError(f"Invalid source: {source}")

        return images

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
        save_folder = self.saved_folder / "masked_crops"
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

                # box = box.xyxy[0].tolist()
                # crop = res[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

                cv2.imwrite(saved_image_path, crop)


if __name__ == "__main__":
    detector = Detector(
        weight_detect=get_weights("/mnt/data/text_detection/v3.0.0/yolo11x_1612.pt"),
        # saved_folder=join(dirname(__file__), "./test_detect"),
    )

    results = detector.detect("data/test_imgs")
    # print(results)

    # (Optional) Save for later use
    import pickle

    with open("data/test_results.pkl", "wb") as f:
        pickle.dump(results, f)
