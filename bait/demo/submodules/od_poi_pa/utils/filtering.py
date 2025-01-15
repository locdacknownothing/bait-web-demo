from math import ceil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from classify_poi import POIClassification


RESOLUTION_RANGES = [2048, 4024, 5568]
AREA_RATIO_THRESHOLDS = [0.001, 0.001, 0.0008, 0.0001]


def get_area_ratio_threshold(resolution):
    for i, range_value in enumerate(RESOLUTION_RANGES):
        if resolution < range_value:
            return AREA_RATIO_THRESHOLDS[i]

    return AREA_RATIO_THRESHOLDS[-1]


def box_area(box) -> float:
    x, y, w, h = box.xywh[0].tolist()
    return ceil(w) * ceil(h)


def is_invalid_area(box, min_area_threshold: float = 1500):
    detect_cls = box.cls.item()
    area = box_area(box)
    return (detect_cls == 1) and (area < min_area_threshold)


def get_poi_cls(boxes, poi_classifier, orig_img):
    # x1, y1, x2, y2 = np_boxes
    np_boxes = [box.xyxy[0] for box in boxes]
    cropped_images = [
        orig_img[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
        for box in np_boxes
    ]

    # ---TESTING: save cropped image, should use cv2 with numpy array---
    # import cv2
    # cv2.imwrite("test_orig_image.jpg", orig_img)
    # cv2.imwrite("test_cropped_image.jpg", cropped_images[0])
    # ---DONE TESTING---

    if not cropped_images:
        return []

    poi_cls = poi_classifier(cropped_images, return_class_only=True)
    return poi_cls


def filter_by_area_threshold(
    detection_result, poi_classifier, min_area_threshold: float = 1500
):
    """Filter POI boxes by area ratio to the raw image's area"""

    def filter_boxes(result):
        if not result.boxes:
            return result

        invalid_area_boxes = np.array(
            [is_invalid_area(box, min_area_threshold) for box in result.boxes]
        )
        poi_cls = get_poi_cls(result.boxes, poi_classifier, result.orig_img)
        invalid_cls_boxes = np.array(poi_cls) == "POI_normal"

        result.boxes = result.boxes[~(invalid_area_boxes & invalid_cls_boxes)]
        if result.boxes:
            return result
        else:
            return None

    filtered_results = [filter_boxes(x) for x in detection_result]
    filtered_results = [x for x in filtered_results if x]
    return filtered_results


def poi_area(image_file: str | Path) -> int:
    image_array = np.fromfile(image_file, dtype=np.uint8)
    h, w = cv2.imdecode(image_array, cv2.IMREAD_COLOR).shape[:2]
    return h * w


def filter_pois_by_area(
    pois: list[str | Path],
    poi_classifier: POIClassification,
    min_area_threshold: float = 1500,
):
    invalid_pois = [poi for poi in pois if poi_area(poi) < min_area_threshold]
    poi_cls = poi_classifier(invalid_pois, return_class_only=True)

    invalid_pois = [x for x, cls in zip(invalid_pois, poi_cls) if cls == "POI_normal"]
    valid_pois = [poi for poi in pois if poi not in invalid_pois]
    return valid_pois


def filter_labels_by_conf(gt_files, conf_threshold: float = 0.5):
    """
    Filters ground truth label files to retain only those entries that have a
    confidence score above the specified threshold.

    Args:
        gt_files (list[str]): List of file paths to ground truth label files.
        conf_threshold (float, optional): Confidence threshold for filtering.
                                          Defaults to 0.5.

    Returns:
        None
    """

    def get_conf(line):
        parts = line.strip().split(" ")
        try:
            conf = float(parts[-1])
            if not 0 <= conf <= 1:
                raise ValueError(f"Invalid confidence score: {conf}")
        except (ValueError, IndexError) as e:
            print(str(e))
            conf = -1

        return conf

    for gt_file in tqdm(gt_files):
        with open(gt_file, "r") as f:
            lines = f.readlines()
            valid_lines = [line for line in lines if get_conf(line) >= conf_threshold]

        if valid_lines:
            with open(gt_file, "w") as f:
                f.writelines(valid_lines)
        else:
            Path(gt_file).unlink()


def is_small_area_ratio(box, orig_shape, area_ratio_threshold: float = 0.001):
    # Filter dynamically based on original shape (resolution)
    """
    Determines if the area ratio of a bounding box is smaller than a given threshold.

    The function calculates the area ratio of the bounding box relative to the
    area of the original image. If the specified area_ratio_threshold is -1,
    it dynamically determines a suitable threshold based on the image resolution.

    Args:
        box: The bounding box with coordinates in the format (x1, y1, x2, y2).
        orig_shape: A tuple representing the original image shape as (height, width).
        area_ratio_threshold (float, optional): Threshold for the area ratio. If
            set to -1, a dynamic threshold is used based on the original image
            resolution. Defaults to 0.001.

    Returns:
        bool: True if the area ratio of the bounding box is smaller than the
        threshold, otherwise False.
    """
    if area_ratio_threshold == -1:
        resolution = max(orig_shape[0], orig_shape[1])
        area_ratio_threshold = get_area_ratio_threshold(resolution)
        # print(f"Area ratio threshold: {area_ratio_threshold}")

    x1, y1, x2, y2 = box.xyxy[0]
    area_ratio = (x2 - x1) * (y2 - y1) / (orig_shape[1] * orig_shape[0])
    return bool(area_ratio < area_ratio_threshold)


def filter_boxes_by_area_ratio(
    results,
    poi_classifier,
    area_ratio_threshold: float = -1,
    returns_invalid_boxes: bool = False,
):
    if not results.boxes:
        return results

    poi_masks = np.array([box.cls.item() == 1 for box in results.boxes])

    invalid_area_masks = np.array(
        [
            is_small_area_ratio(box, results.orig_shape, area_ratio_threshold)
            for box in results.boxes
        ]
    )

    poi_cls = get_poi_cls(results.boxes, poi_classifier, results.orig_img)
    invalid_cls_masks = np.array(poi_cls) == "POI_normal"
    invalid_masks = poi_masks & invalid_area_masks & invalid_cls_masks

    if returns_invalid_boxes:
        invalid_boxes = results.boxes[invalid_masks]
        results.invalid_boxes = invalid_boxes

    results.boxes = results.boxes[~invalid_masks]
    return results


def filter_by_area_ratio(results_list, poi_classifier, **kwargs):
    """
    Filters a list of detection results by area ratio, removing boxes that
    do not meet the criteria.

    This function applies area ratio filtering to each result in the results_list
    using the provided poi_classifier and additional keyword arguments. It retains
    only those results with valid boxes after filtering.

    Args:
        results_list (list): A list of detection results to be filtered.
        poi_classifier: An object used to classify points of interest in the boxes.
        **kwargs: Additional keyword arguments to be passed to the filtering function.

    Returns:
        list: A filtered list of detection results with valid boxes.
    """
    results_list = [
        filter_boxes_by_area_ratio(x, poi_classifier, **kwargs) for x in results_list
    ]
    results_list = [x for x in results_list if x.boxes is not None]
    return results_list
