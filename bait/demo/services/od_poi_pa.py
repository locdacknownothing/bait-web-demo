from pathlib import Path
from PIL import Image
import pandas as pd
from submodules.od_poi_pa.od import OD
# from submodules.od_poi_pa.utils import get_weights


def od(image_path: str | Path, output_dir: str | Path):
    config_path = "./configs/od_poi_pa.yaml"
    # weight_path = get_weights("/mnt/data/od/v2.0.0/yolov9_od_2507.pt")
    weight_path = "./weights/yolov9_od_2507.pt"
    od_model = OD(output_dir, config_path, weight_path)
    od_results = od_model.detect(image_path)[0]

    # weight_poi_cls = get_weights("/mnt/data/od/v2.0.0/poi_cls_1510_64.pt")
    # poi_classifier = POIClassification(weight_path=weight_poi_cls)
    # od_results = filter_by_area_ratio(od_results, poi_classifier)[0]

    # poi_detections = []
    # pa_detections = []
    detections = []

    for box in od_results.boxes.cpu().numpy():
        label = od_results.names[int(box.cls[0].item())]
        detection = {
            "path": od_results.path,
            "xywh": box.xywh[0].tolist(),
            "xyxy": box.xyxy[0].tolist(),
            "label": label,
            "conf": box.conf[0].item(),
        }
        detections.append(detection)
        
    return detections

        # if label == "POI":
        #     poi_detections.append(detection)
        # elif label == "PA":
        #     pa_detections.append(detection)

        # max_poi_detection = max(poi_detections, key=_area_)

    # return poi_detections + pa_detections


def _area_(detection):
    x, y, w, h = detection["xywh"]
    return w * h


def get_annotated_image(image_path, output_dir):
    annotated_image_path = Path(output_dir / "annotations") / image_path.name
    annotated_image = Image.open(annotated_image_path)
    return annotated_image


def get_detections_data(detections, output_dir):
    """
    Processes detection data and returns a pandas DataFrame with relevant information.

    Args:
        detections (list): A list of detection dictionaries, where each dictionary contains 
                           information about a detected object such as its path, label, and confidence.
        output_dir (str or Path): The directory path where cropped images are stored.

    Returns:
        pd.DataFrame: A DataFrame containing processed detection data, including an identifier,
                      label, confidence score, and path to the cropped image view.
                      Returns None if no detections are provided.
    """

    if not detections:
        return None 
    
    cropped_image_paths = [path for path in Path(output_dir / "crops").rglob("*.*")]
    table_data = []
    for i, det in enumerate(detections):
        orig_stem = Path(det["path"]).stem
        stem = f"{orig_stem}_{int(i)}"
        table_data.append(
            {
                "Id": i,  # For identifying the row
                "Label": det["label"],
                "Conf": det["conf"],
                "View": [p for p in cropped_image_paths if stem in p.stem][0],
            }
        )

    return pd.DataFrame(table_data)
