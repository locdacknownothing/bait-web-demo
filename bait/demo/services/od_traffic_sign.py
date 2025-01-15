from pathlib import Path
from ultralytics import YOLO
from ultralytics.cfg import cfg2dict

from PIL import Image
import pandas as pd
from submodules.od_poi_pa.od import OD
from services.text import get_traffic_sign_data




def od_traffic(image_path: str | Path, output_dir: str | Path):

    ## thay đổi thành json sau
    list_label = cfg2dict("/mnt/data/src/bait/demo/configs_traffic/list_sign.yaml")

    
    with open("/mnt/data/src/bait/demo/configs_traffic/list_sign_ocr.txt", "r+", encoding="utf8") as f:
        data_ocr = f.readlines()

    
    list_label_ocr = {}
    for i in data_ocr:
        id, name = i.split("\t")
        list_label_ocr[id] = name

    #######

    config_path = "./configs_traffic/od_traffic_sign.yaml"
    weight_path = "/mnt/data/src/bait/demo/weight_traffic/od_traffic.pt"
    od_model = OD(output_dir, config_path, weight_path)
    od_results = od_model.detect(image_path)[0]

    model_cls = YOLO(
        "/mnt/data/src/model/yolo/runs/Data_Traffic_Sign_Classify3/weights/best.pt"
    )  # load a custom model

    cropped_image_path = [path for path in Path(output_dir / "crops").rglob("*.*")]
    detections = []

    for i in cropped_image_path:
        results = model_cls(i, imgsz=224)
        label = results[0].names


        ocr = ""
        if str(label[results[0].probs.top1]) in list_label_ocr:
            ocr = get_traffic_sign_data(i)
        

       
        detection = {
            "path": od_results.path,
            "label": str(list_label[label[results[0].probs.top1]]),
            "ocr": ocr
        }
     
        detections.append(detection)

    return detections


# Predict with the model

def get_annotated_image_traffic(image_path, output_dir):
    annotated_image_path = Path(output_dir / "annotations") / image_path.name
    annotated_image = Image.open(annotated_image_path)
    return annotated_image


def get_detections_data_traffic(detections, output_dir):
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
                "Detail": det["ocr"],
                "View": [p for p in cropped_image_paths if stem in p.stem][0],
            }
        )

    return pd.DataFrame(table_data)
