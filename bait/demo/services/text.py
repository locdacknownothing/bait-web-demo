from dataclasses import asdict
import re
import pandas as pd
import torch

from submodules.text_detection.detect import Detector
from submodules.text_detection.utils_td.file import get_weights
from submodules.ocr.ocr_run import OCR
from submodules.ocr.ocr_utils.extractor import extract_string
from submodules.ner.ner import NER
from submodules.ner.file import get_model
from submodules.cate_suggestion.predict import Model_Cate


# text_detection_weights = get_weights("/mnt/data/text_detection/v3.0.0/yolo11x_1612.pt")
# text_recognition_weights = get_weights("/mnt/data/ocr/v3.0.0/transformerocr_2911.pth")
# text_recognition_config = get_weights("/mnt/data/ocr/v3.0.0/config_transformer.yml")
# ner_model = get_model("/mnt/data/ner/v1.0.0/model_2511")
# cate_model = get_model("/mnt/data/cate_suggestion/DT_2.0.0/checkpoint-1254600-231224")
# cate_label_list = get_model("/mnt/data/cate_suggestion/DT_2.0.0/dgm_label.json")
# cate_map_name = get_model("/mnt/data/cate_suggestion/DT_2.0.0/list_cate.txt")

# Building Information 
text_detection_weights = "./weights/yolo11x_1612.pt"
text_recognition_weights = "./weights/transformerocr_0701.pth"
text_recognition_config = "./weights/config_transformer.yml"
ner_model = "./weights/model_2511"
cate_model = "./weights/checkpoint-1254600-231224"
cate_label_list = "./weights/dgm_label.json"
cate_map_name = "./weights/list_cate.txt"
cate_config = "submodules/cate_suggestion/configs/default.yaml"

# Traffic Sign
text_detection_weights_traffic = "./weight_traffic/text_seg_sign.pt"
text_recognition_weights_traffic = "./weight_traffic/transformerocr_sign_1401.pth"
text_recognition_config_traffic = "./weights/config_transformer.yml"



def ocr(image: str) -> dict[str, str]:
    detector = Detector(weight_detect=text_detection_weights, conf=0.1)
    ocr = OCR(
        weight_reg=text_recognition_weights,
        config_reg=text_recognition_config_traffic,
    )

    detection_results = detector.detect(image)
    recognition_results = ocr.recognize(detection_results)
    string_dict = extract_string(recognition_results, is_sorted=True)
    return string_dict

def ocr_sign_traffic(image) -> dict[str, str]:
    detector = Detector(weight_detect=text_detection_weights_traffic, conf=0.1,img_size=64)
    ocr = OCR(
        weight_reg=text_recognition_weights_traffic,
        config_reg=text_recognition_config,
    )
    
    detection_results = detector.detect(image)
    recognition_results = ocr.recognize(detection_results)
    string_dict = extract_string(recognition_results, is_sorted=False)

    return string_dict


def ner(text: str) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        ner = NER(ner_model, device=device)
        ner_results = ner(text)[0]
        ner_dict = asdict(ner_results)
    except RuntimeError:
        ner_dict = {"name": [], "keyword": [], "address": []}

    ner_dict["name"] = [] if not ner_dict.get("name") else ner_dict["name"]
    ner_dict["keyword"] = [] if not ner_dict.get("keyword") else ner_dict["keyword"]
    ner_dict["address"] = [] if not ner_dict.get("address") else ner_dict["address"]

    return ner_dict


def cate(text: str) -> list:
    try:
        cate = Model_Cate(
            cate_model,
            cate_label_list,
            cate_map_name,
            cate_config,
        )
        cate_results,prob= cate.predict(text)
        d = {'Cate': cate_results , 'Probs': prob}
        df = pd.DataFrame(data=d)
        return df
    except:
        return None


def get_address_number(address: str) -> str:
    pattern = r"^[1-9]\d*(?: ?(?:[a-z]|[/-] ?\d+[a-z]?))?"

    address = str(address)
    address_number = re.match(pattern, address)
    # print(address_number.group(0))
    return address_number


def get_poi_data(image: str) -> dict:
    image = str(image)
    string_dict = ocr(image)
    text = string_dict[image]

    ner_results = ner(text)
    # print(ner_results)
    ner_text = " ".join(ner_results["name"]) + " ".join(ner_results["keyword"])
    cate_results = cate(text) ## lấy full text từ ocr

    output = {}

    output["ocr"] = text
    output["ner"] = ner_results

    if ner_results["name"]:
        output["name"] = max(ner_results["name"], key=len)

    if cate_results is not None:
        
        output["cate"] = cate_results

    if ner_results["address"]:
        output["address"] = ner_results["address"][0]

        for address in ner_results["address"]:
            address_number = get_address_number(address)
            if address_number:
                output["address_number"] = address_number.group(0)
                break

    # print(output)
    return output

def get_traffic_sign_data(image) -> dict:
    image = str(image)
    string_dict = ocr_sign_traffic(image)
    text = string_dict[image]
    return text
