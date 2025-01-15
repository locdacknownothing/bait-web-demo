import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from ocr.recognition_result import RegRes
from ocr.ocr_seg import get_weights
from ocr.detect import get_masked_crop
from ocr.ocr_utils.extractor import sort_by_bounding_boxes


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_detect = get_weights("/mnt/data/text_detection/yolo_seg/yolov9e-seg-2409.pt")
weight_reg = get_weights("/mnt/data/ocr/RELEASE_0.1.1/weight_transformer_2008.pth")
config_reg = get_weights("/mnt/data/ocr/RELEASE_0.1.1/config_transformer.yml")

detect_model = YOLO(weight_detect)
img_size = 224
threshold = 0.25
iou = 0.2

config = Cfg.load_config_from_file(config_reg)
config["device"] = device
config["weights"] = weight_reg
detector = Predictor(config)


def process_image(image):
    detect_results = detect_model(
        image,
        imgsz=img_size,
        conf=threshold,
        iou=iou,
        device=device,
    )

    results = []

    try:
        for box, mask in zip(detect_results[0].boxes, detect_results[0].masks):
            crop = get_masked_crop(image, mask)
            label = detector.predict(crop)
            xyxy = box.xyxy[0].tolist()[:4]
            conf = box.conf[0].item()
            results.append(RegRes(xyxy, label, conf))
    except (TypeError, cv2.error):
        pass

    sorted_results = [
        res for row_res in sort_by_bounding_boxes(results) for res in row_res
    ]
    sorted_text = [res.text for res in sorted_results]
    sorted_string = " ".join(sorted_text)

    return sorted_string


title = "Interactive demo: TrOCR"
description = "Demo for Microsoft's TrOCR, an encoder-decoder model consisting of an image Transformer encoder and a text Transformer decoder for state-of-the-art optical character recognition (OCR) on single-text line images. This particular model is fine-tuned on IAM, a dataset of annotated handwritten images. To use it, simply upload an image or use the example image below and click 'submit'. Results will show up in a few seconds."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2109.10282'>TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models</a> | <a href='https://github.com/microsoft/unilm/tree/master/trocr'>Github Repo</a></p>"
examples = [["/mnt/data/src/dedup/final_seg/.0009_POI_485/bqKOY_jJ5f.jpg"]]

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title=title,
    description=description,
    article=article,
    examples=examples,
)
iface.launch(debug=True, share=True)
