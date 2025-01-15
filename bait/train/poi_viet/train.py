from ultralytics import YOLO
import os

model = YOLO("yolov10x")
config_data_file = "./config_data.yaml"

# train the model
model.train(
    data=config_data_file,
    epochs=200,
    imgsz=1088,
    close_mosaic=30,
    cls=0.8,
    batch=1, # 4
    project="./train_POI_VN_yolov10/runs/POI_VN",
)

# validate the model
metrics = model.val(data=config_data_file, split="test")
