from ultralytics import YOLO

model = YOLO("yolov10x.yaml")


results = model.train(
    data="config_data.yaml",
    epochs=200,
    imgsz=1088,
    name="POI_Thai_yolov10",
    close_mosaic=30,
    batch=1, # 4
    cls=0.8,
)
metrics = model.val(split="test")
# print(metrics.box)
