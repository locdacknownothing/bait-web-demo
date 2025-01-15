from ultralytics import YOLO

model = YOLO("/home/dgm_bait_i9/model/yolo/runs/detect/POI_VN4/weights/best.pt")

model.predict(
    "/home/dgm_bait_i9/data/7PG7WPMG",
    save=True,
    imgsz=640,
    conf=0.3,
    save_txt=True,
    iou=0.5,
)
