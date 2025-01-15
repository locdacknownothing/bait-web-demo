from ultralytics import YOLO


def train(model, config_data_file):
    # train the model
    model.train(
        data=config_data_file,
        epochs=500,
        imgsz=224,
        close_mosaic=30,
        cls=0.8,
        batch=64,  # 4
        patience=0,  # disable early stopping
        device=[0, 1],
    )


def val(model, config_data_file):
    # validate the model
    metrics = model.val(data=config_data_file, split="val")
    result = {
        "map": metrics.box.map,
        "map50": metrics.box.map50,
        "map75": metrics.box.map75,
        "maps": metrics.box.maps,
    }

    print(result)


if __name__ == "__main__":
    model = YOLO("yolov8x-obb.pt")
    config_data_file = "./config_data.yaml"
    train(model, config_data_file)
    val(model, config_data_file)
