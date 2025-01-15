from pathlib import Path
from PIL import Image, ImageDraw
from tqdm import tqdm


def get_files_by_suffix(
    input_folder: str | Path, suffix: str | list[str]
) -> list[Path]:
    input_folder = Path(input_folder)

    if isinstance(suffix, str):
        suffix = [suffix]

    files = [x for x in input_folder.iterdir() if x.suffix in suffix]
    return files


def get_annotation_from_text_file(text_path: str | Path) -> list:
    data = []

    with open(text_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        data.append(parts)

    return data


def get_bounding_box_coordinates(w, h, x_n, y_n, w_n, h_n) -> list:
    x_center = x_n * w
    y_center = y_n * h
    width = w_n * w
    height = h_n * h
    x_min = int(x_center - width / 2)
    x_max = int(x_center + width / 2)
    y_min = int(y_center - height / 2)
    y_max = int(y_center + height / 2)

    return [x_min, y_min, x_max, y_max]


def main():
    input_folder = Path("/mnt/data/data/predict_change_POIVN")
    save_folder = input_folder / "annotations"
    save_folder.mkdir(parents=True, exist_ok=True)
    files = get_files_by_suffix(input_folder, ".txt")

    for text_file_path in tqdm(files):
        image_file_path = text_file_path.parent / Path(text_file_path.stem + ".jpg")

        if not image_file_path.exists():
            continue

        image = Image.open(image_file_path).copy()
        draw = ImageDraw.Draw(image)
        w, h = image.size

        annotations = [x[:5] for x in get_annotation_from_text_file(text_file_path)]

        count = 0
        for annotation in annotations:
            label, x_n, y_n, w_n, h_n = [float(x) for x in annotation]
            if w_n > 1 or h_n > 1:
                continue

            if label != 1:
                continue

            draw.rectangle(
                get_bounding_box_coordinates(w, h, x_n, y_n, w_n, h_n),
                outline="blue",
                width=2,
            )
            count += 1

        save_image_path = save_folder / image_file_path.name
        if count and not save_image_path.exists():
            image.save(save_image_path)


if __name__ == "__main__":
    main()
