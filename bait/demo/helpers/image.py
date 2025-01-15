from pathlib import Path
from PIL import Image
from io import BytesIO
import base64


def image_path_to_base64(image_path: str | Path):
    if image_path:
        img = Image.open(image_path)
        with BytesIO() as buffer:
            img.save(buffer, "png")
            raw_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{raw_base64}"


def read_image_from_bytes(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    return image


def check_image_resolution(
    image: Image.Image, 
    resolution_threshold: tuple[int, int] = (1440, 1080)
):
    width, height = image.size
    return width >= resolution_threshold[0] and height >= resolution_threshold[1]
