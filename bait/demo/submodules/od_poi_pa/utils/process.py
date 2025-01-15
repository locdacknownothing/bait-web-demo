from pathlib import Path
from PIL import Image, ExifTags


def fix_orientation(filepath: str):
    image = Image.open(filepath)
    try:

        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break

        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        pass

    return image


def save_fixed_image(filepath: str, save_folder: str):
    fixed_image = fix_orientation(filepath)
    fixed_image.save(Path(save_folder) / Path(filepath).name)
