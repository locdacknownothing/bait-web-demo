import cv2
import numpy as np
from pathlib import Path
import requests
from subprocess import run, CalledProcessError
from time import sleep
from tqdm import tqdm
from urllib.parse import urlparse


def get_validate_command(path: str | Path) -> str:
    """
    Get the command to validate a directory or file.

    Args:
        path (str | Path): The path to the directory or file to validate.

    Returns:
        str: The command to run to validate the directory or file.
    """
    abs_path = str(Path(path).resolve())
    command = "find {} -type f -size +1c -size -10M -print0 | xargs -0 file --mime-type | grep -E 'image'".format(
        abs_path
    )
    return command


def is_url(string: str) -> bool:
    try:
        result = urlparse(str(string))
        return all(
            [result.scheme, result.netloc]
        )  # Check if scheme and netloc are present
    except ValueError:
        return False


def validate_url(url: str) -> str | None:
    response = requests.get(url)

    if response.status_code == 200:
        try:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            del image
            del image_array
        except cv2.error:
            url = None
        finally:
            return url
    else:
        return None


def is_file(path: str | Path) -> bool:
    if not Path(path).exists():
        return False
    else:
        return Path(path).is_file()


def validate_file(
    file: str | Path, max_retries: int = 3, wait_time: int = 5
) -> str | Path | None:
    retries = 0

    while True:
        try:
            command = get_validate_command(file)
            result = run(command, shell=True, capture_output=True, text=True)
            valid_lines = result.stdout.strip()
            return file if valid_lines else None
        except CalledProcessError as e:
            print(str(e))
            retries = retries + 1
            if retries >= max_retries:
                raise Exception(
                    "Failed to validate file after {} retries".format(max_retries)
                )
            else:
                print(
                    "Calling process failed. Retry in {} seconds (retries left: {})".format(
                        wait_time, max_retries - retries
                    )
                )
                sleep(wait_time)


def is_dir(path: str | Path) -> bool:
    if not Path(path).exists():
        return False
    else:
        return Path(path).is_dir()


def validate_dir(dir: str | Path, max_retries: int = 3, wait_time: int = 5) -> list:
    retries = 0

    while True:
        try:
            command = get_validate_command(dir)
            result = run(command, shell=True, capture_output=True, text=True)
            output = result.stdout.strip()
            if not output:
                return []

            valid_paths = [line.split(" ")[0][:-1] for line in output.split("\n")]
            return valid_paths
        except CalledProcessError as e:
            print(str(e))
            retries = retries + 1
            if retries >= max_retries:
                raise Exception(
                    "Failed to validate directory after {} retries".format(max_retries)
                )
            else:
                print(
                    "Calling process failed. Retry in {} seconds (retries left: {})".format(
                        wait_time, max_retries - retries
                    )
                )
                sleep(wait_time)


def validate_source(source: str | Path | list):
    """
    Validate source, which can be a file, directory, or url.

    Args:
        source (str | Path | list): The source to validate.

    Returns:
        list: A list of valid paths.

    Raises:
        ValueError: If the source is invalid.
    """
    if isinstance(source, (str, Path)):
        if is_url(source):
            return [validate_url(source)]
        elif is_file(source):
            return [validate_file(source)]
        elif is_dir(source):
            return validate_dir(source)
        else:
            raise ValueError("Invalid source: {}".format(source))
    elif isinstance(source, list):
        images = [validate_source(image) for image in tqdm(source)]
        images = [image for image in images if image is not None]
        return images
    else:
        raise ValueError("Invalid source: {}".format(source))
