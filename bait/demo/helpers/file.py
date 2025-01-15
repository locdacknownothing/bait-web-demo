from pathlib import Path
from shutil import rmtree


def create_temp_dir(path: str | Path = Path.cwd() / "tmp"):
    path = Path(path)
    # if path.exists():
    #     rmtree(path)
    path.mkdir(exist_ok=True, parents=True)

    return path


def delete_temp_dir(path: str | Path = Path.cwd() / "tmp"):
    path = Path(path)
    if path.exists():
        rmtree(path)
        