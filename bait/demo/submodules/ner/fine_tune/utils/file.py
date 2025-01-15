from pathlib import Path


def create_if_not_exist(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

    return path
