from os import makedirs
from os.path import abspath, dirname, exists, join
from subprocess import run as run_subproc, CalledProcessError
from pathlib import Path


IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}


def dirwalk(folder: str):
    if folder.split(".")[-1].lower() in IMG_FORMATS:
        return [folder]
    else:
        path = Path(folder)
        files = sorted(
            x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS
        )
        return files


def get_weights(
    weight_path: str,
    weight_server: str = "ts0107@192.168.1.41",
    out_dir: str = "./weights",
) -> str:
    weight_name = weight_path.split("/")[-1]
    abs_out_dir = abspath(out_dir)
    dst_file = join(abs_out_dir, weight_name)

    if not exists(out_dir):
        makedirs(abs_out_dir)

    if not exists(dst_file):
        command = "rsync -aP {}:{} {}".format(weight_server, weight_path, out_dir)
        try:
            print("Downloading weights ...")
            run_subproc(command, shell=True)
        except CalledProcessError as e:
            raise str(e)
    else:
        print("Weights is already located. No downloading.")

    return dst_file
