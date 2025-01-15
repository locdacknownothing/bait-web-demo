from os import listdir, makedirs
from os.path import abspath, basename, dirname, exists, isdir, islink, join
from shutil import rmtree
from pathlib import Path
import ntpath
import subprocess
import cv2
from pickle import load as load_pkl, dump as dump_pkl
from json import dump as dump_json_, load as load_json_


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


def make_dir(dir):
    if exists(dir):
        rmtree(dir)
    makedirs(dir)


def remove_dir(dir):
    if exists(dir):
        rmtree(dir)


def crop_img_from_txt(path_img, path_txt, dir):

    txts = []
    dirwalk(path_txt, txts, "*.txt")

    poi_path = join(dir, "poi")
    pa_path = join(dir, "pa")

    make_dir(poi_path)
    make_dir(pa_path)

    for txt in txts:
        name = ntpath.split(txt)[-1][0:-4]
        img = cv2.imread(join(path_img, name + ".jpg"))

        # print(join(path_img))

        h_or, w_or, _ = img.shape

        with open(txt, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            count = 0
            for i in lines:
                cls, x, y, w, h = i.split(" ")
                x = float(x) * float(w_or)
                y = float(y) * float(h_or)
                w = float(w) * float(w_or)
                h = float(h) * float(h_or)

                roi_x = x - float(w / 2)
                roi_y = y - float(h / 2)

                roi = img[int(roi_y) : int(roi_y + h), int(roi_x) : int(roi_x + w)]
                name_save = name + "_" + str(count) + ".jpg"

                cls_name = "poi" if int(cls) == 1 else "pa"
                cv2.imwrite(join(join(dir, cls_name), name_save), roi)
                count += 1


def get_weights(
    file_name: str,
    weight_server: str = "ts0107@192.168.1.41",
    out_dir: str = "./weights",
) -> str:
    # Auto-join if not absolute path
    if not file_name.startswith("/"):
        file_name = join("/mnt/data/od/", file_name)

    makedirs(out_dir, exist_ok=True)
    dst_file = join(out_dir, basename(file_name))

    if not exists(dst_file):
        command = "rsync -aP {}:{} {}".format(weight_server, file_name, out_dir)
        try:
            print("Downloading weights ...")
            subprocess.run(command, shell=True)
        except subprocess.CalledProcessError as e:
            raise str(e)
    else:
        print("Weights is already located. No downloading.")

    return dst_file


def save_object(object_: object, save_file: str):
    save_folder = dirname(abspath(save_file))
    if not exists(save_folder):
        makedirs(save_folder)

    with open(save_file, "wb") as file:
        dump_pkl(object_, file)


def load_object(load_file: str) -> object:
    with open(load_file, "rb") as file:
        res = load_pkl(file)

    return res


def save_json(dict_: dict, save_file: str):
    save_folder = dirname(abspath(save_file))
    if not exists(save_folder):
        makedirs(save_folder)

    with open(save_file, "w") as outfile:
        dump_json_(dict_, outfile)


def load_json(load_file: str) -> dict:
    with open(load_file, "r") as outfile:
        res = load_json_(outfile)

    return res
