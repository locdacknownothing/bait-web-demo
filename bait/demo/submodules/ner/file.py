from pathlib import Path
from os import makedirs
from os.path import abspath, dirname, exists, join
from subprocess import run, CalledProcessError
from json import dump as dump_json_, load as load_json_
from pickle import dump as dump_pkl, load as load_pkl
import pandas as pd


def read_df(file_path: str | Path):
    file_path = Path(file_path)
    if file_path.suffix == ".xlsx":
        df = pd.read_excel(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("File extension is not supported.")

    return df


def run_command(command):
    try:
        run(command, shell=True)
    except CalledProcessError as e:
        raise str(e)


def save_json(dict_: dict, save_file: str):
    save_folder = dirname(abspath(save_file))
    if not exists(save_folder):
        makedirs(save_folder)

    with open(save_file, "w") as outfile:
        dump_json_(dict_, outfile, ensure_ascii=False)


def load_json(load_file: str) -> dict:
    with open(load_file, "r") as outfile:
        res = load_json_(outfile)

    return res


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


def get_model(
    src_path: str,
    src_server: str = "ts0107@192.168.1.41",
    dst_path: str = "./models",
) -> str:
    weight_name = src_path.split("/")[-1]
    abs_dst_path = abspath(dst_path)
    dst_file = join(abs_dst_path, weight_name)

    if not exists(dst_path):
        makedirs(abs_dst_path)

    if not exists(dst_file):
        command = "rsync -aP {}:{} {}".format(src_server, src_path, dst_path)
        try:
            print("Downloading model ...")
            run(command, shell=True)
        except CalledProcessError as e:
            raise str(e)
    else:
        print("Model is already located. No downloading.")

    return dst_file
