import argparse
import os
import shutil
import subprocess
import time
import numpy as np
import torch

from model import EAST
from detect import detect_dataset


def eval_model(model_name, test_img_path, test_gt_path, save_flag=True):
    os.chdir(os.getcwd())
    submit_path = os.path.abspath("./submit")

    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(False).to(device)
    checkpoint = torch.load(model_name, map_location=device)
    try:
        model.load_state_dict(checkpoint)
    except:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    start_time = time.time()
    detect_dataset(model, device, test_img_path, submit_path)

    eval_path = os.path.abspath("./evaluate")
    gt_zip = os.path.join(eval_path, "gt.zip")
    submit_zip = os.path.join(eval_path, "submit.zip")

    # if not os.path.exists(gt_zip):
    test_gt_text = os.path.join(test_gt_path, "*.txt")
    res = subprocess.getoutput("zip -q {} {}".format(gt_zip, test_gt_text))

    submit_gt_text = os.path.join(submit_path, "*.txt")
    res = subprocess.getoutput("zip -q {} {}".format(submit_zip, submit_gt_text))

    os.chdir(eval_path)
    res = subprocess.getoutput(f"python3 ./script.py -g=./gt.zip -s=./submit.zip")
    print(res)
    os.remove("./gt.zip")
    os.remove("./submit.zip")
    print("eval time is {}".format(time.time() - start_time))

    if not save_flag:
        shutil.rmtree(submit_path)


def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--model-path", type=str, default="./pths/pretrained/east_vgg16.pth"
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default="/mnt/data/data/Data_Ve_Chu",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    eval_model(
        args.model_path,
        os.path.join(args.test_data_path, "test_img"),
        os.path.join(args.test_data_path, "test_gt"),
    )
