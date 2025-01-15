import argparse
import os
import shutil
import subprocess
import time
import numpy as np
import torch

from model import EAST
from detect import detect_dataset


def eval_model(model_name, test_img_path, test_gt_path, submit_path, save_flag=True):
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST(False).to(device)
    model.load_state_dict(torch.load(model_name))
    model.eval()

    start_time = time.time()
    detect_dataset(model, device, test_img_path, submit_path)
    if not os.path.exists(os.path.join(test_gt_path, "gt.zip")):
        os.chdir(test_gt_path)
        res = subprocess.getoutput("zip -q gt.zip *.txt")
    os.chdir(submit_path)
    res = subprocess.getoutput("zip -q submit.zip *.txt")
    res = subprocess.getoutput("mv submit.zip ../")
    os.chdir("../")
    res = subprocess.getoutput(
        f"python3 ./evaluate/script.py -g={os.path.join(test_gt_path, 'gt.zip')} -s=./submit.zip"
    )
    print(res)
    os.remove("./submit.zip")
    print("eval time is {}".format(time.time() - start_time))

    if not save_flag:
        shutil.rmtree(submit_path)


def parse_args():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--model_path", type=str, default="./pths/pretrained/east_vgg16.pth"
    )
    parser.add_argument(
        "--test_img_path",
        type=str,
        default="/mnt/data1/src/ocr/vinai-vietnamese/test_img",
    )
    parser.add_argument(
        "--test_gt_path",
        type=str,
        default="/mnt/data1/src/ocr/EAST/evaluate",
    )
    parser.add_argument(
        "--submit_path", type=str, default="/mnt/data1/src/ocr/EAST/submit"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    eval_model(args.model_path, args.test_img_path, args.test_gt_path, args.submit_path)
