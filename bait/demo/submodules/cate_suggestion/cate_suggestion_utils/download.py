import pandas as pd
import numpy as np
import cv2
from urllib.error import HTTPError
import urllib.request
import os
import glob


def read_img(url):
    print(url)
    try:
        req = urllib.request.urlopen(url)

    except HTTPError as e:
        return
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)

    try:
        img = cv2.imdecode(arr, -1)  # 'Load it as it is'
    except:
        return

    return img


if __name__ == "__main__":
    df = pd.read_excel(
        "/home/dgm_bait_02/short-text-classification/input/Test_DMS.xlsx"
    )
    for i in range(0, len(df)):
        if i < 103341:
            continue
        name_img = str(df.iloc[i]["id"])
        link = df.iloc[i]["image_url"]

        img = read_img(link)

        if img is None:
            continue

        w, h, _ = img.shape

        if w >= 960 or h >= 960:
            continue

        path_img = (
            "/home/dgm_bait_02/short-text-classification/img_dms/" + name_img + ".jpg"
        )
        cv2.imwrite(path_img, img)

    print("finish")
