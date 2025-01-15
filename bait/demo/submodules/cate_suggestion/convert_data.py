import pandas as pd
import os
import re
import string
import unidecode
import random


def export_txt(link_df):
    name_txt = link_df.split("/")[-1].split(".")[0] + ".txt"
    df = pd.read_excel(link_df)

    save_path = os.path.join(
        "/home/dgm_bait_02/short-text-classification/data/dgm_apple", name_txt
    )
    for i in range(0, len(df)):

        title = df.iloc[i]["title"].lower()

        label_numeric = int(df.iloc[i]["label_numeric"])

        with open(save_path, "a+", encoding="utf-8") as f:
            f.write(str(label_numeric) + "\t" + title + "\n")


if __name__ == "__main__":

    export_txt("/home/dgm_bait_02/cate_suggestion/Data/train_apple.xlsx")
    export_txt("/home/dgm_bait_02/cate_suggestion/Data/val_apple.xlsx")
