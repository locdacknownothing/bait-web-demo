import json
import string
import re
import torch
import pandas as pd
from tqdm import tqdm
import csv
import json
import yaml
import numpy as np

from cate_suggestion_utils.process import check_format_file, convert_dict_cate
from transformers import (
    get_linear_schedule_with_warmup,
    AdamW,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)


def clean_text(text):
    pattern = f"[{re.escape(string.punctuation)}]"
    pattern2 = r"[0-9]"
    text = re.sub(pattern, "", text)
    text = re.sub(pattern2, "", text)
    return " ".join(text.split())


class Model_Cate:
    def __init__(self, model_file, label_list, map_name_cate, config):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_file)
        self.tokenizer = AutoTokenizer.from_pretrained(model_file)
        temp_dict = {}
        with open(label_list, "r") as f:
            temp_dict = json.load(f)

        self.label_dict = {value: key for key, value in temp_dict.items()}

        with open(config, "r") as f:
            self.config = yaml.safe_load(f)

        self.dict_label = convert_dict_cate(map_name_cate)

    def predict(self, text):
        text = clean_text(text).lower()

        test_encodings = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**test_encodings)

        _, predicted = torch.topk(outputs.logits.flatten(), self.config["top-k"])

        # total = torch.sum(outputs.logits.flatten()[outputs.logits.flatten()>self.config["top-k"]])
        # print(total)

        _ = _[_ >= self.config["threshold"]]

        total = torch.sum(_).tolist()
        prob = [x/total for x in _.tolist()]

        if self.config["type"] == "apple":
            return list(
                dict.fromkeys(
                    [
                        self.dict_label[
                            self.label_dict[int(predicted[i])].split(".")[0]
                        ]["apple"][0:]
                        for i in range(0, len(_))
                    ]
                )
            ),prob

        elif self.config["type"] == "here":
            return list(
                dict.fromkeys(
                    [
                        self.dict_label[
                            self.label_dict[int(predicted[i])].split(".")[0]
                        ]["here"][0:-1]
                        for i in range(0, len(_))
                    ]
                )
            )

    def multi_input(self, input):

        if check_format_file(input) == "xlsx":
            data = pd.read_excel(input)
            data = data.to_dict(orient="records")

        elif check_format_file(input) == "json":
            with open(input, "rb") as f:
                data = json.load(f)
        else:
            raise TypeError("Format không hỗ trợ")

        result = {}
        for i in tqdm(data):

            text = i["title"]
            if text is None:
                continue
            out = self.predict(text)

            result[i["id"]] = out

        if self.config["export"]:
            if self.config["path_export"] == "None":
                path = "./output.csv"
            else:
                path = self.config["path_export"]

            with open(path, "w") as output:
                writer = csv.writer(output)
                for key, value in result.items():
                    writer.writerow([key, value])

        return result


if __name__ == "__main__":

    model = Model_Cate(
        model_file="/home/dgm_bait_02/short-text-classification/output/bert-base-uncased-finetuned-dgm_apple/checkpoint-1254600-231224",
        label_list="/home/dgm_bait_02/short-text-classification/dgm_label.json",
        map_name_cate="/home/dgm_bait_02/dms/cate_suggestion/configs/list_cate.txt",
        config="/home/dgm_bait_02/dms/cate_suggestion/configs/default.yaml",
    )
    # print(model.predict("Ngọc Cháo Lòng"))
    model.multi_input(
        "/home/dgm_bait_02/dms/cate_suggestion/Dayta_PA_Re-deliver_Batch_maintennance_.xlsx"
    )
