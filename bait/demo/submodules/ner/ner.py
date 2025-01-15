from os.path import dirname
from sys import path

path.append(dirname(__file__))

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)

from file import get_model, load_json, save_json
from process import preprocess
from schemas.output import NERResult


class NER(object):
    def __init__(self, model_path, device=0):
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.nlp = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
        self.ent_maps = {
            "K": "keyword",
            "N": "name",
            "A": "address",
        }

    def __call__(self, text):
        return self.forward(text)

    def forward(self, source: str | list[str] | dict):
        if type(source) not in [str, list, dict]:
            raise TypeError("Invalid source data type, should be str, list or dict.")
        elif isinstance(source, str):
            text_values = [source]
        elif isinstance(source, dict):
            images = list(source.keys())
            text_values = list(source.values())
        else:
            text_values = source

        text_tuples = [preprocess(x) for x in text_values]
        texts = [x[0] for x in text_tuples]
        unidecode_texts = [x[1] for x in text_tuples]

        results = self.nlp(unidecode_texts)
        result_dict = [self._postprocess_(*params) for params in zip(results, texts)]

        if isinstance(source, dict):
            return {key: value for key, value in zip(images, result_dict)}
        else:
            return result_dict

    def _postprocess_(self, result, orig_text):
        result_dict = {}
        end_char = None

        for ent in result:
            start_char = ent["start"]
            if end_char is not None and start_char <= end_char:
                continue

            while start_char > 0 and orig_text[start_char - 1] != " ":
                start_char += 1
                if start_char == ent["end"]:
                    break

            end_char = ent["end"]
            while end_char < len(orig_text) and orig_text[end_char] != " ":
                end_char += 1

            group = ent["entity_group"]
            group = self.ent_maps.get(group, "O") if self.ent_maps else group
            text = orig_text[start_char:end_char]

            if not text:
                continue

            if group in result_dict:
                result_dict[group].append(text)
            else:
                result_dict[group] = [text]

        result_dict = dict(sorted(list(result_dict.items()), key=lambda x: x[0]))
        return NERResult(**result_dict)


if __name__ == "__main__":
    model = get_model(
        src_path="/mnt/data/ner/v1.0.0/model_2511",
        src_server="ts0107@192.168.1.41",
        dst_path="./models",
    )
    ner = NER(model)

    # Type 2: list of texts
    source = [
        "Thủy Mộc nail 0334 908 256 177 Phạm Như Xương TP. ĐN",
        "Honda Thành Chuyên Sửa Chữa Bảo Dưỡng Xe Máy",
        "",
    ]
    results = ner(source)
    print(results)

    # Type 3: dict of images' text
    # # Load a json file for images' text dictionary
    # source = load_json("image_text.json")
    # result_dict = ner(source)
    # print(result_dict)
