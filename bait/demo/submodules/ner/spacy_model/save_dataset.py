from os.path import join, dirname
import sys

sys.path.append(join(dirname(__file__), ".."))
from pathlib import Path

import spacy
from spacy.util import filter_spans
from spacy.tokens import DocBin
from tqdm import tqdm

from dataset import convert_data, ent_maps
from file import read_df
from phoNER_COVID19_dataset import convert_data as convert_phoNER, extract_data


def save_data(data: dict, save_path: str):
    nlp = spacy.blank("vi")
    doc_bin = DocBin()

    for example in tqdm(data):
        text = example["text"]
        labels = example["entities"]
        doc = nlp.make_doc(text)
        ents = []

        for start, end, label in labels:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if not span:
                pass
            else:
                ents.append(span)

            filtered_ents = filter_spans(ents)
            doc.ents = filtered_ents
            doc_bin.add(doc)

    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    doc_bin.to_disk(str(save_path))


def main_phoNER():
    train_json_file = "./train_syllable.json"
    test_json_file = "./test_syllable.json"

    train_data = convert_phoNER(extract_data(train_json_file))
    test_data = convert_phoNER(extract_data(test_json_file))

    save_data(train_data, "./train.spacy")
    save_data(test_data, "./test.spacy")


def main():
    csv_file = "../ner.xlsx"
    raw_data = read_df(csv_file)
    data = convert_data(raw_data, ent_maps)

    save_data(data, "./train.spacy")


if __name__ == "__main__":
    main()
