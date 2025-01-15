from os.path import join, dirname
import sys

sys.path.append(join(dirname(__file__), ".."))
import random

import spacy
from spacy import displacy
from transformers import pipeline
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from dataset import ents, colors
from file import read_df
from fine_tune.constants import *
from process import preprocess


def get_nlp(model_path, device=0):
    model_fine_tuned = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    nlp = pipeline(
        "ner",
        model=model_fine_tuned,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )
    return nlp


def predict(text, nlp, base_ents=ents, colors=colors):
    text, unidecode_text = preprocess(text)
    result = nlp(unidecode_text)
    spacy_nlp = spacy.blank("en")
    doc = spacy_nlp.make_doc(text)

    ents = []
    end_char = None

    for ent in result:
        start_char = ent["start"]

        if end_char is not None and start_char <= end_char:
            continue

        while start_char > 0 and text[start_char - 1] != " ":
            start_char += 1
            if start_char == ent["end"]:
                break

        end_char = ent["end"]
        while end_char < len(text) and text[end_char] != " ":
            end_char += 1

        group = ent["entity_group"]
        if isinstance(base_ents, list):
            label = group
        else:
            label = base_ents[group]

        doc_ent = doc.char_span(start_char, end_char, label)
        if doc_ent is not None:
            ents.append(doc_ent)
        else:
            # print(ent)
            pass

    doc.ents = ents

    if isinstance(base_ents, dict):
        base_ents = list(base_ents.values())

    color_options = {key: value for key, value in zip(base_ents, colors)}
    html_content = spacy.displacy.render(
        doc, style="ent", jupyter=False, options={"colors": color_options}
    )

    return html_content


# This function can display in jupyter environment only
def display(text, result, base_ents=None, colors=None):
    nlp = spacy.blank("en")
    doc = nlp.make_doc(text)

    ents = []
    for ent in result:
        start_char = ent["start"]
        end_char = ent["end"]
        if isinstance(base_ents, list):
            label = ent["entity_group"]
        else:
            label = base_ents[ent["entity_group"]]

        ent = doc.char_span(start_char, end_char, label)
        if ent is not None:
            ents.append(ent)
        else:
            # print("Skip entity")
            pass

    doc.ents = ents
    if base_ents and colors:
        if isinstance(base_ents, dict):
            base_ents = list(base_ents.values())

        color_options = {key: value for key, value in zip(base_ents, colors)}
        displacy.render(
            doc, style="ent", jupyter=True, options={"colors": color_options}
        )
    else:
        displacy.render(doc, style="ent", jupyter=True)


if __name__ == "__main__":
    test_data = read_df("/mnt/data/src/ner/ner.xlsx")
    test_example = random.choice(test_data["Title"].tolist())

    save_model = "/mnt/data/src/ner/fine_tune/train/model"
    model_fine_tuned = AutoModelForTokenClassification.from_pretrained(save_model)
    tokenizer = AutoTokenizer.from_pretrained(save_model)
    nlp = pipeline(
        "ner",
        model=model_fine_tuned,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0,
    )

    result = nlp(test_example)
    print(result)

    # NOTE: the below line of code works only in jupyter environment
    # display(test_example, result, ents, colors)
