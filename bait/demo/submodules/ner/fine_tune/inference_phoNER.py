import sys

sys.path.append("../")
import random

import spacy
from spacy import displacy
from transformers import pipeline
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

from fine_tune.constants_phoNER import *
from phoNER_COVID19_dataset import extract_data, ents, colors


# This function can display in jupyter environment only
def display(text, result, base_ents=None, colors=None):
    nlp = spacy.blank("en")
    doc = nlp.make_doc(text)

    ents = []
    for ent in result:
        start_char = ent["start"]
        end_char = ent["end"]
        label = ent["entity_group"]

        ent = doc.char_span(start_char, end_char, label)
        if ent is not None:
            ents.append(ent)
        else:
            print("Skip entity")

    doc.ents = ents
    if base_ents and colors:
        color_options = {key: value for key, value in zip(base_ents, colors)}
        displacy.render(
            doc, style="ent", jupyter=True, options={"colors": color_options}
        )
    else:
        displacy.render(doc, style="ent", jupyter=True)


if __name__ == "__main__":
    test_data = extract_data("../data/phoNER_COVID19/test_syllable.json")
    test_example = " ".join(random.choice(test_data["words"]))

    model_fine_tuned = AutoModelForTokenClassification.from_pretrained(SAVE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(SAVE_MODEL)
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
