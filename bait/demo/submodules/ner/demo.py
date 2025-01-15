import argparse
import gradio as gr
import spacy
from spacy_transformers import Transformer

from process import preprocess
from fine_tune.inference import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    ent_maps as base_ents,
    colors,
)


# Load a pre-trained NER model (you can replace this with your own model)
class HuggingfaceNER:
    def __init__(self, model_path=None):
        model_path = "/mnt/data/src/ner/fine_tune/train/model"
        model_fine_tuned = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.nlp = pipeline(
            "ner",
            model=model_fine_tuned,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=0,
        )


class SpacyNER:
    def __init__(self, model_path=None):
        if not model_path:
            model_path = "/mnt/data/src/ner/spacy_model/result_0410/model-best"

        self.nlp = spacy.load(model_path)


# Define the function to perform NER
def ner_huggingface(text, nlp=HuggingfaceNER().nlp, base_ents=base_ents, colors=colors):
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


def ner_spacy(text, nlp=SpacyNER().nlp, base_ents=base_ents, colors=colors):
    # TODO: convert doc back to text
    text, unidecode_text = preprocess(text)
    unidecode_doc = nlp(unidecode_text)
    doc = nlp.make_doc(text)
    doc.ents = unidecode_doc.ents

    if isinstance(base_ents, dict):
        base_ents = list(base_ents.values())

    color_options = {key: value for key, value in zip(base_ents, colors)}
    html_content = spacy.displacy.render(
        doc, style="ent", jupyter=False, options={"colors": color_options}
    )

    return html_content


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", "-t", default="huggingface", type=str)
    parser.add_argument("--model-path", "-p", default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    ner_fn = ner_huggingface if args.model_type == "huggingface" else ner_spacy

    # Define Gradio interface
    iface = gr.Interface(
        fn=ner_fn,
        inputs=gr.Textbox(lines=10, placeholder="Enter text here..."),
        outputs=gr.HTML(label="NER Results"),
        title="Named Entity Recognition (NER) Demo",
        description="Enter text to identify named entities (e.g., persons, organizations, locations).",
        examples=[
            ["Thủy Mộc nail 0334 908 256 177 Phạm Như Xương TP. ĐN"],
            ["Honda Thành Chuyên Sửa Chữa Bảo Dưỡng Xe Máy"],
            ["Veston Phương Ngân"],
        ],
    )

    # Run the interface
    iface.launch(debug=True)
