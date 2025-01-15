from functools import partial
from itertools import groupby

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
from imblearn.over_sampling import SMOTE, SMOTEN, ADASYN

from process import process_text_list, process_tag_list, process_text_with_tag, io2iob2
from augment import augment


ents = ["K", "N", "A"]
ent_maps = {
    "K": "KEYWORD",
    "N": "NAME",
    "A": "ADDRESS",
}
ent_iob2_maps = {
    "B-K": "KEYWORD",
    "B-N": "NAME",
    "B-A": "ADDRESS",
    "I-K": "KEYWORD",
    "I-N": "NAME",
    "I-A": "ADDRESS",
}
colors = [
    "#ffff00",  # Yellow
    "#ff00ff",  # Magenta
    "#00ffff",  # Cyan
]
color_options = {key: value for key, value in zip(list(ent_maps.values()), colors)}


def convert_data(data: pd.DataFrame, ent_maps: dict):
    data = data[["title", "label"]].dropna()
    training_data = []
    for _, row in data.iterrows():
        text_str = str(row["title"])
        tag_str = str(row["label"])

        data = process_text_with_tag(text_str, tag_str)
        if data is None:
            continue

        syllables, tags = data

        temp_dict = {}
        temp_dict["text"] = " ".join(syllables)
        temp_dict["entities"] = []

        start = None
        label = None
        current_position = 0

        for syllable, tag in zip(syllables, tags):
            len_ = len(syllable)

            if tag != "O":
                if start is not None:
                    temp_dict["entities"].append((start, current_position - 1, label))

                start = current_position
                label = ent_maps[tag]
            else:
                if start is not None:
                    temp_dict["entities"].append((start, current_position - 1, label))
                    start = None
                    label = None

            current_position += len_ + 1

        # If the last entity hasn't been appended yet, append it
        if start is not None:
            temp_dict["entities"].append((start, current_position - 1, label))

        training_data.append(temp_dict)

    return training_data


def get_labels(scheme: str = "iob2"):
    if scheme == "io":
        return ["O"] + ents
    else:
        labels = ["O"]

        for ent in ents:
            labels.extend([f"B-{ent}", f"I-{ent}"])
        return labels


def get_id2label(scheme: str = "iob2"):
    labels = get_labels(scheme)
    indices = range(len(labels))

    id2label = {key: value for key, value in zip(indices, labels)}
    return id2label


def get_label2id(scheme: str = "iob2"):
    id2label = get_id2label(scheme)
    return {value: key for key, value in id2label.items()}


def tokenize_and_align_labels(words, tags, tokenizer, label_all_tokens=True):
    """
    ## The below function does 2 jobs

    1. set -100 as the label for special tokens
    2. mask the subword representations after the first subword
    """
    tokenized_input = tokenizer(
        words,
        padding="max_length",
        truncation=True,
        max_length=128,
        is_split_into_words=True,
    )
    labels = []

    for i, label in enumerate(tags):
        word_ids = tokenized_input.word_ids(batch_index=i)
        previous_word_id = None

        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                # set -100 b/c these special characters
                # are ignored by pytorch when training
                label_ids.append(-100)
            elif word_id != previous_word_id:
                # if current word_id != prev when it's the most regular case
                # and add the corresponding token
                label_ids.append(label[word_id])
            else:
                # for sub-word which has the same word_id
                # set -100 as well only if label_all_tokens=False
                if label_all_tokens:
                    label_ids.append(label[word_id])
                else:
                    label_ids.append(-100)

            previous_word_id = word_id

        labels.append(label_ids)

    tokenized_input["labels"] = labels
    return tokenized_input


def get_dataset(raw_data: pd.DataFrame, tokenizer, label2id: dict) -> Dataset:
    raw_data = raw_data[["title", "label"]].dropna()
    words = raw_data["title"].tolist()
    tags = raw_data["label"].tolist()

    zipped_data = [
        process_text_with_tag(str(word), str(tag), label2id)
        for word, tag in zip(words, tags)
    ]
    zipped_data = [x for x in zipped_data if x]

    words = [x[0] for x in zipped_data]
    tags = [x[1] for x in zipped_data]

    tokenized_data = tokenize_and_align_labels(words, tags, tokenizer, True).data
    data = Dataset.from_dict(tokenized_data)
    return data


def clean_data(
    raw_data: pd.DataFrame,
    label2id: dict | None = None,
    use_unidecode: bool = True,
    scheme: str = "iob2",
) -> pd.DataFrame:
    raw_data = raw_data[["title", "label"]].dropna()

    zipped_data = [
        process_text_with_tag(str(word), str(tag), use_unidecode=use_unidecode)
        for word, tag in raw_data.to_numpy().tolist()
    ]
    zipped_data = [x for x in zipped_data if x is not None]

    cleaned_data = pd.DataFrame(zipped_data, columns=["tokens", "ner_tags"])
    if scheme == "iob2":
        cleaned_data["ner_tags"] = cleaned_data["ner_tags"].apply(
            lambda tags: [label2id[tag] if label2id else tag for tag in io2iob2(tags)]
        )
    else:
        cleaned_data["ner_tags"] = cleaned_data["ner_tags"].apply(
            lambda tags: [label2id[tag] if label2id else tag for tag in tags]
        )

    return cleaned_data


def get_conll_dataset(
    cleaned_data: pd.DataFrame, labels: list | None = None, augment_data: bool = False
) -> Dataset:
    data = Dataset.from_pandas(
        cleaned_data,
        features=Features(
            {
                "tokens": Sequence(feature=Value(dtype="string")),
                "ner_tags": Sequence(feature=ClassLabel(names=labels)),
            }
        ),
    )

    if augment_data:
        data = augment(data, keep_original=True)

    return data


def get_train_dataset(conll_dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    train_dataset = conll_dataset.map(
        lambda x: tokenize_and_align_labels(
            x["tokens"], x["ner_tags"], tokenizer, False
        ),
        batched=True,
        remove_columns=["tokens", "ner_tags"],
    )
    return train_dataset


def conlldf2text(df: pd.DataFrame) -> str:
    text = ""
    for _, row in df.iterrows():
        for token, ner_tag in zip(row["tokens"], row["ner_tags"]):
            text += f"{token} {ner_tag}\n"

        text += "\n"

    return text


def conlltext2df(text: str) -> pd.DataFrame:
    data = [[[], []]]
    prev_line = ""

    for line in text.strip().split("\n"):
        line = line.strip()

        if line != "":
            token, ner_tag = line.split(" ")[:2]
            data[-1][0].append(token)
            data[-1][1].append(ner_tag)
        else:
            if prev_line != "":
                data.append([[], []])

        prev_line = line

    df = pd.DataFrame(data, columns=["tokens", "ner_tags"])
    return df


def word2phrase(tokens, ner_tags):
    phrase_tokens = []
    phrase_tags = []
    start = 0

    # Use groupby to find consecutive batches
    for label, group in groupby(ner_tags):
        # Calculate the length of the current group
        group_length = len(list(group))
        end = start + group_length

        phrase_tokens.append(" ".join(tokens[start:end]))
        phrase_tags.append(label)

        # Update start index for the next group
        start = end

    return phrase_tokens, phrase_tags


def convert_to_phrase_data(df: pd.DataFrame) -> pd.DataFrame:
    phrase_data = [
        word2phrase(tokens, ner_tags)
        for tokens, ner_tags in zip(df["tokens"], df["ner_tags"])
    ]

    phrase_data = pd.DataFrame(phrase_data, columns=["tokens", "ner_tags"])
    return phrase_data


def list2one(df: pd.DataFrame) -> pd.DataFrame:
    phrase_data = convert_to_phrase_data(df)
    tokens = phrase_data["tokens"].explode().reset_index(drop=True)
    ner_tags = phrase_data["ner_tags"].explode().reset_index(drop=True)
    df = pd.DataFrame({"tokens": tokens, "ner_tags": ner_tags})
    df.dropna(inplace=True)
    return df


def one2list(df: pd.DataFrame) -> pd.DataFrame:
    df["tokens"] = df["tokens"].apply(lambda x: x.split(" "))

    for i in range(len(df)):
        ner_tags = df.loc[i, "ner_tags"]
        tokens = df.loc[i, "tokens"]
        if ner_tags == "O":
            ner_tags = [ner_tags] * len(tokens)
        else:
            ner_tags = [f"B-{ner_tags}"] + [f"I-{ner_tags}"] * (len(tokens) - 1)

        df.loc[i, "ner_tags"] = " ".join(ner_tags)

    df["ner_tags"] = df["ner_tags"].apply(lambda x: x.split(" "))
    return df[["tokens", "ner_tags"]]


def upsample_data(one_tag_df: pd.DataFrame) -> pd.DataFrame:
    X = one_tag_df["tokens"].apply(str).to_numpy()
    y = one_tag_df["ner_tags"].apply(str).to_numpy()

    sm = SMOTEN()
    X_res, y_res = sm.fit_resample(X.reshape(-1, 1), y)
    balanced_df = pd.DataFrame({"tokens": X_res.reshape(-1), "ner_tags": y_res})
    return balanced_df


def word2phrase2(tokens, ner_tags):
    phrase_tokens = []
    phrase_tags = []
    labels = []
    start = 0

    # Use groupby to find consecutive batches
    for label, group in groupby(ner_tags):
        # Calculate the length of the current group
        group_length = len(list(group))
        end = start + group_length

        labels.append(label)
        phrase_tokens.append(" ".join(tokens[start:end]))

        if label == "O":
            ner_tags = " ".join([label] * group_length)
        else:
            ner_tags = " ".join([f"B-{label}"] + [f"I-{label}"] * (group_length - 1))

        phrase_tags.append(ner_tags)

        # Update start index for the next group
        start = end

    return phrase_tokens, phrase_tags, labels


def list2one2(df: pd.DataFrame) -> pd.DataFrame:
    phrase_data = [
        word2phrase2(tokens, ner_tags)
        for tokens, ner_tags in zip(df["tokens"], df["ner_tags"])
    ]

    df = pd.DataFrame(phrase_data, columns=["tokens", "ner_tags", "labels"])
    df = df.explode(["tokens", "ner_tags", "labels"])
    df.dropna(inplace=True)
    return df


def get_conll_dataset2(one_tag_df: pd.DataFrame, label2id: dict | None) -> Dataset:
    one_tag_df["tokens"] = one_tag_df["tokens"].apply(
        lambda x: list(map(str, x.split(" ")))
    )
    one_tag_df["ner_tags"] = one_tag_df["ner_tags"].apply(
        lambda x: [label2id.get(tag, 0) for tag in x.split(" ")]
    )
    one_tag_df = one_tag_df.reset_index(drop=True)

    data = Dataset.from_pandas(one_tag_df)
    return data


def upsample_tokenized_dataset(dataset: Dataset, labels: list | pd.Series) -> Dataset:
    # Convert the dataset to a pandas DataFrame
    df = dataset.to_pandas()
    X = np.vstack(df.apply(np.hstack, axis=1))
    y = labels

    # Apply upsampling using ADASYN algorithm
    adasyn = ADASYN()
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    X_new = X_resampled.reshape(X_resampled.shape[0], df.shape[-1], -1)
    # X_new = X_new.transpose(1, 0, 2)

    resampled_df = pd.DataFrame(
        {column_name: list(X_new[:, i]) for i, column_name in enumerate(df.columns)}
    )
    # print(resampled_df["labels"].value_counts())
    resampled_dataset = Dataset.from_pandas(resampled_df)
    print(dataset)
    print(resampled_dataset)
    return resampled_dataset
