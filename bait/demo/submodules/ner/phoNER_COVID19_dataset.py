from json import loads as json_loads


colors = [
    "#ff0000",  # Red
    "#00ff00",  # Green
    "#0000ff",  # Blue
    "#ffff00",  # Yellow
    "#ff00ff",  # Magenta
    "#00ffff",  # Cyan
    "#c0c0c0",  # Silver
    "#800000",  # Maroon
    "#808000",  # Olive
    "#008080",  # Teal
]
ents = [
    "PATIENT_ID",
    "NAME",
    "AGE",
    "GENDER",
    "JOB",
    "LOCATION",
    "ORGANIZATION",
    "SYMPTOM_AND_DISEASE",
    "TRANSPORTATION",
    "DATE",
]
color_options = {key: value for key, value in zip(ents, colors)}


def extract_data(json_file: str) -> list:
    data = []
    with open(json_file, "r") as f:
        for line in f:
            data.append(json_loads(line))
    return data


def convert_data(data: list) -> dict:
    training_data = []
    for example in data:
        temp_dict = {}
        temp_dict["text"] = " ".join(example["words"])
        temp_dict["entities"] = []

        start = None
        label = None
        current_pos = 0

        for i, (word, tag) in enumerate(zip(example["words"], example["tags"])):
            word_len = len(word)

            if tag.startswith("B-"):
                if start is not None:
                    temp_dict["entities"].append((start, current_pos - 1, label))

                start = current_pos
                label = tag[2:]
            elif tag.startswith("I-"):
                if start is not None and tag[2:] == label:
                    pass
                else:
                    raise ValueError("Error in annotation.")
            else:
                if start is not None:
                    temp_dict["entities"].append((start, current_pos - 1, label))
                    start = None
                    label = None

            current_pos += word_len + 1

        # If the last entity hasn't been appended yet, append it
        if start is not None:
            temp_dict["entities"].append((start, current_pos - 1, label))

        training_data.append(temp_dict)
    return training_data


def get_tags():
    tags = ["O"]
    for ent in ents:
        tags.extend([f"B-{ent}", f"I-{ent}"])
    return tags


def get_id2label():
    tags = get_tags()
    indices = range(len(tags))

    id2label = {key: value for key, value in zip(indices, tags)}
    return id2label


def get_label2id():
    id2label = get_id2label()
    return {value: key for key, value in id2label.items()}
