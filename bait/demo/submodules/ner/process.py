import string
import re
from unidecode import unidecode


vocab = "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789 "


def remove_urls(text):
    """
    Removes URLs from the text.

    Args:
        text (str): Input text.

    Returns:
        str: Text without URLs.
    """
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.sub(url_pattern, "", text)


def remove_extra_whitespace(text):
    """
    Removes redundant spaces and trims leading/trailing spaces.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text with single spaces.
    """
    return re.sub(r"\s+", " ", text).strip()


def remove_outer_punctuations(text):
    """
    Removes any leading or trailing punctuation from the text.

    Args:
        text (str): Input text.

    Returns:
        str: Text with leading/trailing punctuation removed.
    """
    pattern = f"^[^{re.escape(vocab)}]+|[^{re.escape(vocab)}]+$"
    text = re.sub(pattern, "", text)
    return text


def process_text(text: str) -> str:
    text = remove_urls(text)
    text = remove_outer_punctuations(text)
    text = remove_extra_whitespace(text)
    return text


def process_text_list(text: str | list[str], use_unidecode: bool = True) -> list[str]:
    if isinstance(text, str):
        syllables = text.strip().split(" ")
    else:
        syllables = text

    processed_syllables = [process_text(syllable) for syllable in syllables]

    if use_unidecode:
        text_list = [unidecode(x) for x in processed_syllables if x]
    else:
        text_list = [x for x in processed_syllables if x]
    return text_list


def process_text_string(text: str, use_unidecode: bool = True) -> str:
    text_list = process_text_list(text)
    return " ".join(text_list)


def process_tag_list(tag: str, label2id: dict | None = None) -> list[str]:
    tag_list = tag.strip().split(" ")

    def _map_tag_(tag: str) -> str:
        if tag in label2id:
            return label2id[tag]
        else:
            return "O"

    if label2id is not None:
        tags = [_map_tag_(x) for x in tag_list]
    else:
        tags = tag_list

    return tags


def clean_text(text):
    text = re.sub("<.*?>", "", text).strip()
    text = re.sub("(\s)+", r"\1", text)
    return text


def remove_numbers(text_in):
    for ele in text_in.split():
        if ele.isdigit():
            text_in = text_in.replace(ele, "@")
    for character in text_in:
        if character.isdigit():
            text_in = text_in.replace(character, "@")
    return text_in


def remove_special_characters(text):
    chars = re.escape(string.punctuation)
    return re.sub(r"[" + chars + "]", "", text)


def preprocess(text: str) -> tuple[str, str]:
    text_list = process_text_list(text, False)
    text = " ".join(text_list)
    unidecode_text = unidecode(text.lower())
    return text, unidecode_text


def process_text_with_tag(
    text_str: str,
    tag_str: str,
    label2id: dict | None = None,
    use_unidecode: bool = True,
) -> list[tuple]:
    syllables = text_str.strip().split(" ")
    tags = process_tag_list(tag_str, label2id)

    if len(syllables) != len(tags):
        return None

    processed_syllables = [process_text(syllable) for syllable in syllables]
    if use_unidecode:
        processed_syllables = [unidecode(x) for x in processed_syllables]

    zipped_data = [
        (syllable, tag) for syllable, tag in zip(processed_syllables, tags) if syllable
    ]
    syllables = [x[0] for x in zipped_data]
    tags = [x[1] for x in zipped_data]
    return syllables, tags


def io2iob2(io_tags):
    iob2_tags = []
    previous_tag = "O"  # Initial previous tag
    for tag in io_tags:
        if tag == "O":
            iob2_tags.append("O")
            previous_tag = "O"  # Reset previous tag
        else:
            if previous_tag == tag:
                # If the same entity follows, it's "I-<entity>"
                iob2_tags.append(f"I-{tag}")
            else:
                # Otherwise, it's "B-<entity>"
                iob2_tags.append(f"B-{tag}")
            previous_tag = tag  # Update the previous tag to current
    return iob2_tags
