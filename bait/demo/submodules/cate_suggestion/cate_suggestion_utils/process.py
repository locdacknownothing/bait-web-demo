import json
import pandas

FILE_FORMATS = {"xlsx", "json"}


def check_format_file(input):
    if input.split(".")[-1].lower() in FILE_FORMATS:
        return input.split(".")[-1]

    else:
        return None


def convert_dict_cate(txt):
    result = {}
    with open(txt, "r", encoding="utf8") as f:
        data = f.readlines()

    for i in data:
        id, apple, here = i.split("\t")

        result[id] = {"apple": apple, "here": here}

    return result
