from unidecode import unidecode


def preprocess(s: str) -> str:
    s = s.lower()
    s = unidecode(s)
    s = "".join(c for c in s if c.isalnum())
    return s


def process_list(listStr: str | list[str]) -> list[str]:
    if isinstance(listStr, str):
        try:
            listStr = listStr[1:]
            textList = listStr.split(" ")
            pTextList = []

            for text in textList:
                text = text.strip()
                text = text[1:-2]
                text = preprocess(text)

                if text:
                    pTextList.append(text)

            return pTextList
        except:
            return []
    else:
        processedList = [preprocess(x) for x in listStr]
        return [x for x in processedList if x]
