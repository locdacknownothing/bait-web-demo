import json

# from demo import ner_huggingface
from fine_tune.inference import get_nlp, predict
from file import read_df
from dataset import ent_maps, ent_iob2_maps


def div2li(content: str) -> str:
    return "<li>{}</li>".format(content)


if __name__ == "__main__":
    data_path = "/mnt/data/data/Data_Ner/Data_Ner_Test.xlsx"
    test_data = read_df(data_path)
    test_samples = test_data["title"].tolist()

    model_path = "/mnt/data/src/ner/fine_tune/train_original/model"
    nlp = get_nlp(model_path, device=1)
    html_contents = [
        predict(sample, nlp, base_ents=ent_maps) for sample in test_samples
    ]

    with open("html_contents.json", "w") as f:
        json.dump(html_contents, f)

    html_contents = [div2li(content) for content in html_contents]

    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NER Result</title>
</head>
<body>
    <div style="padding: 10px; margin: 10px;">
        <h3 style="text-align: center; padding-bottom: 10px;">Kết quả NER trên tập test</h3>
        <ol>
            {lis}
        </ol>
    </div>
</body>
</html>
""".format(
        lis="\n".join(html_contents)
    )

    from pathlib import Path

    file_path = Path("ner_result.html")
    file_path.write_text(html_content)
