import json
import pandas as pd


class Exporter:
    def __init__(self, **kwargs):
        self.data = {**kwargs}

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.data, f)

    def to_csv(self, path):
        df = pd.DataFrame([self.data])
        df.to_csv(path, index=False, mode="a")

    def to_excel(self, path):
        df = pd.DataFrame([self.data])
        df.to_excel(path, index=False)
