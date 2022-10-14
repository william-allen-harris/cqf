from dataclasses import dataclass
import pandas as pd

from controllers.base_data import BaseData


@dataclass
class Data(BaseData):
    time_to_predict: int
    feature_set: str

    def dataframe(self):
        df = pd.read_csv(f"data/{self.feature_set}.csv", index_col="Unnamed: 0")
        df["TARGET"] = list(
            map(lambda current, future: 0 if current > future else 1, df.close, df.close.shift(-self.time_to_predict))
        )
        return df
