import pandas as pd
from pandas import DataFrame


class DataReader:
    def __init__(self, data_path: str):
        self.data = DataReader._read_data(path=data_path)

    @staticmethod
    def _read_data(path: str) -> DataFrame:
        dataframe: DataFrame = pd.read_csv(path, index_col=False)
        return dataframe

    def get_data(self) -> DataFrame:
        return self.data
