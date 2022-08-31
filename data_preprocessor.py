from typing import Tuple
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from config import POSTPROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH


class DataPreprocessor:

    def __init__(self, data: DataFrame):
        self.preprocessed_data = self._preprocess_data(data=data)
        self.n_labels = self._count_label_nr()
        self._save_preprocessed_data(path=POSTPROCESSED_DATA_PATH)

    def _preprocess_data(self, data: DataFrame) -> DataFrame:
        data_to_preprocess: DataFrame = data.copy(deep=True)
        data_to_preprocess = self._drop_index_column(data_to_preprocess)
        # data_to_preprocess = self._drop_most_data(data_to_preprocess)
        data_to_preprocess = self._drop_lines_with_empty_content(data_to_preprocess)
        data_to_preprocess = self._rename_label_column(data_to_preprocess)
        data_to_preprocess = self._encode_classes(data_to_preprocess)
        data_to_preprocess = shuffle(data_to_preprocess)
        return data_to_preprocess

    @staticmethod
    def _drop_index_column(data: DataFrame) -> DataFrame:
        data.reset_index(drop=True, inplace=True)
        data.drop("Unnamed: 0", axis=1, inplace=True)
        return data

    @staticmethod
    def _drop_most_data(data: DataFrame, percentage: float = 0.01) -> DataFrame:
        data = shuffle(data)
        return data.head(int(len(data) * percentage))

    @staticmethod
    def _drop_lines_with_empty_content(data: DataFrame) -> DataFrame:
        data.dropna(inplace=True)
        return data

    @staticmethod
    def _rename_label_column(data: DataFrame) -> DataFrame:
        data.rename(columns={"product": "label", "narrative": "text"}, inplace=True)
        return data

    @staticmethod
    def _encode_classes(data: DataFrame) -> DataFrame:
        data['label'].replace(
            [
                'debt_collection',
                'credit_reporting',
                'mortgages_and_loans',
                'credit_card',
                'retail_banking'],
            [0, 1, 2, 3, 4]
            , inplace=True)
        return data

    def _save_preprocessed_data(self, path: str) -> None:
        self.preprocessed_data.to_csv(path, index=False)

    def get_preprocessed_data(self) -> DataFrame:
        return self.preprocessed_data

    def _count_label_nr(self) -> int:
        return len(self.preprocessed_data["label"].unique())

    def get_n_labels(self) -> int:
        return self.n_labels

    def split(self, test_size: float = 0.2) -> Tuple[DataFrame, DataFrame]:
        train_dataset, test_dataset = train_test_split(self.preprocessed_data, test_size=test_size, shuffle=True)
        train_dataset.to_csv(TRAIN_DATA_PATH, index=False)
        test_dataset.to_csv(TEST_DATA_PATH, index=False)
        return train_dataset, test_dataset
