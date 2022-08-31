import logging

from pandas import DataFrame

from data_preprocessor import DataPreprocessor
from data_reader import DataReader
from transformer_model import TransformerModelClassification

logging.basicConfig(level=logging.DEBUG)


def main():
    logging.debug("Started...")

    data_reader = DataReader(data_path="resources/complaints_processed.csv")

    data_orig: DataFrame = data_reader.get_data()

    preprocessor = DataPreprocessor(data=data_orig)

    n_labels: int = preprocessor.get_n_labels()

    preprocessed_data_train, preprocessed_data_test = preprocessor.split(test_size=0.1)

    model = TransformerModelClassification(n_labels)

    train_dataset = model.tokenize_dataset(pd_dataset=preprocessed_data_train)
    test_dataset = model.tokenize_dataset(pd_dataset=preprocessed_data_test)

    model.train(dataset=train_dataset)

    result = model.predict(dataset=test_dataset)

    logging.debug(result.metrics)

    logging.debug(f"Saving model")

    model.save('models/bert-tiny-finetuned')

    logging.debug("End...")


if __name__ == "__main__":
    main()
