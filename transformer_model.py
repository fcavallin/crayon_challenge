import numpy as np
import torch
from datasets import load_metric, Dataset
from pandas import DataFrame
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers.trainer_utils import PredictionOutput


class TransformerModelClassification:

    def __init__(self, num_labels: int):
        self.trainer = None
        self.is_trained = False
        self.tokenizer = AutoTokenizer.from_pretrained("models/huggingface/bert-tiny")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained("models/huggingface/bert-tiny",
                                                                        num_labels=num_labels)
        self.model.to(self.device)
        self._init_training_args()

    def _init_training_args(self):
        self.training_args = TrainingArguments(
            output_dir='output',
            overwrite_output_dir=True,
            num_train_epochs=20,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=0.001,
            weight_decay=0.01,
        )

    def _init_trainer(self, tokenized_dataset_train: Dataset) -> None:
        def compute_metrics(eval_pred):
            metric = load_metric("accuracy")
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=tokenized_dataset_train,
            compute_metrics=compute_metrics,
        )

    def tokenize_dataset(self, pd_dataset: DataFrame) -> Dataset:
        def tokenize_function(examples: DataFrame):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                max_length=256,
                truncation=True)

        dataset: Dataset = Dataset.from_pandas(pd_dataset)
        tokenized_dataset: Dataset = dataset.map(tokenize_function, batched=True)

        return tokenized_dataset

    def train(self, dataset: Dataset) -> None:
        if not self.trainer:
            self._init_trainer(tokenized_dataset_train=dataset)

        self.trainer.train()
        self.is_trained = True

    def predict(self, dataset: Dataset) -> PredictionOutput:
        if not self.is_trained or not self.trainer:
            raise ValueError("Model is not trained!")

        output: PredictionOutput = self.trainer.predict(dataset)

        return output
