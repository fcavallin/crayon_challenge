from typing import List
from fastapi import FastAPI
from pandas import DataFrame
from transformers.pipelines.base import Dataset
from transformers.trainer_utils import PredictionOutput
from server_dataclasses import Response, Labels, Request
from transformer_model import TransformerModelClassification

import pandas as pd
import numpy as np


app = FastAPI()
classifier = TransformerModelClassification(5)
classifier.load('models/bert-tiny-finetuned')


@app.get('/')
def root():
    return {"Crayon Challenge": "Implementing an API REST"}


@app.post('/predict', response_model=Response)
def classify_email(request: Request) -> Response:
    pd_dataset: DataFrame = pd.DataFrame(request.emails, columns=['text'])
    tokenized_emails: Dataset = classifier.tokenize_dataset(pd_dataset=pd_dataset)
    response_predictions: PredictionOutput = classifier.predict(tokenized_emails)
    scores_list: np.ndarray = response_predictions.predictions
    best_scores: np.ndarray = np.argmax(scores_list, axis=1)
    predictions_labels: List[str] = [Labels(score).name for score in best_scores]
    responses: Response = Response(predictions=predictions_labels)
    return responses
