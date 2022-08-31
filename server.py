from typing import List
from fastapi import FastAPI
from pandas import DataFrame
import pandas as pd
from transformers.pipelines.base import Dataset
from transformers.trainer_utils import PredictionOutput

from server_dataclasses import Response, Email, Labels
from transformer_model import TransformerModelClassification

app = FastAPI()
classifier = TransformerModelClassification(5)
classifier.load('models/bert-tiny-finetuned')


@app.get('/')
def root():
    return {"Crayon Challenge": "Implementing an API REST"}


@app.post('/predict', response_model=Response)
def classify_email(email: Email) -> Response:
    pd_dataset: DataFrame = pd.DataFrame([email.text], columns=['text'])
    tokenized_email: Dataset = classifier.tokenize_dataset(pd_dataset=pd_dataset)
    response_predictions: PredictionOutput = classifier.predict(tokenized_email)
    scores_list: List = response_predictions.predictions.tolist()[0]
    best_scores: int = scores_list.index(max(scores_list))
    responses: Response = Response(predictions=str(Labels(best_scores).name))
    return responses
