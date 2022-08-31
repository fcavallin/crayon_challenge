from enum import Enum
from pydantic import BaseModel


class Email(BaseModel):
    text: str


class Response(BaseModel):
    predictions: str


class Labels(Enum):
    debt_collection = 0
    credit_reporting = 1
    mortgages_and_loans = 2
    credit_card = 3
    retail_banking = 4
