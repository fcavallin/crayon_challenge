from enum import Enum
from typing import List

from pydantic import BaseModel


class Request(BaseModel):
    emails: List[str]


class Response(BaseModel):
    predictions: List[str]


class Labels(Enum):
    debt_collection = 0
    credit_reporting = 1
    mortgages_and_loans = 2
    credit_card = 3
    retail_banking = 4
