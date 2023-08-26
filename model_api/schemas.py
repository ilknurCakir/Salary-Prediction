from typing import Any, List, Optional

from pydantic import BaseModel, Field

from model.validation import CensusDataInputs


class Health(BaseModel):
    api_version: str = Field(...)
    model_version: str = Field(...)


class PredictionResults(BaseModel):
    predictions: Optional[List[str]]
    errors: Optional[Any]


class MultipleCensusDataInputs(BaseModel):
    data: List[CensusDataInputs]

    # add data example here
    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {
                        "age": 34,
                        "workclass": "Private",
                        "fnlgt": 24587,
                        "education": "College",
                        "education_num": 12,
                        "marital_status": "Married",
                        "occupation": "Doctor",
                        "relationship": "Husband",
                        "race": "White",
                        "sex": "Male",
                        "capital_gain": 10,
                        "capital_loss": 0,
                        "hours_per_week": 40,
                        "native_country": "United States",
                    }
                ]
            }
        }
