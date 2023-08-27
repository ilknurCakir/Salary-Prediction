from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, ValidationError


def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")


class CensusDataInputs(BaseModel):
    """
    Checks and validates data types
    Coerce conversion where it is possible, e.g. from int to str
    """

    age: int = Field(...)
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True


class MultipleCensusDataInputs(BaseModel):
    data: List[CensusDataInputs]


def fix_column_names(df):
    df = df.copy()
    df.columns = [col.replace(" ", "").replace("-", "_") for col in df.columns]
    df = df.replace("\\s+", "", regex=True)
    return df


def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    error = None
    validated_data = fix_column_names(df)

    try:
        MultipleCensusDataInputs(data=validated_data.to_dict(orient="records"))
    except ValidationError as err:
        error = err.json()

    return (validated_data, error)
