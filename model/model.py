# imports

import logging
from typing import Union

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder

from model.config.core import config
from model.logger import create_logger
from model.preprocessing import load_model
from model.transformers import CategoricalEncoder
from model.validation import validate_data

logger = logging.getLogger(__name__)
create_logger(logger, logging.INFO)


def create_pipeline():
    cat_transformer = Pipeline(
        steps=[
            ("categorical_encoder", CategoricalEncoder(config.categorical_vars)),
            ("onehot_encoder", OneHotEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat_transformer", cat_transformer, config.categorical_vars),
        ],
        remainder="passthrough",
    )

    pipe = Pipeline(
        steps=[
            ("column_transformer", preprocessor),
            ("scaler", MaxAbsScaler()),
            (
                "estimator",
                SGDClassifier(
                    loss="modified_huber",
                    shuffle=True,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    return pipe


def train_model(X_train, y_train):
    pipe = create_pipeline()
    logger.info("Fitting the model..")

    pipe.fit(X_train, y_train)

    return pipe


def compute_model_metrics(y, preds):
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)

    return precision, recall, fbeta


def inference(X: Union[pd.DataFrame, dict]) -> dict:
    X = pd.DataFrame(X)

    validated_data, err = validate_data(X)
    result = {"predictions": None, "errors": err}

    if err is None:
        model = load_model()
        preds = model.predict(validated_data[config.features])
        result["predictions"] = preds

    return result
