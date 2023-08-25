import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from model.config.core import config
from model.logger import create_logger

logger = logging.getLogger(__name__)
create_logger(logger, logging.INFO)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(config.data_path)

    return df


def save_pipeline(
    pipeline_to_persist: Pipeline, path_to_save: Path = config.model_path
) -> None:
    with open(path_to_save, "wb") as f:
        joblib.dump(pipeline_to_persist, f)

    logger.info(f"Pipeline is saved to {path_to_save}")


def save_labelencoder(
    labelencoder_to_persist: LabelEncoder, path_to_save: Path = config.labelencoder_path
) -> None:
    with open(path_to_save, "wb") as f:
        joblib.dump(labelencoder_to_persist, f)

    logger.info(f"LabelEncoder is saved to {path_to_save}")


def load_model(model_path: Path = config.model_path) -> Pipeline:
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    return model
