import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from model.config.core import config
from model.logger import create_logger
from model.model import compute_model_metrics, train_model, compute_metrics_on_slices
from model.preprocessing import load_dataset, save_labelencoder, save_pipeline
from model.validation import fix_column_names, validate_data

logger = logging.getLogger(__name__)
create_logger(logger, logging.INFO)


def process_and_train(df: pd.DataFrame = None):
    if df is None:
        # extract data
        df = load_dataset()
        df = fix_column_names(df)

    X = df[config.features]
    y = df[config.dependent_feature]

    # data validation
    X, err = validate_data(X)
    if err:
        logger.info(f"Error in data validation " f"error message: {err}")

        raise err

    # convert strings in dependent variable to 0 and 1
    le = LabelEncoder()
    y = le.fit_transform(y)

    logger.info("Splitting data into test and train datasets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    logger.info(f"Training data has {len(X_train)} rows. Training the model..")
    model = train_model(X_train, y_train)

    # model metrics
    y_test_preds = model.predict(X_test)
    precision, recall, f1 = compute_model_metrics(y_test, y_test_preds)
    logger.info(
        f"Calculated on {len(y_test)} rows. Metrics with test data: "
        f"precision: {precision:.5f}, recall: {recall:.5f}, f1: {f1:.5f}"
    )

    compute_metrics_on_slices(df, X_test, y_test, y_test_preds, 'education')

    # saving model
    save_pipeline(model)

    # saving label encoder
    save_labelencoder(le)


if __name__ == "__main__":
    process_and_train()
