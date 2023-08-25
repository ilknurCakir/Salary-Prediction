import logging
import os
import unittest
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from model.config.core import config
from model.logger import create_logger
from model.model import create_pipeline, train_model
from model.preprocessing import load_dataset, save_pipeline
from model.run_training import process_and_train
from model.validation import validate_data
from tests.utils import create_original_bad_data, create_original_data

logger = logging.getLogger(__name__)
create_logger(logger, logging.INFO)


class TestLoadDataset(unittest.TestCase):
    def test_load_dataset(self):
        df = load_dataset()

        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertTrue(df.shape[0] > 0)
        self.assertTrue(df.shape[1] > 0)


class TestSavePipeline(unittest.TestCase):
    @patch("model.model.create_pipeline")
    @patch("model.preprocessing.joblib.dump")
    @patch("model.preprocessing.joblib.load")
    def test_save_pipeline(self, mock_pipeline, mock_joblib_dump, mock_joblib_load):
        runner = CliRunner(env={"LC_ALL": "en_US.utf8", "LANG": "en_US.utf-8"})
        with runner.isolated_filesystem():
            # create directory in isolated filesystem
            directory = "./mocked_directory/"
            model_filename = "model.pkl"
            os.mkdir(directory)
            path_to_save = os.path.join(directory, model_filename)

            save_pipeline(mock_pipeline, path_to_save)

            # check files in directory
            files = os.listdir(directory)
            self.assertTrue(len(files) > 0)
            self.assertTrue(model_filename in files)

            # test if joblib is called with correct arguments
            # with open(path_to_save, 'wb') as f:
            #   mock_joblib_dump.assert_called_with(mock_pipeline, f)


class TestCreatePipeline(unittest.TestCase):
    def test_create_pipeline(self):
        actual_pipeline = create_pipeline()

        # check if retuened pipeline is of type Pipeline (from sklearn)
        self.assertTrue(isinstance(actual_pipeline, Pipeline))

        # check names and types of transformers and estimator
        self.assertEqual(
            actual_pipeline.named_steps["estimator"].__class__.__name__, "SGDClassifier"
        )
        self.assertTrue(
            isinstance(
                actual_pipeline.named_steps["column_transformer"], ColumnTransformer
            )
        )
        self.assertEqual(
            actual_pipeline.named_steps["scaler"].__class__.__name__, "MaxAbsScaler"
        )


class TestValidateData(unittest.TestCase):
    def test_validate_data(self):
        X, y = create_original_data()
        validated_data, err = validate_data(X)
        expected_data = X.replace("\\s+", "", regex=True)

        self.assertEqual(err, None)
        self.assertEqual(set(validated_data.columns), set(config.features))
        self.assertTrue((validated_data.values == expected_data.values).all())

    def test_validate_bad_data(self):
        df = create_original_bad_data()
        validated_data, err = validate_data(df)

        self.assertFalse(err is None)

    def test_train_model_w_bad_data(self):
        df = create_original_bad_data()

        with self.assertRaises(KeyError):
            process_and_train(df)


class TestTrainModel(unittest.TestCase):
    def test_train_model(self):
        X, y = create_original_data()
        trained_pipeline = train_model(X, y)

        # check if trained_pipeline is Pipeline
        self.assertTrue(isinstance(trained_pipeline, Pipeline))

        # check if trained_pipeline is fitted
        try:
            check_is_fitted(trained_pipeline)
            logger.info("Pipeline is fitted: Success")

        except NotFittedError as err:
            logger.info("test_train_model: Fail")
            raise err
