import unittest

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import app
from model import __version__ as model_version
from model_api import __version__ as api_version
from model_api import schemas

# Instantiate the testing client with our app
client = TestClient(app)


class TestApiLocallyGetRoot(unittest.TestCase):
    def test_api_locally_get_root(self):
        r = client.get("/")

        self.assertTrue(r.status_code == 200)
        self.assertTrue(isinstance(r.text, str))


class TestApiHealth(unittest.TestCase):
    def test_api_health(self):
        r = client.get("/health")
        self.assertTrue(r.status_code == 200)

        r_text = r.json()
        self.assertTrue(r_text.get("model_version", None) == model_version)
        self.assertTrue(r_text.get("api_version", None) == api_version)

        try:
            schemas.Health(**r_text)
        except ValidationError as err:
            self.fail(f"Health response raised {err}, it is not Health schema")


class TestApiPredict(unittest.TestCase):
    def test_api_predict(self):
        # we can load the data below
        test_data = pd.DataFrame(
            [
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
                    "native_country": "Malaysia",
                }
            ]
        )

        payload = {"data": test_data.replace({np.nan: None}).to_dict(orient="records")}

        # data should be payload
        r = client.post("/predict", json=payload)

        self.assertTrue(r.status_code == 200)

        r_text = r.json()

        try:
            schemas.PredictionResults(**r_text)
        except ValidationError as err:
            self.fail(
                f"Prediction response raised {err}, it is not "
                f"PredictionResults schema"
            )

        self.assertTrue(r_text["errors"] is None)
        expected_result = ["<=50K"]

        self.assertTrue(r_text["predictions"] == expected_result)

    def test_api_predict_w_bad_data(self):
        '''
        Tests the response when one of the columns has string value
        even though expected value is of integer type.
        '''

        # we can load the data below
        test_data = pd.DataFrame(
            [
                {
                    "age": 34,
                    "workclass": "Private",
                    "fnlgt": 'some string here',
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
                    "native_country": "Malaysia",
                }
            ]
        )

        payload = {"data": test_data.replace({np.nan: None}).to_dict(orient="records")}

        # data should be payload
        r = client.post("/predict", json=payload)

        # it should not return 200 status code
        self.assertTrue(r.status_code != 200)
        r_text = r.json()

        # it should not return a response with PredictionResults schema
        try:
            schemas.PredictionResults(**r_text)
            self.fail('Service returns response of PredictionResults'
                      'schema with bad data')
        except Exception:
            pass

    def test_api_predict_w_bad_column_name(self):
        '''
        Tests response when the data column name is with hyphen instead
        of underscore.
        '''

        # we can load the data below
        test_data = pd.DataFrame(
            [
                {
                    "age": 34,
                    "workclass": "Private",
                    "fnlgt": 1234,
                    "education": "College",
                    "education-num": 12,
                    "marital_status": "Married",
                    "occupation": "Doctor",
                    "relationship": "Husband",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 10,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "Malaysia",
                }
            ]
        )

        payload = {"data": test_data.replace({np.nan: None}).to_dict(orient="records")}

        # data should be payload
        r = client.post("/predict", json=payload)

        # it should not return 200 status code
        self.assertTrue(r.status_code != 200)
        r_text = r.json()

        # it should not return a response wirh PredictionResults schema
        try:
            schemas.PredictionResults(**r_text)
            self.fail('Service returns response of PredictionResults'
                      'schema with bad data')
        except Exception:
            pass
