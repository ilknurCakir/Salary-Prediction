import requests
import pandas as pd
import numpy as np

test_data = pd.DataFrame(
    [
        {
            "age": 47,
            "workclass": "Private",
            "fnlgt": 2234,
            "education": "College",
            "education_num": 17,
            "marital_status": "Married",
            "occupation": "Doctor",
            "relationship": "Wife",
            "race": "Turkish",
            "sex": "Female",
            "capital_gain": 1000,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United States",
        }
    ]
)

payload = {"data": test_data.replace({np.nan: None}).to_dict(orient="records")}

# data should be payload
r = requests.post("https://ilknurcakir-salary-prediction.onrender.com/predict", json=payload)

status_code = r.status_code
inference_result = r.json()

print('Live post to https://ilknurcakir-salary-prediction.onrender.com/predict')
print('------------------------------------------------------------------------')
print(f'Live Post Status Code: {status_code}')
print(f'Love Post Inference Result: {inference_result}')
