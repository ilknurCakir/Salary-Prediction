import json
import logging
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse

from model import __version__ as model_version
from model.config.core import config
from model.logger import create_logger
from model.model import inference
from model_api import __version__ as api_version
from model_api import schemas


logger = logging.getLogger(__name__)
create_logger(logger, logging.INFO)

app = FastAPI()

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
async def get_health(request: Request) -> dict:
    return_val = {"api_version": api_version, "model_version": model_version}

    return return_val


@api_router.get("/", status_code=200)
async def index(request: Request) -> dict:
    return_val = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API - Predicting if person makes greater than 50K</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(return_val)


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleCensusDataInputs) -> Any:
    input_df = pd.DataFrame(jsonable_encoder(input_data.data))

    logger.info(f"Making inference on data {input_data.data}")
    result = inference(input_df)

    # if results["errors"] is not None:
    #     logger.warning(f"Prediction validation error: {results.get('errors')}")
    #     raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    if result["errors"]:
        logger.info(
            f"data {input_data.data} gives error. Error message: " f'{result["errors"]}'
        )
        raise HTTPException(status_code=400, detail=json.loads(result["errors"]))

    # inverse tranform y values
    with open(config.labelencoder_path, "rb") as f:
        le = joblib.load(f)

    result["predictions"] = list(le.inverse_transform(result["predictions"]))
    logger.info(f'result is {result["predictions"]}')
    return result


app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="localhost", port=8001)
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
