from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from strictyaml import YAML, load

ROOT_PATH = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_PATH / "config.yml"


class Config(BaseModel):
    data_path: str = Field(...)
    model_path: str = Field(...)
    labelencoder_path: str = Field(...)
    dependent_feature: str = Field(...)
    features: List[str]
    categorical_vars: List[str]
    test_size: float
    random_state: int


def find_config_file():
    if CONFIG_PATH.is_file():
        return CONFIG_PATH
    else:
        raise Exception(f"Config not found at {CONFIG_PATH}")


def fetch_and_parse_config(config_file: Optional[Path] = None) -> YAML:
    if not config_file:
        config_file = find_config_file()

    with open(config_file, "r") as fileObj:
        _config = load(fileObj.read())

    return _config


def create_config():
    parsed_config = fetch_and_parse_config()
    config = Config(**parsed_config.data)

    return config


config = create_config()
