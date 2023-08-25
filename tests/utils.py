import pandas as pd

from model.config.core import config


def create_original_data():
    data = [
        [
            39,
            " State-gov",
            77516,
            " Bachelors",
            13,
            " Never-married",
            "Adm-clerical",
            " Not-in-family",
            " White",
            " Male",
            2174,
            0,
            40,
            " United-States",
        ],
        [
            28,
            " Private",
            338409,
            " Bachelors",
            13,
            " Married-civ-spouse",
            " Prof-specialty",
            " Wife",
            " Black",
            " Female",
            0,
            0,
            40,
            " Cuba",
        ],
        [
            36,
            " Private",
            102864,
            "HS-grad",
            9,
            " Never-married",
            " Machine-op-inspct",
            " Own-child",
            " White",
            " Female",
            0,
            0,
            40,
            " United-States",
        ],
    ]

    X = pd.DataFrame(data, columns=config.features)
    y = pd.Series([0, 0, 1])

    return X, y


def create_original_bad_data():
    data = [
        [
            39,
            " State-gov",
            77516,
            " Bachelors",
            13,
            " Never-married",
            "Adm-clerical",
            " Not-in-family",
            " White",
            " Male",
            2174,
            0,
            40,
            " United-States",
        ],
        [
            28,
            " Private",
            338409,
            " Bachelors",
            13,
            " Married-civ-spouse",
            " Prof-specialty",
            " Wife",
            " Black",
            " Female",
            0,
            0,
            40,
            " Cuba",
        ],
        [
            36,
            " Private",
            102864,
            "HS-grad",
            9,
            " Never-married",
            " Machine-op-inspct",
            " Own-child",
            " White",
            " Female",
            0,
            0,
            40,
            " United-States",
        ],
    ]

    # beware of 'wrokclass' instead of 'workclass'
    columns = [
        "age",
        "wrokclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        " capital-gain",
        " capital-loss",
        " hours-per-week",
        " native-country",
    ]

    X = pd.DataFrame(data, columns=columns)

    return X
