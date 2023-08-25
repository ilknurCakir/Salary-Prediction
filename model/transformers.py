from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols: Union[str, List]):
        if isinstance(cat_cols, str):
            cat_cols = [cat_cols]
        self.cat_cols = cat_cols

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None):
        self.mapping = {}
        for col in self.cat_cols:
            unique_vals = (
                X[col].value_counts().sort_values(ascending=False).index.values.tolist()
            )
            unique_vals = unique_vals + ["unseen"]
            self.mapping[col] = {val: i for val, i in enumerate(unique_vals)}

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]):
        X = X.copy()

        for col in self.cat_cols:
            n_keys = len(self.mapping[col].keys())
            X[col] = X[col].apply(lambda x: self.mapping[col].get(x, n_keys - 1))
            # it maps all unseen/new categories to 'unseen' category in each column
            # that's the reason why it is n_keys-1

        return X
