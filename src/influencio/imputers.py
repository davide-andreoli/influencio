import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ForwardFillImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.ffill()
        else:
            X_df = pd.DataFrame(X)
            return X_df.ffill().values
