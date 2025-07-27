from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ForwardFillImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.ffill().values
        else:
            X_df = pd.DataFrame(X)
            return X_df.ffill().values


class BackwardFillImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.bfill().values
        else:
            X_df = pd.DataFrame(X)
            return X_df.bfill().values


class InterpolateImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.interpolate().values
        else:
            X_df = pd.DataFrame(X)
            return X_df.interpolate().values
