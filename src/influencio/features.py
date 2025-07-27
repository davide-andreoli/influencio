from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts temporal features from datetime columns.
    """

    def __init__(
        self,
        extract_year: bool = True,
        extract_month: bool = True,
        extract_day: bool = True,
        extract_dayofweek: bool = True,
        extract_hour: bool = True,
        extract_minute: bool = False,
        extract_quarter: bool = True,
        extract_is_weekend: bool = True,
        extract_is_month_end: bool = False,
        extract_is_month_start: bool = False,
        cyclical_encoding: bool = True,
        drop_original: bool = True,
    ):
        self.extract_year = extract_year
        self.extract_month = extract_month
        self.extract_day = extract_day
        self.extract_dayofweek = extract_dayofweek
        self.extract_hour = extract_hour
        self.extract_minute = extract_minute
        self.extract_quarter = extract_quarter
        self.extract_is_weekend = extract_is_weekend
        self.extract_is_month_end = extract_is_month_end
        self.extract_is_month_start = extract_is_month_start
        self.cyclical_encoding = cyclical_encoding
        self.drop_original = drop_original

        self.feature_names_out_ = None
        self.input_features_ = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit the transformer."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.input_features_ = list(X.columns)

        self.feature_names_out_ = self._get_feature_names_out(X)

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Transform datetime columns to temporal features."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.input_features_)
        else:
            X = X.copy()

        for col in X.columns:
            if self.extract_year:
                X[f"{col}_year"] = X[col].dt.year

            if self.extract_month:
                X[f"{col}_month"] = X[col].dt.month
                if self.cyclical_encoding:
                    X[f"{col}_month_sin"] = np.sin(2 * np.pi * X[f"{col}_month"] / 12)
                    X[f"{col}_month_cos"] = np.cos(2 * np.pi * X[f"{col}_month"] / 12)

            if self.extract_day:
                X[f"{col}_day"] = X[col].dt.day
                if self.cyclical_encoding:
                    X[f"{col}_day_sin"] = np.sin(2 * np.pi * X[f"{col}_day"] / 31)
                    X[f"{col}_day_cos"] = np.cos(2 * np.pi * X[f"{col}_day"] / 31)

            if self.extract_dayofweek:
                X[f"{col}_dayofweek"] = X[col].dt.dayofweek
                if self.cyclical_encoding:
                    X[f"{col}_dayofweek_sin"] = np.sin(
                        2 * np.pi * X[f"{col}_dayofweek"] / 7
                    )
                    X[f"{col}_dayofweek_cos"] = np.cos(
                        2 * np.pi * X[f"{col}_dayofweek"] / 7
                    )

            if self.extract_hour:
                X[f"{col}_hour"] = X[col].dt.hour
                if self.cyclical_encoding:
                    X[f"{col}_hour_sin"] = np.sin(2 * np.pi * X[f"{col}_hour"] / 24)
                    X[f"{col}_hour_cos"] = np.cos(2 * np.pi * X[f"{col}_hour"] / 24)

            if self.extract_minute:
                X[f"{col}_minute"] = X[col].dt.minute
                if self.cyclical_encoding:
                    X[f"{col}_minute_sin"] = np.sin(2 * np.pi * X[f"{col}_minute"] / 60)
                    X[f"{col}_minute_cos"] = np.cos(2 * np.pi * X[f"{col}_minute"] / 60)

            if self.extract_quarter:
                X[f"{col}_quarter"] = X[col].dt.quarter

            if self.extract_is_weekend:
                X[f"{col}_is_weekend"] = (X[col].dt.dayofweek >= 5).astype(int)

            if self.extract_is_month_end:
                X[f"{col}_is_month_end"] = X[col].dt.is_month_end.astype(int)

            if self.extract_is_month_start:
                X[f"{col}_is_month_start"] = X[col].dt.is_month_start.astype(int)

            if self.drop_original:
                X = X.drop(columns=[col])

        return X

    def _get_feature_names_out(self, X: pd.DataFrame) -> List[str]:
        """Get output feature names."""
        output_features = []

        for col in X.columns:
            if not self.drop_original:
                output_features.append(col)

            if self.extract_year:
                output_features.append(f"{col}_year")
            if self.extract_month:
                output_features.append(f"{col}_month")
                if self.cyclical_encoding:
                    output_features.extend([f"{col}_month_sin", f"{col}_month_cos"])
            if self.extract_day:
                output_features.append(f"{col}_day")
                if self.cyclical_encoding:
                    output_features.extend([f"{col}_day_sin", f"{col}_day_cos"])
            if self.extract_dayofweek:
                output_features.append(f"{col}_dayofweek")
                if self.cyclical_encoding:
                    output_features.extend(
                        [f"{col}_dayofweek_sin", f"{col}_dayofweek_cos"]
                    )
            if self.extract_hour:
                output_features.append(f"{col}_hour")
                if self.cyclical_encoding:
                    output_features.extend([f"{col}_hour_sin", f"{col}_hour_cos"])
            if self.extract_minute:
                output_features.append(f"{col}_minute")
                if self.cyclical_encoding:
                    output_features.extend([f"{col}_minute_sin", f"{col}_minute_cos"])
            if self.extract_quarter:
                output_features.append(f"{col}_quarter")
            if self.extract_is_weekend:
                output_features.append(f"{col}_is_weekend")
            if self.extract_is_month_end:
                output_features.append(f"{col}_is_month_end")
            if self.extract_is_month_start:
                output_features.append(f"{col}_is_month_start")

        return output_features

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if self.feature_names_out_ is None:
            raise RuntimeError("Transformer has not been fitted yet.")
        return np.array(self.feature_names_out_)


class DatetimeOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Converts datetime columns to ordinal values (days/seconds since reference).
    """

    def __init__(self, unit="days", reference_date=None):
        """
        Parameters:
        - unit: 'days' or 'seconds' for the time unit
        - reference_date: Reference date for ordinal encoding (default: min date)
        """
        self.unit = unit
        self.reference_date = reference_date
        self.reference_dates_ = {}
        self.datetime_columns_ = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """Fit the encoder."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for col in X.columns:
            col_data = X[col].copy()
            if self.reference_date is not None:
                self.reference_dates_[col] = pd.to_datetime(self.reference_date)
            else:
                self.reference_dates_[col] = col_data.min()

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Transform datetime columns to ordinal values."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        else:
            X = X.copy()

        for col in X.columns:
            col_data = X[col].copy()

            time_diff = col_data - self.reference_dates_[col]

            if self.unit == "days":
                X[col] = time_diff.dt.total_seconds() / (24 * 3600)
            elif self.unit == "seconds":
                X[col] = time_diff.dt.total_seconds()
            else:
                raise ValueError(
                    f"Unit '{self.unit}' not supported. Use 'days' or 'seconds'."
                )

        return X
