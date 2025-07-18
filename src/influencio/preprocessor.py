from .enums import ColumnType
from typing import List, Tuple, Optional, Dict, Any, Union, cast
import pandas as pd
import numpy as np
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from .utils import determine_column_type
from .imputers import ForwardFillImputer
import logging

logger = logging.getLogger(__name__)


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cardinality_threshold: int = 10,
        missing_threshold: float = 0.5,
        scale_numeric: bool = True,
    ) -> None:
        self.cardinality_threshold = cardinality_threshold
        self.scale_numeric = scale_numeric
        self.missing_threshold = missing_threshold
        self.preprocessor: Optional[ColumnTransformer] = None

    def _get_missing_value_imputer(
        self, column: pd.Series
    ) -> Union[str, TransformerMixin, None]:
        column_type = determine_column_type(column, self.cardinality_threshold)
        missing_percent = column.isnull().sum() / len(column) * 100

        if missing_percent == 0:
            return None  # no imputing

        if missing_percent > self.missing_threshold * 100:
            return "drop_column"

        if column_type == ColumnType.CATEGORICAL:
            unique_ratio = column.nunique() / len(column.dropna())
            if unique_ratio > 0.9:
                return "drop_column"
            elif missing_percent < 5:
                return SimpleImputer(strategy="most_frequent")
            else:
                return SimpleImputer(strategy="constant", fill_value="missing")

        elif column_type == ColumnType.NUMERICAL:
            if missing_percent < 5:
                return SimpleImputer(strategy="mean")
            elif missing_percent < 20:
                return SimpleImputer(strategy="median")
            else:
                return KNNImputer(n_neighbors=3)

        elif column_type == ColumnType.TIME:
            if missing_percent < 5:
                if (
                    column.dropna().is_monotonic_increasing
                    or column.dropna().is_monotonic_decreasing
                ):
                    return ForwardFillImputer()
                elif (
                    column.dropna().mode().count() > 0
                    and (
                        column.dropna().mode().value_counts().iloc[0]
                        / len(column.dropna())
                    )
                    > 0.5
                ):
                    return SimpleImputer(strategy="most_frequent")
                else:
                    return "drop_column"
            else:
                return "drop_column"

    def _get_encoder(self, column: pd.Series) -> Optional[TransformerMixin]:
        column_type = determine_column_type(column, self.cardinality_threshold)
        if column_type == ColumnType.CATEGORICAL:
            if column.nunique() <= self.cardinality_threshold:
                return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                return OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
        return None

    def _get_scaler(self, column: pd.Series) -> Optional[TransformerMixin]:
        if (
            determine_column_type(column, self.cardinality_threshold)
            == ColumnType.NUMERICAL
            and self.scale_numeric
        ):
            return StandardScaler()
        return None

    def _build_transformers(
        self, X: pd.DataFrame
    ) -> List[Tuple[str, Pipeline, List[str]]]:
        transformers_dict: Dict[str, Tuple[Any, List[str]]] = {}

        for column in X.columns:
            col_series = cast(pd.Series, X[column])
            imputer = self._get_missing_value_imputer(col_series)
            encoder = self._get_encoder(col_series)
            scaler = self._get_scaler(col_series)

            if imputer == "drop_column":
                if "drop" not in transformers_dict:
                    transformers_dict["drop"] = ("drop", [column])
                else:
                    transformers_dict["drop"][1].append(column)
                continue

            steps = []
            if imputer is not None:
                steps.append(("imputer", imputer))
            if encoder is not None:
                steps.append(("encoder", encoder))
            if scaler is not None:
                steps.append(("scaler", scaler))

            if not steps:
                continue

            pipeline = Pipeline(steps)
            pipeline_key = str(pipeline)

            if pipeline_key not in transformers_dict:
                transformers_dict[pipeline_key] = (pipeline, [column])
            else:
                transformers_dict[pipeline_key][1].append(column)

        transformers = []
        for idx, (key, (transformer, cols)) in enumerate(transformers_dict.items()):
            name = f"pipeline_{idx}" if key != "drop" else "drop_columns"
            transformers.append((name, transformer, cols))

        return transformers

    def fit(self, X: Union[pd.DataFrame, np.ndarray, spmatrix], y=None):
        """
        Fit the preprocessor using direct transformer grouping.
        """

        transformers = self._build_transformers(cast(pd.DataFrame, X))

        # Create the final ColumnTransformer
        if transformers:
            self.preprocessor = ColumnTransformer(transformers)
            self.preprocessor.fit(X)
        else:
            raise ValueError("No columns to transform after preprocessing analysis")

        return self

    def transform(self, X: pd.DataFrame) -> Union[np.ndarray, spmatrix]:
        """Transform the input data using the fitted preprocessor."""
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.preprocessor.transform(X)  # pyright: ignore[reportReturnType]

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray, spmatrix], y=None, **fit_params
    ) -> np.ndarray:
        """Fit and transform the input data."""
        return cast(np.ndarray, self.fit(X, y).transform(cast(pd.DataFrame, X)))

    def get_transformer(self) -> ColumnTransformer:
        """Get the fitted ColumnTransformer."""
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.preprocessor

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Returns the output feature names after transformation.

        Args:
            input_features (Optional[List[str]]): Optional list of original input feature names.

        Returns:
            np.ndarray: Array of transformed feature names.
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.preprocessor.get_feature_names_out(input_features)
