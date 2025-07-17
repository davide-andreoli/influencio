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

    def _determine_column_type(self, column: pd.Series) -> ColumnType:
        """
        Determines the type of a column based on its data type and unique values.
        Args:
            column (pd.Series): The input column to determine the type
        Returns:
            ColumnType: The determined type of the column (categorical, numerical, or time)
        """
        if (
            pd.api.types.is_bool_dtype(column)
            or pd.api.types.is_object_dtype(column)
            or (
                pd.api.types.is_numeric_dtype(column)
                and column.nunique() <= self.cardinality_threshold
            )
        ):
            return ColumnType.CATEGORICAL
        elif pd.api.types.is_datetime64_any_dtype(column):
            return ColumnType.TIME
        elif pd.api.types.is_numeric_dtype(column):
            return ColumnType.NUMERICAL
        else:
            raise ValueError(f"Unhandled column type: {column.dtype}")

    def _get_missing_value_imputer(self, column: pd.Series) -> str:
        column_type = self._determine_column_type(column)
        missing_count = column.isnull().sum()
        missing_percent = missing_count / len(column) * 100

        if missing_percent == 0:
            return "none"

        if missing_percent > self.missing_threshold * 100:
            return "drop_column"

        if column_type == ColumnType.CATEGORICAL:
            unique_ratio = column.nunique() / len(column.dropna())

            if unique_ratio > 0.9:  # High cardinality
                return "drop_column"
            elif missing_percent < 5:
                return "mode"
            else:
                return "constant"  # Use "missing" as category

        elif column_type == ColumnType.NUMERICAL:
            if missing_percent < 5:
                return "mean"
            elif missing_percent < 20:
                return "median"
            else:
                return "knn"  # More sophisticated for high missing

        else:  # TIME
            if missing_percent < 5:
                return "forward_fill"
            else:
                return "drop_column"

    def _get_scaler(self, column: pd.Series) -> str:
        column_type = self._determine_column_type(column)

        if column_type == ColumnType.NUMERICAL and self.scale_numeric:
            return "standard"
        else:
            return "none"

    def _get_encoder(self, column: pd.Series) -> str:
        column_type = self._determine_column_type(column)

        if column_type == ColumnType.CATEGORICAL:
            if column.nunique() <= self.cardinality_threshold:
                return "one-hot"
            else:
                return "ordinal"
        else:
            return "none"

    def _build_transformers(
        self, X: pd.DataFrame
    ) -> List[Tuple[str, Pipeline, List[str]]]:
        transformers_dict: Dict[str, Tuple[Any, List[str]]] = {}

        for column in X.columns:
            imputer_strategy = self._get_missing_value_imputer(
                cast(pd.Series, X[column])
            )
            encoder_strategy = self._get_encoder(cast(pd.Series, X[column]))
            scaler_strategy = self._get_scaler(cast(pd.Series, X[column]))

            if imputer_strategy == "drop_column":
                pipeline_key = "drop"
                if pipeline_key not in transformers_dict:
                    transformers_dict[pipeline_key] = ("drop", [column])
                else:
                    transformers_dict[pipeline_key][1].append(column)
                continue

            steps = []

            if imputer_strategy == "mean":
                steps.append(("imputer", SimpleImputer(strategy="mean")))
            elif imputer_strategy == "median":
                steps.append(("imputer", SimpleImputer(strategy="median")))
            elif imputer_strategy == "mode":
                steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            elif imputer_strategy == "constant":
                steps.append(
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    )
                )
            elif imputer_strategy == "knn":
                steps.append(("imputer", KNNImputer(n_neighbors=3)))
            elif imputer_strategy == "forward_fill":
                steps.append(
                    ("imputer", SimpleImputer(strategy="constant", fill_value=np.nan))
                )

            if encoder_strategy == "one-hot":
                steps.append(
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    )
                )
            elif encoder_strategy == "ordinal":
                steps.append(
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    )
                )

            if scaler_strategy == "standard":
                steps.append(("scaler", StandardScaler()))

            pipeline = Pipeline(steps)
            pipeline_key = str(steps)

            if pipeline_key not in transformers_dict:
                transformers_dict[pipeline_key] = (pipeline, [column])
            else:
                transformers_dict[pipeline_key][1].append(column)

        transformers = []
        for idx, (key, (transformer, columns)) in enumerate(transformers_dict.items()):
            name = f"pipeline_{idx}" if key != "drop" else "drop_columns"
            transformers.append((name, transformer, columns))

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
