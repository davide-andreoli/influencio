import pandas as pd
import numpy as np
from .enums import ColumnType
from .utils import determine_column_type
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self, min_rows: int = 10) -> None:
        self.min_rows = min_rows

    def validate_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Validates input data quality and structure.

        Args:
            X: Feature dataframe
            y: target series

        Raises:
            ValueError: if data does not meet the minimum requirements
        """

        if len(X) < 10:
            raise ValueError(
                f"The given dataset is too small: {len(X)}. Please use a dataset containing at least 10 rows"
            )

        if y.isnull().all():
            raise ValueError("The target variable contains only null values.")

        if y.isnull().sum() > len(y) * 0.5:
            logger.warning(
                f"Target variable has {y.isnull().sum()/len(y)*100:.1f}% missing values."
            )

        numeric_columns = X.select_dtypes(include=[np.number]).columns
        low_variance_columns = []

        for col in numeric_columns:
            if X[col].to_numpy().var() < 1e-10:
                low_variance_columns.append(col)

        if low_variance_columns:
            logger.warning(f"Low variance features detected: {low_variance_columns}")

        high_missing_columns = []
        for col in X.columns:
            missing_percent = X[col].isnull().sum() / len(X)
            if missing_percent > 0.8:
                high_missing_columns.append((col, missing_percent))

        if high_missing_columns:
            logger.warning(f"Features with >80% missing values: {high_missing_columns}")

        target_type = determine_column_type(y)
        if target_type == ColumnType.CATEGORICAL:
            unique_values = y.nunique()
            if unique_values < 2:
                raise ValueError(
                    f"Categorical target must have at least 2 classes. Found: {unique_values}"
                )
            if unique_values > 50:
                logger.warning(
                    f"High cardinality target: {unique_values} classes. Consider grouping to improve performances."
                )

        logger.info("Data validation completed successfully.")
