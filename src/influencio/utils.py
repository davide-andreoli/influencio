import pandas as pd
from .enums import ColumnType


def determine_column_type(
    column: pd.Series, cardinality_threshold: int = 10
) -> ColumnType:
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
            and column.nunique() <= cardinality_threshold
        )
    ):
        return ColumnType.CATEGORICAL
    elif pd.api.types.is_datetime64_any_dtype(column):
        return ColumnType.TIME
    elif pd.api.types.is_numeric_dtype(column):
        return ColumnType.NUMERICAL
    else:
        raise ValueError(f"Unhandled column type: {column.dtype}")
