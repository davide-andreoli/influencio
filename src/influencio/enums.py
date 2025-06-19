from enum import StrEnum


class ColumnType(StrEnum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    TIME = "time"


class TreeType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
