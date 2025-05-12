from enum import Enum


class ColumnType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    TIME = "time"


class TreeType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
