from enum import StrEnum


class DataCharacteristic(StrEnum):
    """Enum for different data characteristics that influence metric selection"""

    BALANCED = "balanced"
    IMBALANCED = "imbalanced"
    MULTICLASS = "multiclass"
    BINARY = "binary"
    HIGH_DIMENSIONAL = "high_dimensional"
    SMALL_SAMPLE = "small_sample"


class MetricType(StrEnum):
    """Enum for metric types"""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1_macro"
    ROC_AUC = "roc_auc_ovr"
    BALANCED_ACCURACY = "balanced_accuracy"
    MATTHEWS_CORR = "matthews_corrcoef"
    LOG_LOSS = "neg_log_loss"
    # Regression metrics
    R2 = "r2"
    MSE = "neg_mean_squared_error"
    MAE = "neg_mean_absolute_error"
    MAPE = "neg_mean_absolute_percentage_error"
    EXPLAINED_VARIANCE = "explained_variance"


class ColumnType(StrEnum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    TIME = "time"


class TreeType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
