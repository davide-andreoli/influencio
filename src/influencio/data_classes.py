from dataclasses import dataclass
from sklearn.base import BaseEstimator
import numpy as np
from typing import Dict, Any, Optional, List
from .enums import MetricType


@dataclass
class MetricConfig:
    """Configuration for a single metric"""

    metric_type: MetricType
    weight: float = 1.0
    higher_is_better: bool = True
    sklearn_name: Optional[str] = None

    def __post_init__(self):
        if self.sklearn_name is None:
            self.sklearn_name = self.metric_type.value


@dataclass
class ModelEvaluationResult:
    """Result of model evaluation containing scores and metadata"""

    model_name: str
    model: BaseEstimator
    primary_score: float
    all_scores: Dict[str, float]
    best_params: Dict[str, Any]
    weighted_score: float
    cv_scores: Dict[str, np.ndarray]
    std_scores: Dict[str, float]
    metric_configs: Optional[List[MetricConfig]] = None  # NEW FIELD
