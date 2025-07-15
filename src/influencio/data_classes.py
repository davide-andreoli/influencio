from dataclasses import dataclass
from sklearn.base import ClassifierMixin, RegressorMixin
import numpy as np
from typing import Dict, Any, Optional, List, Union
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

        if self.sklearn_name.startswith("neg_"):
            self.higher_is_better = False

    def handle_negative_scores(self, raw_score: float) -> float:
        """Convert sklearn score to consistent 'higher is better' format"""
        if not self.sklearn_name:
            return raw_score
        if self.sklearn_name.startswith("neg_"):
            return -raw_score
        return raw_score

    def get_display_score(self, raw_score: float) -> float:
        """Get score for display purposes (original metric interpretation)"""
        if not self.sklearn_name:
            return raw_score
        if self.sklearn_name.startswith("neg_"):
            return -raw_score
        return raw_score


@dataclass
class ModelEvaluationResult:
    """Result of model evaluation containing scores and metadata"""

    model_name: str
    model: Union[ClassifierMixin, RegressorMixin]
    primary_score: np.float64
    all_scores: Dict[str, np.float64]
    best_params: Dict[str, Any]
    weighted_score: float
    cv_scores: Dict[str, np.ndarray]
    std_scores: Dict[str, np.float64]
    metric_configs: Optional[List[MetricConfig]] = None  # NEW FIELD
