import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from .evaluator import ModelEvaluator, EvaluationResult
from .candidates import CLASSIFICATION_CANDIDATES, REGRESSION_CANDIDATES
from typing import Union, Optional, Tuple, Dict
from .preprocessor import Preprocessor
import logging

logger = logging.getLogger(__name__)


class ModelSelector:
    def __init__(
        self,
        evaluator: ModelEvaluator,
        preprocessor: Preprocessor,
        task: str,
        user_model: Optional[Union[ClassifierMixin, RegressorMixin]] = None,
        candidates: Optional[
            Dict[str, Tuple[Union[ClassifierMixin, RegressorMixin], Dict]]
        ] = None,
        tuning: bool = True,
    ):
        self.evaluator = evaluator
        self.preprocessor = preprocessor
        self.user_model = user_model
        self.candidates = candidates
        self.task = task
        self.tuning = tuning
        self.evaluation_results: Optional[list[EvaluationResult]] = None
        self.best_result: Optional[EvaluationResult] = None

    def select_best_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Union[ClassifierMixin, RegressorMixin]:
        """
        Selects the best model for the given target type (classification or regression) based on cross-validation scores between a set of candidate models and parameter grids.
        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.
            target_type (ColumnType): The type of the target variable (categorical or numerical).
            tuning (bool): Whether to perform hyperparameter tuning using RandomizedSearchCV.
        Returns:
            Union[ClassifierMixin, RegressorMixin]: The best model selected based on cross-validation scores and hyperparameter tuning.
        """
        if self.user_model is not None:
            logger.info("Using user provided model for prediction.")
            self.model_metrics = self.evaluator.evaluate_single_model(
                self.user_model, "user_provided_model", X, y
            )
            return self.user_model

        candidates = (
            (
                CLASSIFICATION_CANDIDATES
                if self.task == "classification"
                else REGRESSION_CANDIDATES
            )
            if not self.candidates
            else self.candidates
        )

        results = self.evaluator.evaluate_multiple_models(
            models=candidates,
            X=X,
            y=y,
            preprocessor=self.preprocessor,
            tune_hyperparameters=self.tuning,
        )

        self.evaluation_results = results
        self.best_result = self.evaluation_results[0]
        logger.info(f"Best model selected: {self.best_result}")
        return self.best_result.model
