from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Literal, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.base import ClassifierMixin, RegressorMixin
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Simple container for evaluation results"""

    model_name: str
    model: Union[ClassifierMixin, RegressorMixin]
    best_params: Dict[str, Any]
    scores: Dict[str, float]
    score_stds: Dict[str, float]
    primary_score: float


class ModelEvaluator:
    DEFAULT_CLASSIFICATION_METRICS = ["accuracy", "f1_weighted", "roc_auc_ovr"]
    DEFAULT_REGRESSION_METRICS = [
        "r2",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
    ]

    def __init__(
        self,
        task_type: Literal["classification", "regression"],
        metrics: Optional[List[str]] = None,
        primary_metric: Optional[str] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        n_iter: int = 10,
    ):
        """
        Initialize the evaluator.

        Args:
            task_type: Type of ML task
            metrics: List of sklearn metric names to evaluate
            primary_metric: Main metric for model selection (first metric if None)
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            n_iter: Number of iterations for hyperparameter search
        """
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_iter = n_iter

        # Set default metrics if none provided
        if metrics is None:
            self.metrics = (
                self.DEFAULT_CLASSIFICATION_METRICS
                if task_type == "classification"
                else self.DEFAULT_REGRESSION_METRICS
            )
        else:
            self.metrics = metrics

        self.primary_metric = primary_metric or self.metrics[0]

        if self.primary_metric not in self.metrics:
            raise ValueError(
                f"Primary metric '{self.primary_metric}' not in metrics list"
            )

    def evaluate_single_model(
        self,
        model: Union[ClassifierMixin, RegressorMixin],
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, Any]] = None,
        preprocessor: Optional[Any] = None,
        tune_hyperparameters: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a single model with cross-validation.

        Args:
            model: The model to evaluate
            model_name: Name for the model
            X: Features
            y: Target variable
            param_grid: Parameters for hyperparameter tuning
            preprocessor: Optional preprocessing pipeline
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            EvaluationResult with scores and best model
        """
        try:
            if preprocessor:
                pipeline = Pipeline(
                    [("preprocessor", preprocessor), ("predictor", model)]
                )
                param_grid = {
                    (k if k.startswith("predictor__") else f"predictor__{k}"): v
                    for k, v in (param_grid or {}).items()
                }
            else:
                pipeline = model
                param_grid = param_grid or {}

            scoring = {metric: metric for metric in self.metrics}

            if tune_hyperparameters and param_grid:
                search = RandomizedSearchCV(
                    estimator=pipeline,  # pyright: ignore[reportArgumentType]
                    param_distributions=param_grid,
                    cv=self.cv_folds,
                    scoring=scoring,
                    refit=self.primary_metric,  # pyright: ignore[reportArgumentType]
                    n_jobs=-1,
                    random_state=self.random_state,
                    n_iter=self.n_iter,
                )

                search.fit(X, y)
                best_model = search.best_estimator_.named_steps["predictor"]  # pyright: ignore[reportAttributeAccessIssue]
                best_params = search.best_params_

                scores = {}
                score_stds = {}
                for metric in self.metrics:
                    test_scores = [
                        search.cv_results_[f"split{i}_test_{metric}"][
                            search.best_index_
                        ]
                        for i in range(self.cv_folds)
                    ]
                    scores[metric] = np.mean(test_scores)
                    score_stds[metric] = np.std(test_scores)

            else:
                cv_results = cross_validate(
                    pipeline,  # pyright: ignore[reportArgumentType]
                    X,
                    y,
                    cv=self.cv_folds,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=-1,
                )

                pipeline.fit(X, y)  # pyright: ignore[reportAttributeAccessIssue]
                best_model = pipeline
                best_params = {}

                scores = {}
                score_stds = {}
                for metric in self.metrics:
                    test_scores = cv_results[f"test_{metric}"]
                    scores[metric] = np.mean(test_scores)
                    score_stds[metric] = np.std(test_scores)

            primary_score = scores[self.primary_metric]

            return EvaluationResult(
                model_name=model_name,
                model=best_model,  # pyright: ignore[reportArgumentType]
                best_params=best_params,
                scores=scores,
                score_stds=score_stds,
                primary_score=primary_score,
            )

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {str(e)}")
            raise

    def evaluate_multiple_models(
        self,
        models: Dict[
            str, Tuple[Union[ClassifierMixin, RegressorMixin], Dict[str, Any]]
        ],
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor: Optional[Any] = None,
        tune_hyperparameters: bool = True,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple models and return results sorted by primary metric.

        Args:
            models: Dictionary mapping model names to (model, param_grid) tuples
            X: Features
            y: Target variable
            preprocessor: Optional preprocessing pipeline
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            List of EvaluationResult objects sorted by primary metric (descending)
        """
        results = []

        for model_name, (model, param_grid) in models.items():
            try:
                result = self.evaluate_single_model(
                    model=model,
                    model_name=model_name,
                    X=X,
                    y=y,
                    param_grid=param_grid,
                    preprocessor=preprocessor,
                    tune_hyperparameters=tune_hyperparameters,
                )
                results.append(result)

                logger.info(
                    f"Evaluated {model_name}: {self.primary_metric} = {result.primary_score:.4f}"
                )

            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {str(e)}")
                continue

        if not results:
            raise ValueError("No models could be successfully evaluated")

        reverse_sort = not self.primary_metric.startswith("neg_")
        results.sort(key=lambda r: r.primary_score, reverse=reverse_sort)

        return results

    def get_comparison_dataframe(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """
        Create a comparison DataFrame from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            DataFrame with model comparison
        """
        data = []
        for result in results:
            row = {
                "model_name": result.model_name,
                "primary_score": result.primary_score,
                **result.scores,
                **{f"{metric}_std": std for metric, std in result.score_stds.items()},
            }
            data.append(row)

        return pd.DataFrame(data)

    def print_results(self, results: List[EvaluationResult], top_n: int = 5):
        """
        Print a formatted summary of evaluation results.

        Args:
            results: List of evaluation results
            top_n: Number of top models to show in detail
        """
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION RESULTS ({self.task_type.upper()})")
        print(f"{'='*60}")
        print(f"Primary metric: {self.primary_metric}")
        print(f"CV folds: {self.cv_folds}")
        print(f"Total models evaluated: {len(results)}")

        print(f"\n{'Top Models:':<15} {self.primary_metric:>15}")
        print(f"{'-'*35}")

        for i, result in enumerate(results[:top_n], 1):
            print(f"{i}. {result.model_name:<15} {result.primary_score:>15.4f}")

        if results:
            best = results[0]
            print(f"\n{'BEST MODEL DETAILS'}")
            print(f"{'-'*30}")
            print(f"Model: {best.model_name}")
            print(f"Best parameters: {best.best_params}")
            print("\nAll metrics:")
            for metric, score in best.scores.items():
                std = best.score_stds[metric]
                print(f"  {metric:<20}: {score:>8.4f} (±{std:.4f})")
