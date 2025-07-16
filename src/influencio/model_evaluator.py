from .data_classes import MetricConfig, ModelEvaluationResult
from .enums import MetricType, DataCharacteristic
from typing import cast, Optional, List, Literal, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, RandomizedSearchCV
import logging
from sklearn.base import ClassifierMixin, RegressorMixin

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(
        self,
        custom_metrics: Optional[List[MetricConfig]] = None,
        cv_folds: int = 5,
        scoring_strategy: Literal["adaptive", "user_defined", "single"] = "adaptive",
        primary_metric: Optional[MetricType] = None,
        random_state: int = 42,
    ) -> None:
        self.custom_metrics = custom_metrics
        self.cv_folds = cv_folds
        self.scoring_strategy = scoring_strategy
        self.primary_metric = primary_metric
        self.random_state = random_state

        self.CLASSIFICATION_METRICS = {
            DataCharacteristic.BALANCED: [
                MetricConfig(MetricType.ACCURACY, weight=1.0),
                MetricConfig(MetricType.F1, weight=0.8),
                MetricConfig(MetricType.ROC_AUC, weight=0.6),
            ],
            DataCharacteristic.IMBALANCED: [
                MetricConfig(MetricType.BALANCED_ACCURACY, weight=1.0),
                MetricConfig(MetricType.F1, weight=0.9),
                MetricConfig(MetricType.PRECISION, weight=0.7),
                MetricConfig(MetricType.RECALL, weight=0.7),
            ],
            DataCharacteristic.MULTICLASS: [
                MetricConfig(MetricType.ACCURACY, weight=1.0),
                MetricConfig(MetricType.F1, weight=0.8),
                MetricConfig(MetricType.MATTHEWS_CORR, weight=0.6),
            ],
            DataCharacteristic.SMALL_SAMPLE: [
                MetricConfig(MetricType.BALANCED_ACCURACY, weight=1.0),
                MetricConfig(MetricType.F1, weight=0.8),
            ],
        }

        self.REGRESSION_METRICS = {
            DataCharacteristic.BALANCED: [
                MetricConfig(MetricType.R2, weight=1.0),
                MetricConfig(
                    MetricType.MAE,
                    weight=0.8,
                    higher_is_better=False,
                    sklearn_name="neg_mean_absolute_error",
                ),
                MetricConfig(
                    MetricType.MSE,
                    weight=0.6,
                    higher_is_better=False,
                    sklearn_name="neg_mean_squared_error",
                ),
            ],
            DataCharacteristic.HIGH_DIMENSIONAL: [
                MetricConfig(MetricType.R2, weight=1.0),
                MetricConfig(
                    MetricType.MAE,
                    weight=0.9,
                    higher_is_better=False,
                    sklearn_name="neg_mean_absolute_error",
                ),
                MetricConfig(MetricType.EXPLAINED_VARIANCE, weight=0.7),
            ],
            DataCharacteristic.SMALL_SAMPLE: [
                MetricConfig(
                    MetricType.MAE,
                    weight=1.0,
                    higher_is_better=False,
                    sklearn_name="neg_mean_absolute_error",
                ),
                MetricConfig(MetricType.R2, weight=0.8),
            ],
        }

    def _analyze_classification_data(self, y: pd.Series) -> List[DataCharacteristic]:
        """Analyze classification target variable characteristics"""
        # TODO: see if it makes sense to move to a dedicated class (staticmethod)
        characteristics = []

        class_counts = y.value_counts()
        n_classes = len(class_counts)

        if n_classes == 2:
            characteristics.append(DataCharacteristic.BINARY)
        else:
            characteristics.append(DataCharacteristic.MULTICLASS)

        min_class_ratio = class_counts.min() / class_counts.max()
        if min_class_ratio < 0.4:
            characteristics.append(DataCharacteristic.IMBALANCED)
        else:
            characteristics.append(DataCharacteristic.BALANCED)

        return characteristics

    def _analyze_regression_data(self, y: pd.Series) -> List[DataCharacteristic]:
        """Analyze regression target variable characteristics"""
        characteristics = []

        skewness = abs(float(y.skew()))  # pyright: ignore[reportArgumentType]
        if skewness > 1:
            characteristics.append(DataCharacteristic.HIGH_DIMENSIONAL)

        return characteristics

    def _analyze_dataset_size(self, X: pd.DataFrame) -> List[DataCharacteristic]:
        """Analyze dataset size characteristics"""
        characteristics = []

        n_samples, n_features = X.shape

        if n_samples < 1000:
            characteristics.append(DataCharacteristic.SMALL_SAMPLE)

        if n_features > n_samples * 0.1:
            characteristics.append(DataCharacteristic.HIGH_DIMENSIONAL)

        return characteristics

    def _select_metrics(
        self,
        characteristics: List[DataCharacteristic],
        task_type: str,
        user_metrics: Optional[List[MetricConfig]] = None,
    ) -> List[MetricConfig]:
        """Select metrics based on data characteristics and user preferences"""

        if user_metrics:
            return user_metrics

        metrics_dict = (
            self.CLASSIFICATION_METRICS
            if task_type == "classification"
            else self.REGRESSION_METRICS
        )

        selected_metrics = []

        for characteristic in characteristics:
            if characteristic in metrics_dict:
                for metric in metrics_dict[characteristic]:
                    existing_metric = next(
                        (
                            m
                            for m in selected_metrics
                            if m.metric_type == metric.metric_type
                        ),
                        None,
                    )
                    if existing_metric:
                        existing_metric.weight = max(
                            existing_metric.weight, metric.weight
                        )
                    else:
                        selected_metrics.append(metric)

        if not selected_metrics:
            default_char = DataCharacteristic.BALANCED
            selected_metrics = metrics_dict.get(default_char, [])

        return selected_metrics

    def analyze_and_select_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: Literal["classification", "regression"],
    ):
        if task_type == "classification":
            target_characteristics = self._analyze_classification_data(y)
        else:
            target_characteristics = self._analyze_regression_data(y)

        dataset_characteristics = self._analyze_dataset_size(X)
        all_characteristics = target_characteristics + dataset_characteristics

        if self.scoring_strategy == "user_defined" and self.custom_metrics:
            selected_metrics = self.custom_metrics
        elif self.scoring_strategy == "single" and self.primary_metric:
            selected_metrics = [MetricConfig(self.primary_metric)]
        else:
            selected_metrics = self._select_metrics(
                all_characteristics, task_type, self.custom_metrics
            )

        analysis_summary = (
            f"Data characteristics: {[c.value for c in all_characteristics]}"
        )

        return selected_metrics, analysis_summary

    def _extract_cv_scores(
        self,
        raw_scores: Dict[str, np.ndarray[Any, Any]],
        metric_configs: List[MetricConfig],
        prefix: str = "test_",
        best_index: Optional[int] = None,
        is_search: bool = False,
    ) -> Tuple[Dict[str, np.float64], Dict[str, np.ndarray], Dict[str, np.float64]]:
        """
        Extract mean, std, and raw scores from CV results.
        `best_index` only needed for grid/randomized search.
        """
        means = {}
        stds = {}
        cv_values = {}

        for metric in metric_configs:
            key = metric.metric_type.value
            sk_key = f"{prefix}{key}"

            if is_search:
                values = np.array(
                    [
                        raw_scores[f"split{i}_{sk_key}"][best_index]
                        for i in range(self.cv_folds)
                    ]
                )
            else:
                values = raw_scores[sk_key]

            correct_values = np.array(
                [metric.handle_negative_scores(v) for v in values]
            )

            means[key] = np.mean(correct_values)
            stds[key] = np.std(correct_values)
            cv_values[key] = correct_values

        return means, cv_values, stds

    def evaluate_model(
        self,
        model: Union[ClassifierMixin, RegressorMixin],
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, Any],
        pipeline,
        task_type: Literal["classification", "regression"],
        tuning: bool = True,
    ) -> ModelEvaluationResult:
        """Evaluate a single model with selected metrics in a single CV run."""

        metrics, analysis = self.analyze_and_select_metrics(X, y, task_type)

        scoring = {metric.metric_type.value: metric.sklearn_name for metric in metrics}
        primary_metric_id = metrics[0].metric_type.value

        cv_results: Optional[Dict[str, np.ndarray[Any, Any]]] = None
        cv_result: Optional[Dict[str, np.ndarray[Any, Any]]] = None
        search: Optional[RandomizedSearchCV] = None

        if tuning and param_grid:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                cv=self.cv_folds,
                scoring=scoring,
                refit=primary_metric_id,
                n_jobs=-1,
                random_state=self.random_state,
                n_iter=10,
            )

            search.fit(X, y)
            best_model = search.best_estimator_.named_steps.get(  # pyright: ignore[reportAttributeAccessIssue]
                "predictor", search.best_estimator_
            )
            best_params = search.best_params_

            cv_results = search.cv_results_

        else:
            pipeline.fit(X, y)
            best_model = model
            best_params = {}

            cv_result = cross_validate(
                pipeline,
                X,
                y,
                cv=self.cv_folds,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
            )

        if not cv_result and not cv_results:
            raise ValueError("No cross-validation results available.")

        mean_scores, cv_scores, std_scores = self._extract_cv_scores(
            raw_scores=cv_results if tuning else cv_result,  # pyright: ignore [reportArgumentType]
            metric_configs=metrics,
            best_index=search.best_index_ if tuning and search else None,
            is_search=tuning,
        )

        weighted_score = self._calculate_weighted_score(mean_scores, metrics)
        if np.isnan(weighted_score):
            logger.warning(
                f"Weighted score for {model_name} is NaN. Check metric config and scoring."
            )
        primary_score = mean_scores.get(primary_metric_id, cast(np.float64, 0.0))

        return ModelEvaluationResult(
            model_name=model_name,
            model=best_model,
            primary_score=primary_score,
            all_scores=mean_scores,
            best_params=best_params,
            weighted_score=weighted_score,
            cv_scores=cv_scores,
            std_scores=std_scores,
            metric_configs=metrics,
        )

    def evaluate_candidates(
        self,
        candidates: Dict[
            str, Tuple[Union[ClassifierMixin, RegressorMixin], Dict[str, Any]]
        ],
        X: pd.DataFrame,
        y: pd.Series,
        task_type: Literal["classification", "regression"],
        preprocessor: Optional[ColumnTransformer] = None,
        tuning: bool = True,
    ) -> List[ModelEvaluationResult]:
        """Evaluates all candidate models and selects the best one."""
        results = []

        for name, (model, param_grid) in candidates.items():
            pipeline = Pipeline([("preprocessor", preprocessor), ("predictor", model)])

            try:
                result = self.evaluate_model(
                    model=model,
                    model_name=name,
                    X=X,
                    y=y,
                    param_grid=param_grid,
                    pipeline=pipeline,
                    task_type=task_type,
                    tuning=tuning,
                )
                results.append(result)
                logger.info(
                    f"Evaluated {name}: Primary Score = {result.primary_score:.4f}, Weighted Score = {result.weighted_score:.4f}"
                )
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {str(e)}")

        if not results:
            raise ValueError("No models could be successfully evaluated.")

        sorted_results = self.compare_models(results)

        return sorted_results

    def _calculate_weighted_score(
        self,
        scores: Dict[str, np.float64],
        metrics: List[MetricConfig],
        normalization_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> float:
        """
        Calculate weighted score using normalized metric values.
        `normalization_bounds` should contain (min, max) for each metric across all models.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for metric in metrics:
            key = metric.metric_type.value
            if key not in scores:
                continue
            if np.isnan(scores[key]):
                logger.warning(f"Score for metric {key} is NaN. Check metric config.")
                continue

            raw_score = scores[key]

            score = raw_score if metric.higher_is_better else -raw_score

            if normalization_bounds and key in normalization_bounds:
                min_val, max_val = normalization_bounds[key]
                if max_val != min_val:
                    score = (score - min_val) / (max_val - min_val)
                else:
                    score = 1.0

            weighted_sum += score * metric.weight
            total_weight += metric.weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def compare_models(
        self, results: List[ModelEvaluationResult]
    ) -> List[ModelEvaluationResult]:
        if not results:
            raise ValueError("No model results to compare")

        all_scores = [r.all_scores for r in results]
        metrics = list(all_scores[0].keys())

        normalization_bounds = {}
        for metric in metrics:
            values = [s[metric] for s in all_scores if metric in s]
            min_val = np.min(values)
            max_val = np.max(values)
            normalization_bounds[metric] = (min_val, max_val)

        for result in results:
            result.weighted_score = self._calculate_weighted_score(
                result.all_scores,
                metrics=result.metric_configs if result.metric_configs else [],
                normalization_bounds=normalization_bounds,
            )

        sorted_results = sorted(results, key=lambda x: x.weighted_score, reverse=True)

        return sorted_results

    def print_evaluation_summary(self, result: ModelEvaluationResult):
        """Print a summary of model evaluation results"""

        print(f"\n{'='*50}")
        print(f"Best Model: {result.model_name}")
        print(f"{'='*50}")
        print(f"Primary Score: {result.primary_score:.4f}")
        print(f"Weighted Score: {result.weighted_score:.4f}")
        print(f"Best Parameters: {result.best_params}")
        print("\nDetailed Scores:")
        print(f"{'-'*30}")

        for metric, score in result.all_scores.items():
            std = result.std_scores.get(metric, 0.0)
            print(f"{metric:20}: {score:.4f} (±{std:.4f})")

    @classmethod
    def from_focus(
        cls,
        task_type: Literal["classification", "regression"],
        focus_on: Optional[Literal["precision", "recall", "balanced"]],
    ):
        custom_metrics = []
        if focus_on == "precision" and task_type == "classification":
            custom_metrics = [
                MetricConfig(MetricType.PRECISION, weight=1.0),
                MetricConfig(MetricType.F1, weight=0.8),
                MetricConfig(MetricType.ROC_AUC, weight=0.6),
            ]
        elif focus_on == "recall" and task_type == "classification":
            custom_metrics = [
                MetricConfig(MetricType.RECALL, weight=1.0),
                MetricConfig(MetricType.F1, weight=0.8),
                MetricConfig(MetricType.BALANCED_ACCURACY, weight=0.6),
            ]
        elif focus_on == "balanced" and task_type == "classification":
            custom_metrics = [
                MetricConfig(MetricType.BALANCED_ACCURACY, weight=1.0),
                MetricConfig(MetricType.F1, weight=0.9),
                MetricConfig(MetricType.MATTHEWS_CORR, weight=0.7),
            ]
        return ModelEvaluator(
            custom_metrics=custom_metrics,
            scoring_strategy="user_defined",
        )
