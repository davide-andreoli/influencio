import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from shap import Explainer
from shap._explanation import Explanation
from .visualizations import (
    plot_global_feature_importance,
    plot_local_feature_importance,
)
from .tree import (
    extract_feature_contributions,
    extract_tree_rules,
    extract_tree_insights,
)
from .evaluator import ModelEvaluator, EvaluationResult
from .candidates import CLASSIFICATION_CANDIDATES, REGRESSION_CANDIDATES
from .enums import ColumnType, TreeType
from .preprocessor import Preprocessor
from .validator import DataValidator
from .utils import determine_column_type
from typing import cast, Optional, Tuple, Dict, Any, List, Union
import logging

logger = logging.getLogger(__name__)


class KeyInfluencers:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
        model: Optional[Union[ClassifierMixin, RegressorMixin]] = None,
        tree_model: Optional[Union[ClassifierMixin, RegressorMixin]] = None,
        tuning: bool = True,
        tuning_candidates: Optional[
            Dict[str, Tuple[Union[ClassifierMixin, RegressorMixin], Dict[str, Any]]]
        ] = None,
    ):
        """
        KeyInfluencers is a class that provides methods to analyze and visualize the key influencers of a target variable in a dataset.
        It uses SHAP (SHapley Additive exPlanations) values to explain the predictions of a machine learning model.
        The class supports both classification and regression tasks.
        Args:
            dataframe (pd.DataFrame): The input dataframe containing features and target variable.
            target (str): The name of the target variable in the dataframe.
            model (Optional[Union[ClassifierMixin, RegressorMixin]]): A user-provided machine learning model for prediction. If None, a default model will be selected based on the target type.
            tree_model (Optional[Union[ClassifierMixin, RegressorMixin]]): A user-provided decision tree model for extracting insights. If None, a default decision tree will be used.
        """

        self.dataframe = dataframe
        self.target = target

        self.preprocessor = Preprocessor()
        self.validator = DataValidator()
        self.tuning: bool = tuning
        self.tuning_candidates: Optional[
            Dict[str, Tuple[Union[ClassifierMixin, RegressorMixin], Dict[str, Any]]]
        ] = tuning_candidates
        self.model: Optional[Union[ClassifierMixin, RegressorMixin]] = model
        self.tree_model: Optional[Union[ClassifierMixin, RegressorMixin]] = tree_model
        self.model_pipeline: Optional[Pipeline] = None
        self.tree_pipeline: Optional[Pipeline] = None
        self.explainer: Optional[Explainer] = None
        self.class_names: Optional[List[str]] = None
        self.input_feature_names: List[str] = cast(
            List[str], dataframe.drop(target, axis=1).columns.to_list()
        )
        self.transformed_feature_names: Optional[List[str]] = None
        self.shap_values: Optional[Explanation] = None
        self.target_type: Optional[ColumnType] = None
        self.model_metrics: Optional[EvaluationResult] = None
        self._model_evaluator: Optional[ModelEvaluator] = None
        self.evaluation_results: Optional[List] = None

    @property
    def model_evaluator(self) -> ModelEvaluator:
        if not self._model_evaluator:
            task_type = (
                "classification"
                if self.target_type == ColumnType.CATEGORICAL
                else "regression"
            )
            self._model_evaluator = ModelEvaluator(task_type=task_type)
        return self._model_evaluator

    def _select_best_model(
        self, X: pd.DataFrame, y: pd.Series, target_type: ColumnType
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
        if self.model is not None:
            logger.info("Using user provided model for prediction.")
            self.model_metrics = self.model_evaluator.evaluate_single_model(
                self.model, "user_provided_model", X, y
            )
            return self.model

        candidates = (
            (
                CLASSIFICATION_CANDIDATES
                if target_type == ColumnType.CATEGORICAL
                else REGRESSION_CANDIDATES
            )
            if not self.tuning_candidates
            else self.tuning_candidates
        )

        results = self.model_evaluator.evaluate_multiple_models(
            models=candidates,
            X=X,
            y=y,
            preprocessor=self.preprocessor,
            tune_hyperparameters=self.tuning,
        )

        self.evaluation_results = results
        logger.info(f"Best evaluated model: {results[0]}")
        self.model_metrics = results[0]
        return results[0].model

    def fit(self) -> None:
        """
        Fits the model pipeline to the provided dataframe and prepares it for predictions and explanations.
        This method performs the following steps:
            1. Splits the dataframe into features (X) and target (y)
            2. Identifies categorical, numerical and time based columns
            3. Creates a preprocessing pipeline for the features
            4. If a model is not provided by the user, it selects the best model based on cross-validation scores
            5. Fits the model pipeline to the data
            6. Creates a SHAP explainer for the fitted model and computes SHAP values
        Notes:
            - Time-based columns are currently not handled and require additional preprocessing.
        """
        X: pd.DataFrame = self.dataframe.drop(self.target, axis=1)
        y: pd.Series = cast(pd.Series, self.dataframe[self.target])

        self.validator.validate_data(X, y)

        self.target_type = determine_column_type(y)

        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        # TODO: Add automatic model choice based on performance
        if self.target_type == ColumnType.CATEGORICAL and not self.tree_model:
            tree_predictor = DecisionTreeClassifier(max_depth=3)
        elif self.target_type == ColumnType.NUMERICAL and not self.tree_model:
            tree_predictor = DecisionTreeRegressor(max_depth=3)
        else:
            tree_predictor = self.tree_model

        predictor = self._select_best_model(X, y, self.target_type)

        self.model_pipeline = Pipeline(
            [("preprocessor", self.preprocessor), ("predictor", predictor)]
        )

        self.tree_pipeline = Pipeline(
            [("preprocessor", self.preprocessor), ("tree", tree_predictor)]
        )

        self.model_pipeline.fit(X, y)
        self.tree_pipeline.fit(X, y)

        self.transformed_feature_names = self.model_pipeline.named_steps[
            "preprocessor"
        ].get_feature_names_out()
        if self.target_type == ColumnType.CATEGORICAL:
            self.class_names = self.model_pipeline.named_steps[
                "predictor"
            ].classes_.tolist()
        else:
            self.class_names = None

        self.explainer = Explainer(
            lambda X: self.model_pipeline.predict_proba(X)  # pyright: ignore[reportOptionalMemberAccess]
            if self.target_type == ColumnType.CATEGORICAL
            else self.model_pipeline.predict(X),  # pyright: ignore[reportOptionalMemberAccess]
            X,
            feature_names=self.input_feature_names,
            output_names=self.class_names,
        )
        self.shap_values = self.explainer(X)

    def global_feature_importance(self, max_display: int = 10) -> None:
        """
        Plots the global feature importance using SHAP values.
        Args:
            max_display (int): The maximum number of features to display in the plot.
        """

        if not self.shap_values:
            raise NotFittedError(
                "The KeyInfluencers object should be fitted using .fit() before calling graphing methods."
            )

        plot_global_feature_importance(
            self.shap_values,
            max_display=max_display,
            feature_names=self.input_feature_names,
            class_names=self.class_names,
            target_type=self.target_type,
        )

    def local_feature_importance(self, index: int, max_display: int = 10) -> None:
        """
        Plots the local feature importance using SHAP values.
        Args:
            index (int): The index of the instance for which to plot local feature importance
            max_display (int): The maximum number of features to display in the plot
        """
        if index < 0 or index >= len(self.dataframe):
            raise IndexError("Index out of range for the dataframe.")

        if not self.shap_values:
            raise NotFittedError(
                "The KeyInfluencers object should be fitted using .fit() before calling graphing methods."
            )

        predicted_class_index = None

        if self.target_type == ColumnType.CATEGORICAL:
            predicted_probabilities = self.model_pipeline.predict_proba(  # pyright: ignore[reportOptionalMemberAccess]
                self.dataframe.drop(self.target, axis=1).iloc[index : index + 1]
            )
            predicted_class_index = np.argmax(predicted_probabilities)
            shap_values = self.shap_values.values[index, :, predicted_class_index]  # pyright: ignore[reportCallIssue, reportArgumentType]
        else:
            shap_values = self.shap_values.values[index]

        plot_local_feature_importance(
            shap_values,
            max_display=max_display,
            feature_names=self.input_feature_names,
            class_name=self.class_names[predicted_class_index]
            if self.target_type == ColumnType.CATEGORICAL
            and self.class_names is not None
            and predicted_class_index
            else None,
        )

    def key_segments(
        self, top_n: int = 5, focus_class: Optional[str] = None
    ) -> Tuple[Any, Any, Any]:
        if not self.tree_pipeline or self.transformed_feature_names is None:
            raise NotFittedError(
                "The KeyInfluencers object should be fitted using .fit() before calling graphing methods."
            )

        tree = cast(
            Union[DecisionTreeClassifier, DecisionTreeRegressor], self.tree_pipeline[-1]
        )

        if self.target_type == ColumnType.CATEGORICAL:
            y = self.dataframe[self.target]
            class_counts = y.value_counts()

            if focus_class is None:
                focus_class = cast(str, class_counts.idxmax())

            focus_class_index = self.class_names.index(focus_class)  # pyright: ignore [reportOptionalMemberAccess]
            overall_mean = (y == focus_class).mean()

            feature_contributions = extract_feature_contributions(
                tree, self.transformed_feature_names
            )
            rules = extract_tree_rules(tree, self.transformed_feature_names)
            insights = extract_tree_insights(
                tree,
                self.transformed_feature_names,
                overall_mean,
                TreeType.CLASSIFICATION,
                top_n=top_n,
                focus_class_index=focus_class_index,
                focus_class=focus_class,
            )
        else:
            y = self.dataframe[self.target]
            overall_mean = cast(float, y.mean())
            feature_contributions = extract_feature_contributions(
                tree, self.transformed_feature_names
            )
            rules = extract_tree_rules(tree, self.transformed_feature_names)
            insights = extract_tree_insights(
                tree,
                self.transformed_feature_names,
                overall_mean,
                TreeType.REGRESSION,
                top_n=top_n,
                target=self.target,
            )

        return feature_contributions, rules, insights
