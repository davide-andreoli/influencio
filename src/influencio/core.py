import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.exceptions import NotFittedError
from shap import Explainer
from shap._explanation import Explanation
from .evaluator import ModelEvaluator
from .enums import ColumnType
from .preprocessor import Preprocessor
from .validator import DataValidator
from .selector import ModelSelector
from .utils import determine_column_type
from .visualizer import DataVisualizer
from .explainer import ExplanationGenerator
from .tree import TreeInsightsExtractor
from typing import cast, Optional, Tuple, Any, List, Literal
import logging

logger = logging.getLogger(__name__)


class KeyInfluencers:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
        task: Optional[str] = None,  # 'classification' or 'regression'
        auto_tune: bool = True,
        model: Optional[Any] = None,
        tree_depth: int = 3,
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
        self.task = (
            task
            if task
            else (
                "classification"
                if determine_column_type(cast(pd.Series, self.dataframe[self.target]))
                == ColumnType.CATEGORICAL
                else "regression"
            )
        )
        self.preprocessor = Preprocessor()
        self.validator = DataValidator()
        self.model_evaluator = ModelEvaluator(
            task_type=cast(Literal["classification", "regression"], self.task)
        )
        self.model_selector = ModelSelector(
            evaluator=self.model_evaluator,
            preprocessor=self.preprocessor,
            user_model=model,
            tuning=auto_tune,
            task=self.task,
        )
        self.data_visualizer = DataVisualizer()
        self.tree_depth = tree_depth
        self.model_pipeline: Optional[Pipeline] = None
        self.tree_pipeline: Optional[Pipeline] = None
        self.explainer_generator: Optional[ExplanationGenerator] = None
        self.explainer: Optional[Explainer] = None
        self.class_names: Optional[List[str]] = None
        self.input_feature_names: List[str] = cast(
            List[str], dataframe.drop(target, axis=1).columns.to_list()
        )
        self.shap_values: Optional[Explanation] = None

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

        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        if self.task == "classification":
            tree_predictor = DecisionTreeClassifier(max_depth=self.tree_depth)
        else:
            tree_predictor = DecisionTreeRegressor(max_depth=self.tree_depth)

        predictor = self.model_selector.select_best_model(X, y)

        self.model_pipeline = Pipeline(
            [("preprocessor", self.preprocessor), ("predictor", predictor)]
        )

        self.tree_pipeline = Pipeline(
            [("preprocessor", self.preprocessor), ("tree", tree_predictor)]
        )

        self.model_pipeline.fit(X, y)
        self.tree_pipeline.fit(X, y)

        if self.task == "classification":
            self.class_names = self.model_pipeline.named_steps[
                "predictor"
            ].classes_.tolist()
        else:
            self.class_names = None

        self.explainer_generator = ExplanationGenerator(
            pipeline=self.model_pipeline,
            task=self.task,
            class_names=self.class_names,
            input_feature_names=self.input_feature_names,
        )

        self.explainer, self.shap_values = self.explainer_generator.create_explainer(X)
        self.tree_insights_extractor = TreeInsightsExtractor(
            tree_pipeline=self.tree_pipeline,
            task=self.task,
            dataframe=self.dataframe,
            target=self.target,
            class_names=self.class_names,
        )

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

        self.data_visualizer.plot_global_feature_importance(
            self.shap_values,
            feature_names=self.input_feature_names,
            class_names=self.class_names,
            target_type=ColumnType.CATEGORICAL
            if self.task == "classification"
            else ColumnType.NUMERICAL,
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

        # TODO: maybe move this inside plot_local_feature_importance
        if self.task == "classification":
            predicted_probabilities = self.model_pipeline.predict_proba(  # pyright: ignore[reportOptionalMemberAccess]
                self.dataframe.drop(self.target, axis=1).iloc[index : index + 1]
            )
            predicted_class_index = np.argmax(predicted_probabilities)
            shap_values = self.shap_values.values[index, :, predicted_class_index]  # pyright: ignore[reportCallIssue, reportArgumentType]
        else:
            shap_values = self.shap_values.values[index]

        self.data_visualizer.plot_local_feature_importance(
            shap_values,
            feature_names=self.input_feature_names,
            class_name=self.class_names[predicted_class_index]
            if self.task == "classification"
            and self.class_names is not None
            and predicted_class_index
            else None,
        )

    def key_segments(
        self, top_n: int = 5, focus_class: Optional[str] = None
    ) -> Tuple[Any, Any, Any]:
        if not self.tree_pipeline:
            raise NotFittedError(
                "The KeyInfluencers object should be fitted using .fit() before calling graphing methods."
            )

        return self.tree_insights_extractor.key_segments(
            top_n=top_n, focus_class=focus_class
        )
