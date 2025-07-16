import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
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
from .evaluator import ModelEvaluator, MetricConfig
from .candidates import CLASSIFICATION_CANDIDATES, REGRESSION_CANDIDATES
from .enums import ColumnType, TreeType
from typing import cast, Optional, Tuple, Dict, Any, List, Union, Literal
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
        evaluation_strategy: str = "adaptive",
        custom_metrics: Optional[List[MetricConfig]] = None,
        focus_on: Optional[Literal["precision", "recall", "balanced"]] = None,
        cv_folds: int = 5,
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

        self.preprocessor: Optional[ColumnTransformer] = None
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
        self.model_metrics: Optional[Dict[str, np.float64]] = None

        self.evaluation_strategy = evaluation_strategy
        self.custom_metrics = custom_metrics
        self.focus_on = focus_on
        self.cv_folds = cv_folds
        self.model_evaluator: Optional[ModelEvaluator] = None
        self.evaluation_results: Optional[List] = None
        self.best_model_result: Optional[Any] = None

    def _create_model_evaluator(
        self, task_type: Literal["classification", "regression"]
    ) -> ModelEvaluator:
        """Create and configure the model evaluator"""
        if self.evaluation_strategy == "user_defined" and self.custom_metrics:
            return ModelEvaluator(
                custom_metrics=self.custom_metrics,
                cv_folds=self.cv_folds,
                scoring_strategy="user_defined",
            )
        elif self.focus_on:
            return ModelEvaluator.from_focus(
                task_type=task_type,
                focus_on=self.focus_on,  # pyright: ignore[reportArgumentType]
            )
        else:
            return ModelEvaluator(
                cv_folds=self.cv_folds,
                scoring_strategy="adaptive",
            )

    def _validate_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Validates input data quality and structure.

        Args:
            X: Feature dataframe
            y: target series

        Raises:
            ValueError: if data does not meet the minimum requirements
        """

        if len(X) < 10:
            raise ValueError(
                f"The given dataset is too small: {len(X)}. Please use a dataset containing at least 10 rows"
            )

        if y.isnull().all():
            raise ValueError("The target variable contains only null values.")

        if y.isnull().sum() > len(y) * 0.5:
            logger.warning(
                f"Target variable has {y.isnull().sum()/len(y)*100:.1f}% missing values."
            )

        numeric_columns = X.select_dtypes(include=[np.number]).columns
        low_variance_columns = []

        for col in numeric_columns:
            if X[col].to_numpy().var() < 1e-10:
                low_variance_columns.append(col)

        if low_variance_columns:
            logger.warning(f"Low variance features detected: {low_variance_columns}")

        high_missing_columns = []
        for col in X.columns:
            missing_percent = X[col].isnull().sum() / len(X)
            if missing_percent > 0.8:
                high_missing_columns.append((col, missing_percent))

        if high_missing_columns:
            logger.warning(f"Features with >80% missing values: {high_missing_columns}")

        target_type = self._determine_column_type(y)
        if target_type == ColumnType.CATEGORICAL:
            unique_values = y.nunique()
            if unique_values < 2:
                raise ValueError(
                    f"Categorical target must have at least 2 classes. Found: {unique_values}"
                )
            if unique_values > 50:
                logger.warning(
                    f"High cardinality target: {unique_values} classes. Consider grouping to improve performances."
                )

        logger.info("Data validation completed successfully.")

    def _evaluate_model_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        predictor: Union[ClassifierMixin, RegressorMixin],
    ) -> Dict[str, np.float64]:
        """
        Evaluates chosen model performance using cross-validation.

        Args:
            X: Feature dataframe
            y: Target series

        Returns:
            Dictionary containing performance metrics
        """
        if not self.preprocessor:
            raise ValueError("Preprocessor is not initialized.")

        if not self.model_evaluator:
            task_type = (
                "classification"
                if self.target_type == ColumnType.CATEGORICAL
                else "regression"
            )
            self.model_evaluator = self._create_model_evaluator(task_type)

        model_pipeline = Pipeline(
            [("preprocessor", self.preprocessor), ("predictor", predictor)]
        )

        dummy_param_grid = {}

        eval_result = self.model_evaluator.evaluate_model(
            model=model_pipeline.named_steps["predictor"],
            model_name="final_model",
            X=X,
            y=y,
            param_grid=dummy_param_grid,
            pipeline=model_pipeline,
            task_type="classification"
            if self.target_type == ColumnType.CATEGORICAL
            else "regression",
            tuning=False,
        )

        return eval_result.all_scores

    def print_model_performance(self) -> None:
        """
        Prints a formatted summary of model performance metrics.
        """
        pass

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
            self.model_metrics = self._evaluate_model_performance(X, y, self.model)
            return self.model

        task_type = (
            "classification" if target_type == ColumnType.CATEGORICAL else "regression"
        )

        if not self.model_evaluator:
            self.model_evaluator = self._create_model_evaluator(task_type)

        candidates = (
            (
                CLASSIFICATION_CANDIDATES
                if target_type == ColumnType.CATEGORICAL
                else REGRESSION_CANDIDATES
            )
            if not self.tuning_candidates
            else self.tuning_candidates
        )

        results = self.model_evaluator.evaluate_candidates(
            candidates=candidates,
            X=X,
            y=y,
            task_type=task_type,
            preprocessor=self.preprocessor,
            tuning=self.tuning,
        )

        self.evaluation_results = results
        logger.info(f"Best evaluated model: {results[0]}")
        self.model_metrics = results[0].all_scores
        return results[0].model

    def get_model_comparison(self) -> pd.DataFrame:
        """
        Return a DataFrame comparing all evaluated models
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run fit() first.")

        comparison_data = []
        for result in self.evaluation_results:
            row = {
                "Model": result.model_name,
                "Primary_Score": result.primary_score,
                "Weighted_Score": result.weighted_score,
                **result.all_scores,
                **{f"{k}_std": v for k, v in result.std_scores.items()},
            }
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df.sort_values("Weighted_Score", ascending=False)

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get detailed evaluation summary
        """
        if not self.best_model_result:
            raise ValueError("No evaluation results available. Run fit() first.")

        return {
            "best_model": self.best_model_result.model_name,
            "best_params": self.best_model_result.best_params,
            "primary_score": self.best_model_result.primary_score,
            "weighted_score": self.best_model_result.weighted_score,
            "all_scores": self.best_model_result.all_scores,
            "score_stds": self.best_model_result.std_scores,
            "cv_folds": self.cv_folds,
            "evaluation_strategy": self.evaluation_strategy,
        }

    def plot_model_comparison(self):
        """
        Plot comparison of all evaluated models
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run fit() first.")

        comparison_df = self.get_model_comparison()

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        sns.barplot(data=comparison_df, x="Primary_Score", y="Model")
        plt.title("Primary Metric Comparison")

        plt.subplot(2, 2, 2)
        sns.barplot(data=comparison_df, x="Weighted_Score", y="Model")
        plt.title("Weighted Score Comparison")

        plt.subplot(2, 1, 2)
        metrics_cols = [
            col
            for col in comparison_df.columns
            if col not in ["Model", "Primary_Score", "Weighted_Score"]
            and not col.endswith("_std")
        ]

        if metrics_cols:
            heatmap_data = comparison_df[["Model"] + metrics_cols].set_index("Model")
            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlBu_r")
            plt.title("All Metrics Heatmap")

        plt.tight_layout()
        plt.show()

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

        self._validate_data(X, y)

        categorical_columns = [
            column
            for column in X.columns
            if self._determine_column_type(cast(pd.Series, X[column]))
            == ColumnType.CATEGORICAL
        ]
        # time_columns = [
        #    column
        #    for column in X.columns
        #    if self._determine_column_type(X[column]) == ColumnType.TIME
        # ]
        numerical_columns = [
            column
            for column in X.columns
            if self._determine_column_type(cast(pd.Series, X[column]))
            == ColumnType.NUMERICAL
        ]

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        numerical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", StandardScaler()),
            ]
        )

        # TODO: Add preprocessor for time data
        # TODO: Maybe add a feature selection step in the pipeline
        preprocessor = ColumnTransformer(
            [
                ("categorical", categorical_pipeline, categorical_columns),
                ("numerical", numerical_pipeline, numerical_columns),
            ]
        )

        self.preprocessor = preprocessor

        self.target_type = self._determine_column_type(y)

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
            [("preprocessor", preprocessor), ("predictor", predictor)]
        )

        self.tree_pipeline = Pipeline(
            [("preprocessor", preprocessor), ("tree", tree_predictor)]
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

    def _determine_column_type(self, column: pd.Series) -> ColumnType:
        """
        Determines the type of a column based on its data type and unique values.
        Args:
            column (pd.Series): The input column to determine the type
        Returns:
            ColumnType: The determined type of the column (categorical, numerical, or time)
        """
        if (
            pd.api.types.is_bool_dtype(column)
            or pd.api.types.is_object_dtype(column)
            or column.nunique() <= 10  # TODO: Understand if this limit is good
        ):
            return ColumnType.CATEGORICAL
        elif pd.api.types.is_datetime64_any_dtype(column):
            return ColumnType.TIME  # pragma: no cover
        elif pd.api.types.is_numeric_dtype(column):
            return ColumnType.NUMERICAL
        else:
            raise ValueError(f"Unhandled column type: {column.dtype}")

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
