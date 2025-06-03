import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.base import BaseEstimator
import shap
from .visualizations import (
    plot_global_feature_importance,
    plot_local_feature_importance,
)
from .tree import (
    extract_feature_contributions,
    extract_tree_rules,
    extract_tree_insights,
)
from .candidates import CLASSIFICATION_CANDIDATES, REGRESSION_CANDIDATES
from .enums import ColumnType, TreeType
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class KeyInfluencers:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        target: str,
        model: Optional[BaseEstimator] = None,
        tree_model: Optional[BaseEstimator] = None,
        tuning: bool = True,
    ):
        """
        KeyInfluencers is a class that provides methods to analyze and visualize the key influencers of a target variable in a dataset.
        It uses SHAP (SHapley Additive exPlanations) values to explain the predictions of a machine learning model.
        The class supports both classification and regression tasks.
        Args:
            dataframe (pd.DataFrame): The input dataframe containing features and target variable.
            target (str): The name of the target variable in the dataframe.
            model (Optional[BaseEstimator]): A user-provided machine learning model for prediction. If None, a default model will be selected based on the target type.
            tree_model (Optional[BaseEstimator]): A user-provided decision tree model for extracting insights. If None, a default decision tree will be used.
        """

        self.dataframe = dataframe
        self.target = target

        self.preprocessor = None
        self.tuning = tuning
        self.model = model
        self.tree_model = tree_model
        self.model_pipeline = None
        self.tree_pipeline = None
        self.explainer = None
        self.class_names = None
        self.input_feature_names = dataframe.drop(target, axis=1).columns
        self.transformed_feature_names = None
        self.shap_values = None
        self.target_type = None

    def _select_best_model(
        self, X: pd.DataFrame, y: pd.Series, target_type: ColumnType
    ) -> BaseEstimator:
        """
        Selects the best model for the given target type (classification or regression) based on cross-validation scores between a set of candidate models and parameter grids.
        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.
            target_type (ColumnType): The type of the target variable (categorical or numerical).
            tuning (bool): Whether to perform hyperparameter tuning using RandomizedSearchCV.
        Returns:
            BaseEstimator: The best model selected based on cross-validation scores and hyperparameter tuning.
        """
        # TODO: Investigate if it makes sense to make this more gneric, acceppting candicates and scoring as parameters
        if self.model is not None:
            logger.info("Using user provided model for prediction.")
            return self.model

        if target_type == ColumnType.CATEGORICAL:
            candidate_models = CLASSIFICATION_CANDIDATES
            scoring = "accuracy"
        else:
            candidate_models = REGRESSION_CANDIDATES
            scoring = "r2"

        best_model = None
        best_score = -float("inf")
        for name, (model, param_grid) in candidate_models.items():
            pipeline = Pipeline(
                [
                    ("preprocessor", self.preprocessor),
                    ("predictor", model),
                ]
            )

            if self.tuning:
                search = RandomizedSearchCV(
                    pipeline, param_grid, cv=3, scoring=scoring, n_jobs=-1
                )
                search.fit(X, y)

                if search.best_score_ > best_score:
                    best_score = search.best_score_
                    best_model = search.best_estimator_.named_steps["predictor"]
                    best_parameters = search.best_params_
            else:  # pragma: no cover
                # Performing cross-validation without hyperparameter tuning is not recommended
                scores = cross_val_score(pipeline, X, y, cv=3, scoring=scoring)
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_parameters = {}

        logger.info(
            f"The model automatically selected is {best_model.__class__.__name__}, with the parameters {best_parameters} and a cross-validation score of {best_score:.4f}"
        )
        return best_model

    def fit(self):
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
        X = self.dataframe.drop(self.target, axis=1)
        y = self.dataframe[self.target]

        categorical_columns = [
            column
            for column in X.columns
            if self._determine_column_type(X[column]) == ColumnType.CATEGORICAL
        ]
        # time_columns = [
        #    column
        #    for column in X.columns
        #    if self._determine_column_type(X[column]) == ColumnType.TIME
        # ]
        numerical_columns = [
            column
            for column in X.columns
            if self._determine_column_type(X[column]) == ColumnType.NUMERICAL
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
            self.class_names = self.model_pipeline.named_steps["predictor"].classes_
        else:
            self.class_names = None

        # TODO: Add option for seeing the shap values for the transformed data
        # self.explainer = shap.Explainer(
        #     self.model_pipeline.named_steps["predictor"],
        #     self.model_pipeline.named_steps["preprocessor"].transform(X),
        #     feature_names=self.transformed_feature_names,
        #     output_names=self.class_names,
        # )
        self.explainer = shap.Explainer(
            lambda X: self.model_pipeline.predict_proba(X)
            if self.target_type == ColumnType.CATEGORICAL
            else self.model_pipeline.predict(X),
            X,
            feature_names=self.input_feature_names,
            output_names=self.class_names,
        )
        self.shap_values = self.explainer(X)

    def global_feature_importance(self, max_display: int = 10):
        """
        Plots the global feature importance using SHAP values.
        Args:
            max_display (int): The maximum number of features to display in the plot.
        """
        plot_global_feature_importance(
            self.shap_values,
            max_display=max_display,
            feature_names=self.input_feature_names,
            class_names=self.class_names,
        )

    def local_feature_importance(self, index: int, max_display: int = 10):
        """
        Plots the local feature importance using SHAP values.
        Args:
            index (int): The index of the instance for which to plot local feature importance
            max_display (int): The maximum number of features to display in the plot
        """
        if index < 0 or index >= len(self.dataframe):
            raise IndexError("Index out of range for the dataframe.")

        if self.target_type == ColumnType.CATEGORICAL:
            predicted_probabilities = self.model_pipeline.predict_proba(
                self.dataframe.drop(self.target, axis=1).iloc[index : index + 1]
            )
            predicted_class_index = np.argmax(predicted_probabilities)
            shap_values = self.shap_values.values[index, :, predicted_class_index]
        else:
            # TODO: Add support for regression
            shap_values = self.shap_values.values[index]

        plot_local_feature_importance(
            shap_values,
            max_display=max_display,
            feature_names=self.input_feature_names,
            class_name=self.class_names[predicted_class_index]
            if self.target_type == ColumnType.CATEGORICAL
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
        if column.dtype in ["object", "category", "bool"] or len(column.unique()) <= 10:
            return ColumnType.CATEGORICAL
        elif column.dtype == "datetime64":
            return ColumnType.TIME
        else:
            return ColumnType.NUMERICAL

    def key_segments(self, top_n: int = 5, focus_class: str = None):
        if self.target_type == ColumnType.CATEGORICAL:
            y = self.dataframe[self.target]
            class_counts = y.value_counts()
            if focus_class is None:
                focus_class = class_counts.idxmax()
            focus_class_index = list(self.class_names).index(focus_class)
            overall_mean = (y == focus_class).mean()
            tree = self.tree_pipeline[-1]
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
            overall_mean = y.mean()
            tree = self.tree_pipeline[-1]
            feature_contributions = extract_feature_contributions(
                tree, self.transformed_feature_names
            )
            rules = extract_tree_rules(tree, self.transformed_feature_names)
            insights = extract_tree_insights(
                tree,
                self.transformed_feature_names,
                overall_mean,
                "regression",
                top_n=top_n,
                target=self.target,
            )

        return feature_contributions, rules, insights
