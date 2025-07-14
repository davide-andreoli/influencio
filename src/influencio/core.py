import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
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
from .candidates import CLASSIFICATION_CANDIDATES, REGRESSION_CANDIDATES
from .enums import ColumnType, TreeType
from typing import cast, Optional, Tuple, Dict, Any, List, Union
import logging
from joblib import Parallel, delayed

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

    def _evaluate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        name: str,
        model: Union[ClassifierMixin, RegressorMixin],
        scoring: str,
        param_grid: Dict[str, Any],
    ) -> Tuple[
        float, Union[ClassifierMixin, RegressorMixin], Dict[str, Any], str
    ]:  # pragma: no cover
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
            return (
                search.best_score_,
                search.best_estimator_.named_steps["predictor"],  # type: ignore
                search.best_params_,
                name,
            )
        else:
            scores = cross_val_score(pipeline, X, y, cv=3, scoring=scoring)
            return (
                float(np.mean(scores)),
                model,
                {},
                name,
            )

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
        # TODO: Investigate if it makes sense to make this more gneric, acceppting candicates and scoring as parameters
        if self.model is not None:
            logger.info("Using user provided model for prediction.")
            return self.model

        if target_type == ColumnType.CATEGORICAL:
            candidate_models: Dict[
                str, Tuple[Union[ClassifierMixin, RegressorMixin], Dict[str, Any]]
            ] = (
                CLASSIFICATION_CANDIDATES
                if not self.tuning_candidates
                else self.tuning_candidates
            )
            scoring = "accuracy"
        elif target_type == ColumnType.NUMERICAL:
            candidate_models: Dict[
                str, Tuple[Union[ClassifierMixin, RegressorMixin], Dict[str, Any]]
            ] = (
                REGRESSION_CANDIDATES
                if not self.tuning_candidates
                else self.tuning_candidates
            )
            scoring = "r2"
        else:
            raise ValueError(
                "Unsupported target type. Only categorical and numerical targets are supported."
            )

        best_model = None
        best_score = -float("inf")

        tasks = [
            delayed(self._evaluate_model)(X, y, name, model, scoring, param_grid)
            for name, (model, param_grid) in candidate_models.items()
        ]

        results = cast(
            List[
                Tuple[
                    float, Union[ClassifierMixin, RegressorMixin], Dict[str, Any], str
                ]
            ],
            Parallel(n_jobs=-1)(tasks),
        )

        best_score, best_model, best_parameters, best_name = max(
            results, key=lambda x: x[0]
        )

        logger.info(
            f"The model automatically selected is {best_name}, with the parameters {best_parameters} and a cross-validation score of {best_score:.4f}"
        )
        return best_model

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
