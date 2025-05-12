import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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
from .enums import ColumnType, TreeType


class KeyInfluencers:
    def __init__(self, dataframe: pd.DataFrame, target: str):
        self.dataframe = dataframe
        self.target = target

        self.model_pipeline = None
        self.explainer = None
        self.input_feature_names = dataframe.drop(target, axis=1).columns
        self.transformed_feature_names = None
        self.shap_values = None
        self.target_type = None

    def fit(self):
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

        self.target_type = self._determine_column_type(y)

        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        # TODO: Add automatic model choice based on performance
        if self.target_type == ColumnType.CATEGORICAL:
            predictor = LogisticRegression(solver="lbfgs", max_iter=1000)
            tree_predictor = DecisionTreeClassifier(max_depth=3)
        else:
            predictor = LinearRegression()
            tree_predictor = DecisionTreeRegressor(max_depth=3)

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
        self.shap_values = self.explainer(
            self.model_pipeline.named_steps["preprocessor"].transform(X)
        )

    def global_feature_importance(self, max_display: int = 10):
        plot_global_feature_importance(
            self.shap_values,
            max_display=max_display,
            feature_names=self.input_feature_names,
            class_names=self.class_names,
        )

    def local_feature_importance(self, index: int, max_display: int = 10):
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
