import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import shap
from .visualizations import plot_global_feature_importance, plot_local_feature_importance, plot_segments_tree

class KeyInfluencers():

    def __init__(self, dataframe: pd.DataFrame, target: str):

        self.dataframe = dataframe
        self.target = target

        self.model_pipeline = None
        self.explainer = None
        self.feature_names = None
        self.shap_values = None
        self.target_type = None
    
    def fit(self):
        
        X = self.dataframe.drop(self.target, axis=1)
        y = self.dataframe[self.target]

        categorical_columns = [column for column in X.columns if self._determine_column_type(X[column]) == 'categorical']
        time_columns = [column for column in X.columns if self._determine_column_type(X[column]) == 'time']
        numerical_columns = [column for column in X.columns if self._determine_column_type(X[column]) == 'numerical']

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', StandardScaler())
        ])

        #TODO: Add preprocessor for time data

        preprocessor = ColumnTransformer([
            ('categorical', categorical_pipeline, categorical_columns),
            ('numerical', numerical_pipeline, numerical_columns)
        ])

        self.target_type = self._determine_column_type(y)

        # X_train, X_test, y_train, y_test = train_test_split(X, y)
        # TODO: Add automatic model choice based on performance
        if self.target_type == 'categorical':
            predictor = LogisticRegression(solver='lbfgs', max_iter=1000)
            tree_predictor = DecisionTreeClassifier(max_depth=3)
        else:
            predictor = LinearRegression()
            tree_predictor = DecisionTreeRegressor(max_depth=3)

        self.model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('predictor', predictor)
        ])

        self.tree_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('tree', tree_predictor)
        ])

        self.model_pipeline.fit(X, y)
        self.tree_pipeline.fit(X, y)

        self.feature_names = self.model_pipeline.named_steps['preprocessor'].get_feature_names_out()
        if self.target_type == 'categorical':
            self.class_names = self.model_pipeline.named_steps['predictor'].classes_
        else:
            self.class_names = None
        self.explainer = shap.Explainer(self.model_pipeline.named_steps['predictor'], self.model_pipeline.named_steps['preprocessor'].transform(X), feature_names=self.feature_names, output_names=self.class_names)
        self.shap_values = self.explainer(self.model_pipeline.named_steps['preprocessor'].transform(X))

    def global_feature_importance(self, max_display=10):

        plot_global_feature_importance(self.shap_values, max_display=max_display, feature_names=self.feature_names, class_names=self.class_names)

    def local_feature_importance(self, index: int, max_display=10):

        if index < 0 or index >= len(self.dataframe):
            raise IndexError("Index out of range for the dataframe.")

        if self.target_type == 'categorical':
            predicted_probabilities = self.model_pipeline.predict_proba(self.dataframe.drop(self.target, axis=1).iloc[index:index+1])
            predicted_class_index = np.argmax(predicted_probabilities)
            shap_values = self.shap_values.values[index, :, predicted_class_index]
        else:
            #TODO: Add support for regression
            shap_values = self.shap_values.values[index]

        plot_local_feature_importance(shap_values, max_display=max_display, feature_names=self.feature_names)

    def plot_segments(self):
        tree = self.tree_pipeline.named_steps['tree']
        plot_segments_tree(tree, self.feature_names, self.target)

    def _determine_column_type(self, column: pd.Series):
        #TODO: make this an enum
        if column.dtype in ['object', 'category', 'bool'] or len(column.unique()) <= 10:
            return 'categorical'
        elif column.dtype == 'datetime64' :
            return 'time'
        else:
            return 'numerical'