import pytest
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier


@pytest.fixture
def sample_classification_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target_names[iris.target]
    return df


@pytest.fixture
def sample_regression_data():
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df["target"] = diabetes.target
    return df


@pytest.fixture
def sample_regression_model():
    model = LinearRegression()
    return model


@pytest.fixture
def sample_classification_model():
    model = LogisticRegression(max_iter=5)
    return model


@pytest.fixture
def sample_regression_tuning_candidates():
    return {
        "LinearRegression": (LinearRegression(), {}),
    }


@pytest.fixture
def sample_classification_tuning_candidates():
    return {
        "LogisticRegression": (LogisticRegression(), {}),
    }


@pytest.fixture
def sample_decision_tree_classifier():
    model = DecisionTreeClassifier()
    return model


@pytest.fixture
def fake_plot_global_feature_importance(monkeypatch):
    called = {}

    def fake_plot(shap_values, max_display, feature_names, class_names):
        called["shap_values"] = shap_values
        called["max_display"] = max_display
        called["feature_names"] = feature_names
        called["class_names"] = class_names

    import influencio.core

    monkeypatch.setattr(influencio.core, "plot_global_feature_importance", fake_plot)
    return called
