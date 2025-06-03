import pytest
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression


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
