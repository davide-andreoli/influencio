import pandas as pd
import pytest
from influencio.core import KeyInfluencers


@pytest.fixture
def sample_classification_data():
    data = {
        "feature1": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "feature2": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_regression_data():
    data = {
        "feature1": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "feature2": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": [10, 20, 30, 40, 50, 60, 70, 80],
    }
    return pd.DataFrame(data)


def test_key_influencers_initialization_with_classification_data(
    sample_classification_data,
):
    target_column = "target"
    key_influencers = KeyInfluencers(sample_classification_data, target_column)
    assert key_influencers.dataframe.equals(sample_classification_data)
    assert key_influencers.target == target_column
    assert key_influencers.model_pipeline is None
    assert key_influencers.explainer is None
    assert key_influencers.feature_names is None
    assert key_influencers.shap_values is None
    assert key_influencers.target_type is None


def test_key_influencers_initialization_with_regression_data(sample_regression_data):
    target_column = "target"
    key_influencers = KeyInfluencers(sample_regression_data, target_column)
    assert key_influencers.dataframe.equals(sample_regression_data)
    assert key_influencers.target == target_column
    assert key_influencers.model_pipeline is None
    assert key_influencers.explainer is None
    assert key_influencers.feature_names is None
    assert key_influencers.shap_values is None
    assert key_influencers.target_type is None
