from influencio.core import KeyInfluencers


def test_key_influencers_initialization_with_classification_data(
    sample_classification_data,
):
    target_column = "target"
    key_influencers = KeyInfluencers(sample_classification_data, target_column)
    assert key_influencers.dataframe.equals(sample_classification_data)
    assert key_influencers.target == target_column
    assert key_influencers.model_pipeline is None
    assert key_influencers.explainer is None
    assert key_influencers.transformed_feature_names is None
    assert key_influencers.shap_values is None
    assert key_influencers.target_type is None


def test_key_influencers_initialization_with_regression_data(sample_regression_data):
    target_column = "target"
    key_influencers = KeyInfluencers(sample_regression_data, target_column)
    assert key_influencers.dataframe.equals(sample_regression_data)
    assert key_influencers.target == target_column
    assert key_influencers.model_pipeline is None
    assert key_influencers.explainer is None
    assert key_influencers.transformed_feature_names is None
    assert key_influencers.shap_values is None
    assert key_influencers.target_type is None


def test_key_influencers_fit_with_classification_data(
    sample_classification_data,
):
    target_column = "target"
    key_influencers = KeyInfluencers(sample_classification_data, target_column)
    key_influencers.fit()

    assert key_influencers.model_pipeline is not None
    assert key_influencers.explainer is not None
    assert key_influencers.transformed_feature_names is not None
    assert key_influencers.shap_values is not None
    assert key_influencers.target_type == "categorical"


def test_key_influencers_fit_with_regression_data(
    sample_regression_data,
):
    target_column = "target"
    key_influencers = KeyInfluencers(sample_regression_data, target_column)
    key_influencers.fit()

    assert key_influencers.model_pipeline is not None
    assert key_influencers.explainer is not None
    assert key_influencers.transformed_feature_names is not None
    assert key_influencers.shap_values is not None
    assert key_influencers.target_type == "numerical"


def test_key_influencers_with_given_model(
    sample_classification_data, sample_classification_model
):
    target_column = "target"
    key_influencers = KeyInfluencers(
        sample_classification_data, target_column, model=sample_classification_model
    )
    key_influencers.fit()

    assert key_influencers.model == sample_classification_model
