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
    sample_classification_data, sample_classification_tuning_candidates
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
    sample_classification_data,
    sample_classification_model,
    sample_decision_tree_classifier,
    sample_classification_tuning_candidates,
):
    target_column = "target"
    key_influencers = KeyInfluencers(
        sample_classification_data,
        target_column,
        model=sample_classification_model,
        tree_model=sample_decision_tree_classifier,
        tuning_candidates=sample_classification_tuning_candidates,
    )
    key_influencers.fit()

    assert key_influencers.model == sample_classification_model
    assert key_influencers.tree_model == sample_decision_tree_classifier


def test_global_feature_importance_calls_plot(
    sample_classification_data,
    sample_classification_tuning_candidates,
    fake_plot_global_feature_importance,
):
    called = fake_plot_global_feature_importance

    key_influencers = KeyInfluencers(
        sample_classification_data,
        "target",
        tuning_candidates=sample_classification_tuning_candidates,
    )
    key_influencers.fit()
    key_influencers.global_feature_importance(max_display=7)

    assert "shap_values" in called
    assert called["max_display"] == 7
    assert list(called["feature_names"]) == list(key_influencers.input_feature_names)
    assert (called["class_names"] == key_influencers.class_names).all() or (
        called["class_names"] == key_influencers.class_names
    )


def test_global_feature_importance_regression(
    sample_regression_data,
    sample_regression_tuning_candidates,
    fake_plot_global_feature_importance,
):
    called = fake_plot_global_feature_importance

    key_influencers = KeyInfluencers(
        sample_regression_data,
        "target",
        tuning_candidates=sample_regression_tuning_candidates,
    )
    key_influencers.fit()
    key_influencers.global_feature_importance(max_display=5)

    assert "shap_values" in called
    assert called["max_display"] == 5
    assert list(called["feature_names"]) == list(key_influencers.input_feature_names)
    assert called["class_names"] is None
