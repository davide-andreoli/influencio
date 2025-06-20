from influencio.core import KeyInfluencers
import pytest


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


def test_key_influencers_fit_with_classification_data(sample_classification_data):
    target_column = "target"
    key_influencers = KeyInfluencers(sample_classification_data, target_column)
    key_influencers.fit()

    assert key_influencers.model_pipeline is not None
    assert key_influencers.explainer is not None
    assert key_influencers.transformed_feature_names is not None
    assert key_influencers.shap_values is not None
    assert key_influencers.target_type == "categorical"


def test_key_influencers_fit_with_regression_data(sample_regression_data):
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
):
    target_column = "target"
    key_influencers = KeyInfluencers(
        sample_classification_data,
        target_column,
        model=sample_classification_model,
        tree_model=sample_decision_tree_classifier,
    )
    key_influencers.fit()

    assert key_influencers.model == sample_classification_model
    assert key_influencers.tree_model == sample_decision_tree_classifier


def test_key_influencers_with_given_tuning_candidates(
    sample_classification_data, sample_classification_tuning_candidates
):
    target_column = "target"
    key_influencers = KeyInfluencers(
        sample_classification_data,
        target_column,
        tuning_candidates=sample_classification_tuning_candidates,
    )
    key_influencers.fit()

    assert key_influencers.tuning_candidates == sample_classification_tuning_candidates
    assert key_influencers.model_pipeline is not None


def test_global_feature_importance_calls_plot(
    sample_classification_data,
    sample_classification_model,
    fake_plot_global_feature_importance,
):
    called = fake_plot_global_feature_importance

    key_influencers = KeyInfluencers(
        sample_classification_data, "target", model=sample_classification_model
    )
    key_influencers.fit()
    key_influencers.global_feature_importance(max_display=7)

    assert "shap_values" in called
    assert called["max_display"] == 7
    assert list(called["feature_names"]) == list(key_influencers.input_feature_names)
    assert called["class_names"] == key_influencers.class_names


def test_global_feature_importance_regression(
    sample_regression_data,
    sample_regression_model,
    fake_plot_global_feature_importance,
):
    called = fake_plot_global_feature_importance

    key_influencers = KeyInfluencers(
        sample_regression_data,
        "target",
        model=sample_regression_model,
    )
    key_influencers.fit()
    key_influencers.global_feature_importance(max_display=5)

    assert "shap_values" in called
    assert called["max_display"] == 5
    assert list(called["feature_names"]) == list(key_influencers.input_feature_names)
    assert called["class_names"] is None


def test_local_feature_importance_calls_plot(
    sample_classification_data,
    sample_classification_model,
    fake_plot_local_feature_importance,
):
    called = fake_plot_local_feature_importance

    key_influencers = KeyInfluencers(
        sample_classification_data,
        "target",
        model=sample_classification_model,
    )
    key_influencers.fit()
    key_influencers.local_feature_importance(index=0, max_display=7)

    assert "shap_values" in called
    assert called["max_display"] == 7
    assert list(called["feature_names"]) == list(key_influencers.input_feature_names)


def test_local_feature_importance_regression_data(
    sample_regression_data,
    sample_regression_model,
    fake_plot_local_feature_importance,
):
    called = fake_plot_local_feature_importance

    key_influencers = KeyInfluencers(
        sample_regression_data,
        "target",
        model=sample_regression_model,
    )
    key_influencers.fit()
    key_influencers.local_feature_importance(index=0, max_display=7)

    assert "shap_values" in called
    assert called["max_display"] == 7
    assert list(called["feature_names"]) == list(key_influencers.input_feature_names)


def test_local_feature_importance_invalid_index(
    sample_classification_data,
    sample_classification_model,
):
    key_influencers = KeyInfluencers(
        sample_classification_data,
        "target",
        model=sample_classification_model,
    )
    key_influencers.fit()
    with pytest.raises(IndexError) as e_info:
        key_influencers.local_feature_importance(index=999999, max_display=7)

    assert str(e_info.value) == "Index out of range for the dataframe."


def test_key_segments_classification_default_focus_class(
    sample_classification_data, sample_classification_model
):
    key_influencers = KeyInfluencers(
        sample_classification_data, "target", model=sample_classification_model
    )
    key_influencers.fit()
    feature_contributions, rules, insights = key_influencers.key_segments(top_n=3)
    assert isinstance(feature_contributions, list)
    assert isinstance(rules, list)
    assert isinstance(insights, list)
    assert len(insights) <= 3


def test_key_segments_classification_with_focus_class(
    sample_classification_data, sample_classification_model
):
    key_influencers = KeyInfluencers(
        sample_classification_data, "target", model=sample_classification_model
    )
    key_influencers.fit()
    focus_class = sample_classification_data["target"].unique()[0]
    feature_contributions, rules, insights = key_influencers.key_segments(
        top_n=2, focus_class=focus_class
    )
    assert isinstance(feature_contributions, list)
    assert isinstance(rules, list)
    assert isinstance(insights, list)
    assert len(insights) <= 2


def test_key_segments_regression(sample_regression_data, sample_regression_model):
    key_influencers = KeyInfluencers(
        sample_regression_data, "target", model=sample_regression_model
    )
    key_influencers.fit()
    feature_contributions, rules, insights = key_influencers.key_segments(top_n=4)
    assert isinstance(feature_contributions, list)
    assert isinstance(rules, list)
    assert isinstance(insights, list)
    assert len(insights) <= 4


def test_key_segments_regression_with_target_arg(
    sample_regression_data, sample_regression_model
):
    key_influencers = KeyInfluencers(
        sample_regression_data, "target", model=sample_regression_model
    )
    key_influencers.fit()
    feature_contributions, rules, insights = key_influencers.key_segments(top_n=1)
    assert isinstance(feature_contributions, list)
    assert isinstance(rules, list)
    assert isinstance(insights, list)
    assert len(insights) <= 1
