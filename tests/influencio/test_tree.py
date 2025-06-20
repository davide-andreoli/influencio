# tests/test_tree.py
from influencio.tree import extract_feature_contributions, extract_tree_rules
from influencio.core import KeyInfluencers


def test_tree_feature_contributions(sample_classification_data):
    ki = KeyInfluencers(sample_classification_data, "target")
    ki.fit()
    assert ki.transformed_feature_names is not None
    assert ki.tree_pipeline is not None
    tree = ki.tree_pipeline[-1]
    contributions = extract_feature_contributions(tree, ki.transformed_feature_names)  # pyright: ignore[reportArgumentType]
    assert isinstance(contributions, list)
    assert all(isinstance(c, tuple) for c in contributions)


def test_tree_rules_extraction(sample_classification_data):
    ki = KeyInfluencers(sample_classification_data, "target")
    ki.fit()
    assert ki.transformed_feature_names is not None
    assert ki.tree_pipeline is not None
    tree = ki.tree_pipeline[-1]
    rules = extract_tree_rules(tree, ki.transformed_feature_names)  # pyright: ignore[reportArgumentType]
    assert isinstance(rules, list)
