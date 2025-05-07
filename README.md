# Influencio

Influencio is a small explorative package with a simple premise: find out what features are influencing a target variable the most in a given dataset.

## Features

As of right now, Influencio offers the following features:

- global explanation of feature importance, given a dataset and a target
- local explanation of feature importance, given a dataset, a target and an observation
- high level inisghts extraction: this is still experimental and not fleshed out

## Internals

The way Influencio works is pretty simple:

1. Takes in a raw dataset (not feature engineered)
2. Performs basic feature engineering
3. Models the target variable: depending on the type of target, the model can be a regression model or a classification model.

It uses two modelling approaches:

- standard mode: for SHAP-based global/local feature importances
- tree model: to extract key segments influencing the target variable

## Why use Influencio?

Why not just model your data yourself and use the `shap` library directly?

To be honest, if you can to that, you definitely should; you'll have way more control over:

- How features are engineered
- Which models are applied

But if you lack the time or skills to dive deep into this kind of analysis, Influencio has your back. It automates the hard parts so you can focus on the insights.

## Hoe to use it

The interface is pretty simple, you just have to open a Jupyter Notebook, import the library, instantiate a KeyInfluencers object with the dataframe and the name of the target column, and call its fit method

```python
from influencio.keyinfluencers import KeyInfluencers

ki = KeyInfluencers(df, "target")
ki.fit()
```

At this point, you have two methods available:

- `ki.global_feature_importance()` --> it outputs a global feature importance plot
- `ki.local_feature_importance(index: int, max_display: int = 10)` --> it outputs the local feature importance graph for the given index, displaying the top `max_display` features and collapsing the other into one line

## Roadmap

- Support time features
- Implement model evaluation
- Implement model automatic selection between a list of models
- Implement hyper parameter optimization for the chosen model
- Give option to skip pre processing
- Flesh out the tree insights
- Flesh out and improve graphs
- Summary report
