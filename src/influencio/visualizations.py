import numpy as np
from typing import List, Optional
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from shap._explanation import Explanation


def plot_global_feature_importance(
    shap_values: Explanation,
    feature_names: List[str],
    class_names: Optional[List[str]],
    max_display: int = 10,
):
    feature_class_importance = np.sum(np.abs(shap_values.values), axis=0)
    total_importance = np.sum(feature_class_importance, axis=1)
    sorted_indices = np.argsort(-total_importance)

    feature_class_importance = feature_class_importance[sorted_indices]
    feature_names_sorted = [feature_names[i] for i in sorted_indices]

    if max_display is not None and max_display < len(feature_names_sorted):
        feature_class_importance = feature_class_importance[:max_display]
        feature_names_sorted = feature_names_sorted[:max_display]

    data = pd.DataFrame(
        feature_class_importance, columns=class_names, index=feature_names_sorted
    )
    data = data.reset_index().melt(
        id_vars="index", var_name="Class", value_name="Importance"
    )
    data.rename(columns={"index": "Feature"}, inplace=True)

    # TODO: Format the values to be more readable
    fig = px.bar(
        data,
        x="Feature",
        y="Importance",
        color="Class",
        title="Stacked Feature Importances by Class (SHAP)",
        labels={"Feature": "Feature", "Importance": "Total SHAP Importance"},
        text="Importance",
    )
    fig.show()


def plot_local_feature_importance(
    shap_values: Explanation,
    feature_names: List[str],
    max_display=10,
    class_name: Optional[str] = None,
):
    graph_title = (
        "Local Feature Importance"
        if class_name is None
        else f"Local Feature Importance for Class {class_name}"
    )

    sorted_indices = np.argsort(-np.abs(shap_values))
    feature_class_importance = shap_values[sorted_indices]
    feature_names_sorted = [feature_names[i] for i in sorted_indices]

    if max_display is not None and max_display < len(feature_names_sorted):
        remaining_features = feature_class_importance[max_display:]
        feature_class_importance = np.concatenate(
            (feature_class_importance[:max_display], [np.sum(remaining_features)])
        )
        feature_names_sorted = feature_names_sorted[:max_display] + ["Other"]

    fig = go.Figure(
        go.Waterfall(
            name="Feature Importance for Class",
            orientation="h",
            measure=["relative" for _ in feature_class_importance],
            y=feature_names_sorted,
            x=feature_class_importance,
            connector={
                "mode": "between",
                "line": {"width": 4, "color": "rgb(0, 0, 0)", "dash": "solid"},
            },
        )
    )

    fig.update_layout(title=graph_title)

    fig.show()
