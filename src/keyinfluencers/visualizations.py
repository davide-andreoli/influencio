import numpy as np
import matplotlib.pyplot as plt
from typing import List
import plotly.express as px
import pandas as pd

def plot_feature_importance(shap_values, feature_names: List[str], class_names: List[str],  max_display=10):
    feature_class_importance = np.sum(np.abs(shap_values.values), axis=0)
    total_importance = np.sum(feature_class_importance, axis=1)
    sorted_indices = np.argsort(-total_importance)

    feature_class_importance = feature_class_importance[sorted_indices]
    feature_names_sorted = [feature_names[i] for i in sorted_indices]

    if max_display is not None and max_display < len(feature_names_sorted):
        feature_class_importance = feature_class_importance[:max_display]
        feature_names_sorted = feature_names_sorted[:max_display]

    data = pd.DataFrame(feature_class_importance, columns=class_names, index=feature_names_sorted)
    data = data.reset_index().melt(id_vars="index", var_name="Class", value_name="Importance")
    data.rename(columns={"index": "Feature"}, inplace=True)

    #TODO: Format the values to be more readable
    fig = px.bar(
        data,
        x="Feature",
        y="Importance",
        color="Class",
        title="Stacked Feature Importances by Class (SHAP)",
        labels={"Feature": "Feature", "Importance": "Total SHAP Importance"},
        text="Importance"
    )
    fig.show()





def plot_feature_importance_legacy(shap_values, feature_names: List[str], class_names: List[str],  max_display=10):
    feature_class_importance = np.sum(np.abs(shap_values.values), axis=0)
    total_importance = np.sum(feature_class_importance, axis=1)
    sorted_indices = np.argsort(-total_importance)

    feature_class_importance = feature_class_importance[sorted_indices]
    feature_names_sorted = [feature_names[i] for i in sorted_indices]

    #TODO: limit to max_display features   


    x = np.arange(len(feature_names))
    bottom = np.zeros(len(feature_names))

    #TODO: plot using seaborn or plotly for better aesthetics
    plt.figure(figsize=(10, 6))

    for class_idx in range(len(class_names)):
        values = feature_class_importance[:, class_idx]
        plt.bar(
            x,
            values,
            bottom=bottom,
            label=class_names[class_idx]
        )
        bottom += values  

    plt.xticks(x, feature_names_sorted, rotation=45)
    plt.ylabel("Total SHAP Importance")
    plt.title("Stacked Feature Importances by Class (SHAP)")
    plt.legend()
    plt.tight_layout()
    plt.show()