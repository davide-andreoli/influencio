from shap import Explainer, Explanation
from sklearn.pipeline import Pipeline
import pandas as pd
from typing import Optional, Tuple, List, Literal
import logging

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Creates SHAP explainers and computes SHAP values for a fitted pipeline.
    Handles both classification and regression tasks.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        task: Literal["classification", "regression"],
        class_names: Optional[List[str]] = None,
        input_feature_names: Optional[List[str]] = None,
    ):
        """
        Args:
            pipeline (Pipeline): A fitted sklearn pipeline including preprocessing and model.
            target_type (ColumnType): The type of the target (CATEGORICAL or NUMERICAL).
            class_names (Optional[List[str]]): Class labels for classification problems.
            input_feature_names (Optional[List[str]]): Original input feature names.
        """
        self.pipeline = pipeline
        self.task = task
        self.class_names = class_names
        self.input_feature_names = input_feature_names

    def create_explainer(self, X: pd.DataFrame) -> Tuple[Explainer, Explanation]:
        """
        Creates a SHAP explainer and computes SHAP values for the dataset X.

        Args:
            X (pd.DataFrame): The original (non-transformed) feature data.

        Returns:
            Tuple[Explainer, Explanation]: A fitted SHAP explainer and SHAP values.
        """
        preprocessor = self.pipeline.named_steps["preprocessor"]
        predictor = self.pipeline.named_steps["predictor"]

        X_transformed = preprocessor.transform(X)

        def model_on_transformed(X_transformed_input):
            return (
                predictor.predict_proba(X_transformed_input)
                if self.task == "classification"
                else predictor.predict(X_transformed_input)
            )

        # TODO: Understand why class names are not passed to the explainer
        # TODO: Chose the right explainer based on model
        explainer = Explainer(
            model_on_transformed,
            X_transformed,
            feature_names=preprocessor.get_feature_names_out(self.input_feature_names),
            output_names=self.class_names,
        )

        shap_values = explainer(X_transformed)

        return explainer, shap_values
