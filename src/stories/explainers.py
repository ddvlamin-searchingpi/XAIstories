import pandas as pd
import shap
import numpy as np
from typing import Iterable, Optional

from stories.utils import get_estimator, is_pipeline, is_linear, get_pipe

def _transform_input(model, X: pd.DataFrame):
    """
    If model contains preprocessing transformers apply them to X
    """
    try:
        pipe = get_pipe(model)
        transformed_X = pipe.transform(X)
        X = pd.DataFrame(transformed_X, columns=X.columns, index=X.index)
    except ValueError:
        pass
    return X

def make_linear_interventional_shap_explainer(model, X: pd.DataFrame, max_story_features=None, only_positive_shaps=None):
    #average of original feature values
    feature_means = X.mean().to_frame().T

    X = _transform_input(model, X)
 
    estimator = get_estimator(model)
    masker = shap.maskers.Independent(X)
    explainer = shap.LinearExplainer(estimator, masker=masker)
    return ShapExplainer(model, explainer, feature_means, max_story_features, only_positive_shaps)

class ShapExplainer:
    def __init__(self, model, explainer, feature_means: Optional[pd.DataFrame] = None, max_story_features=None, only_positive_shaps=None):
        self.max_story_features = max_story_features
        self.only_positive_shaps = only_positive_shaps
        self.feature_means = feature_means
        self.model = model
        self.explainer = explainer

    def get_explanations(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _transform_input(self.model, X)

        shap_vals = self.explainer.shap_values(X)
        if len(shap_vals.shape) == 3: #model had multiple outputs
            if shap_vals.shape[2] == 2:
                shap_vals = shap_vals[:,:,1]
            else:
                raise Exception("SHAP explanations for stories are not available for a model with more than 2 outputs")
        
        shap_df = pd.DataFrame(shap_vals, columns=X.columns, index=X.index)

        return shap_df

    def get_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        """
        class_probabilities = self.model.predict_proba(X)

        if class_probabilities.shape[1] > 2:
            raise Exception("SHAP explanations for stories are not available for a model with more than 2 outputs")

        # Generate predictions and scores
        y_pred   = class_probabilities.argmax(axis=1).astype(np.int32)
        y_scores = class_probabilities.max(axis=1)
        
        predictions_df = pd.DataFrame({
            "class"  : y_pred,
            "probability" : y_scores,
        })

        return predictions_df
    
    def make_explanation_table(self, x: pd.Series) -> pd.DataFrame:
        x_df = x.to_frame().T
        feature_attributions = self.get_explanations(x_df)
        shap_values = feature_attributions.iloc[0].values

        feature_names = feature_attributions.columns
        feature_values = x_df[feature_names].iloc[0].values

        row_df = pd.DataFrame({
            "Feature name": feature_names,
            "Feature value": feature_values,
            "Shap value": shap_values
        })

        if is_linear(self.model):
            estimator = get_estimator(self.model)
            if hasattr(estimator, "coef_"):
                row_df["Model coefficient"] = estimator.coef_[0]

        if self.feature_means is not None:
            row_df["Average feature value"] = self.feature_means[feature_names].iloc[0].values

        # only take features with positive shap values
        if self.only_positive_shaps:
            row_df = row_df[row_df["Shap value"] > 0]

        sorted_row_df = row_df.sort_values(
            by="Shap value",
            key=lambda s: s.abs(),
            ascending=False)

        # only take the top self.max_story_features features
        sorted_row_df = sorted_row_df.head(self.max_story_features)

        return sorted_row_df
