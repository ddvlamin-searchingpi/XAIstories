import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
from PIL import Image

from sklearn.pipeline import Pipeline
import sklearn.model_selection as model_selection
from sklearn.linear_model._base import LinearModel, LinearClassifierMixin
from sklearn.tree import BaseDecisionTree
from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
from sklearn.ensemble._gb import BaseGradientBoosting

TREE_BASES = (
    BaseDecisionTree,
    ForestClassifier,
    ForestRegressor,
    BaseGradientBoosting,
)

LINEAR_BASES = (LinearModel, LinearClassifierMixin)

def unwrap(est):
    # Grid/Randomized/Halving search objects
    if hasattr(est, "best_estimator_"):
        return unwrap(est.best_estimator_)
    # Pipeline or FeatureUnion expose final steps via steps / transformer_list
    if isinstance(est, Pipeline):
        # last step is a (name, estimator) tuple
        return unwrap(est.steps[-1][1])
    if hasattr(est, "estimator"):  # e.g., CalibratedClassifierCV
        return unwrap(est.estimator)
    return est

def is_linear(est):
    est = unwrap(est)
    return isinstance(est, LINEAR_BASES)

def is_tree(est):
    est = unwrap(est)
    if isinstance(est, TREE_BASES):
        return True
    # ensembles expose base estimators through `estimators_`
    return hasattr(est, "estimators_") and all(isinstance(e, BaseDecisionTree) for e in est.estimators_)

def is_pipeline(est):
    return isinstance(est, Pipeline)

class SHAPstory():

  def __init__(self, model, explainer, llm, prompt_template, feature_desc, max_story_features=50,
               only_positive_shaps=False, include_linear_model_coefficients=False):
    """Initializes the SHAPstory class with necessary parameters.
    
    Arguments:
    ----------
    model : object
        A trained model which supports SHAP explanations.
    explainer : object
        A trained explainer object from SHAP.
    llm : object
        An instance of the LLMWrapper class.
    prompt_template : str
        Prompt string with placeholders for predicted_label, predicted_score, shap_df, feature_desc
    """
    self.llm = llm
    self.explainer = explainer
    self.model = model
    self.max_story_features = max_story_features
    self.only_positive_shaps = only_positive_shaps
    self.include_linear_model_coefficients = include_linear_model_coefficients
    self.prompt_template = prompt_template

    #unwrap the estimator from the grid_search and/or pipeline wrapper
    #if the wrapped model is linear, the shap values can be generated using the linear explainer instead of a more generic one
    self.estimator = unwrap(self.model)

    self.feature_desc = feature_desc.set_index("Feature name")

  def gen_shap_feature_df(self, x):
    if is_pipeline(self.model):
       transformed_x = self.model[:-1].transform(x)
       x = pd.DataFrame(transformed_x, columns=x.columns, index=x.index)

    shap_vals = self.explainer.shap_values(x)
    if len(shap_vals.shape) == 3: #model had multiple outputs
       if shap_vals.shape[2] == 2:
          shap_vals = shap_vals[:,:,1]
       else:
          raise Exception("SHAP stories are not available for a model with more than 2 outputs")
    
    shap_df = pd.DataFrame(shap_vals, columns=x.columns, index=x.index)

    return shap_df

  def gen_variables(self, x):
    """
    Generate necessary variables including SHAP values and predictions.
    
    Parameters:
    -----------
    model : object
        A trained model which supports SHAP explanations.
    x : DataFrame
        The input data.
    """
    class_probabilities = self.model.predict_proba(x)

    if class_probabilities.shape[1] > 2:
      raise Exception("SHAP stories are not available for a model with more than 2 outputs")

    # Generate table with shap values for all instances given
    shap_feature_df = self.gen_shap_feature_df(x)

    # Generate predictions and scores
    y_pred   = class_probabilities.argmax(axis=1).astype(np.int32)
    y_scores = class_probabilities.max(axis=1)
    
    predictions_df = pd.DataFrame({
      "class"  : y_pred,
      "probability" : y_scores,
    })

    return shap_feature_df, predictions_df

  def generate_prompt(self, x: pd.Series, prediction, score, shap_values):
    """
    Generates the prompt for the provided LLM to generate a narrative.
    
    Parameters:
    -----------
    iloc_pos : int
        Index position of the instance for which the prompt is generated.
        
    Returns:
    --------
    str
        The generated prompt.
    """
    feature_values = list(x.values)
    feature_names = list(x.index)

    row_df = pd.DataFrame({
      "Feature name" : feature_names,
      "Feature value"   : feature_values,
      "Shap value"    : shap_values
    })
    include_coefficient = False
    if is_linear(self.estimator) and hasattr(self.estimator, "coef_") and self.include_linear_model_coefficients:
      row_df["Model coefficient"] = self.estimator.coef_[0]
      include_coefficient = True

    #only take features with positive shap values
    if self.only_positive_shaps:
      row_df = row_df[row_df["Shap value"] > 0]

    sorted_row_df = row_df.sort_values(
       by="Shap value", 
       key=lambda s: s.abs(), 
       ascending=False)

    #only take the top self.max_story_features features
    sorted_row_df = sorted_row_df.head(self.max_story_features)

    #sort and select rows in self.feature_desc the same way
    new_feature_desc = self.feature_desc.reindex(sorted_row_df["Feature name"])

    prompt_string = self.prompt_template.format(
      predicted_label = int(prediction),
      predicted_score = format(score, ".2f"),
      shap_df = sorted_row_df.to_string(index=False),
      feature_desc = new_feature_desc.to_string(index=True),
    )
    
    return prompt_string

  def generate_response(self, prompt):
    return self.llm.generate_response(prompt)

  def generate_stories(self, x):
    """
    Generates SHAPstories for each instance in the given data.
    
    Parameters:
    -----------
    model : object
        A trained model which supports SHAP explanations.
    x : DataFrame
        The input data.
    
    Returns:
    --------
    list of str
        A list containing the generated SHAPstories for each instance.
    """
    shap_df, predictions_df = self.gen_variables(x)

    stories = []
    for i in range(len(x)):
      x_values = x.iloc[i]
      shap_values = shap_df.iloc[i].values
      prediction, score = predictions_df.iloc[i].values

      prompt = self.generate_prompt(x_values, prediction, score, shap_values)

      story = self.generate_response(prompt)

      stories.append(story)

    return stories