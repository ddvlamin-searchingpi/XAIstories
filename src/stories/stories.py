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

def get_feature_scaler(est, name_scaler = "scaler"):
  if is_pipeline(est):
    return est.named_steps[name_scaler]
  else:
    return None
  
def has_feature_scaler(est, name_scaler = "scaler"):
  return get_feature_scaler(est, name_scaler) is not None

class SHAPstory():

  def __init__(self, model, explainer, llm, feature_desc, task_description, 
               input_description, class_descriptions):
    """Initializes the SHAPstory class with necessary parameters."""
    if not task_description.startswith("predict"):
       raise Exception("Task description must begin with 'predict'")
    for _, class_desc in class_descriptions.items():
       if not class_desc.startswith(f"class"):
          raise Exception("Class descriptions must begin with 'class'")

    self.feature_desc = feature_desc
    self.task_description = task_description
    self.input_description = input_description
    self.class_descriptions = class_descriptions
    self.llm = llm
    self.explainer = explainer
    self.model = model
    self.estimator = unwrap(self.model)

    self.class_descriptions_str = ""
    for class_label, class_desc in class_descriptions.items():
      self.class_descriptions_str += f"- class label {class_label} represents the {class_desc}\n"

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

  def generate_prompt(self, x, predictions_df, shap_df, iloc_pos):
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

    prediction, score = predictions_df.iloc[iloc_pos].values
    shap_values = shap_df.iloc[iloc_pos].values
    feature_values = x.iloc[iloc_pos].values
    feature_names = shap_df.columns

    row_df = pd.DataFrame({
      "Feature name" : feature_names,
      "Feature value"   : feature_values,
      "Shap value"    : shap_values
    })
    include_coefficient = False
    if is_linear(self.estimator) and hasattr(self.estimator, "coef_"):
      row_df["Model coefficient"] = self.estimator.coef_[0]
      include_coefficient = True

    sorted_row_df = row_df.sort_values(
       by="Shap value", 
       key=lambda s: s.abs(), 
       ascending=False)

    prompt_string = f"""
An AI model was used to {self.task_description}. 
The input features to the model include data about {self.input_description}. 
The target variable represents one of the following classes:
{self.class_descriptions_str}
The AI model predicted a certain instance of the dataset to belong to the class with label {int(prediction)} 
(i.e. {self.class_descriptions[prediction]}) with probability {score:.2%}. 

The provided SHAP table was generated to explain this
outcome. It includes every feature along with its value for that instance, and the
SHAP value assigned to it. 

The goal of SHAP is to explain the prediction of an instance by 
computing the contribution of each feature to the prediction. The
SHAP explanation method computes Shapley values from coalitional game
theory. The feature values of a data instance act as players in a coalition.
Shapley values tell us how to fairly distribute the “payout” (= the prediction)
among the features. A player can be an individual feature value, e.g. for tabular
data. The scores in the table are sorted from most positive to most negative.

Can you come up with a plausible, fluent story as to why the model could have
predicted this outcome, based on the most influential positive and most influential
negative SHAP values? Focus on the features with the highest absolute
SHAP values. Try to explain the most important feature values in this story, as
well as potential interactions that fit the story. No need to enumerate individual
features outside of the story. Conclude with a short summary of why this
classification may have occurred. Limit your answer to 8 sentences.

Table containing feature values and SHAP values{" and model coefficients" if include_coefficient is not None else ""}:
{sorted_row_df.to_string(index=False)}

Additional clarification of the features:
{self.feature_desc.to_string(index=False)}
    """
    
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

    stories = [self.generate_response(self.generate_prompt(x, predictions_df, shap_df, i)) for i in range(len(x))]

    return stories