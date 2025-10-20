import numpy as np
import pandas as pd

SHAP_TABLE_COLUMNS = [
    "Feature name",
    "Feature value",
    "Average feature value",
    "Feature description",
    "Shap value"
]

class SHAPstory():

  def __init__(self, explainer, llm, prompt_template, feature_desc, feature_attribution_table_columns=SHAP_TABLE_COLUMNS):
    """Initializes the SHAPstory class with necessary parameters.
    
    Arguments:
    ----------
    explainer : object
        A trained explainer object from SHAP.
    llm : object
        An instance of the LLMWrapper class.
    prompt_template : str
        Prompt string with placeholders for predicted_label, predicted_score, shap_df, feature_desc
    """
    self.llm = llm
    self.explainer = explainer
    self.prompt_template = prompt_template
    self.feature_attribution_table_columns = feature_attribution_table_columns

    self.feature_desc = feature_desc.set_index("Feature name")

  def generate_prompt(self, prediction, score, feature_attribution_table: pd.DataFrame):
    """
    """
    prompt_string = self.prompt_template.format(
      predicted_label = int(prediction),
      predicted_score = format(score, ".2f"),
      feature_attribution_table=feature_attribution_table.to_string(
          index=False),
    )
    
    return prompt_string

  def generate_response(self, prompt):
    return self.llm.generate_response(prompt)

  def generate_stories(self, X):
    """
    """
    predictions_df = self.explainer.get_predictions(X)

    stories = []
    for i in range(len(X)):
      x_values = X.iloc[i]
      _, _, target_probability = predictions_df.iloc[i].values
      explanation_table = self.explainer.make_explanation_table(x_values)
      explanation_table = explanation_table.merge(self.feature_desc, how="left", on="Feature name")

      prompt = self.generate_prompt(
          self.explainer.model.classes_[-1], 
          target_probability,
          explanation_table[self.feature_attribution_table_columns]
      )

      story = self.generate_response(prompt)

      stories.append({"explanation_table": explanation_table, "story": story, "generation_prompt": prompt})

    return stories