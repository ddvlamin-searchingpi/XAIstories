class Partial(dict):
    def __missing__(self, key):
        # leave the placeholder in the string for future passes
        return '{' + key + '}'

narrative_prompt_template = """An AI model was used to {task_description}. The input features to the model include data about {input_description}. The target variable represents one of the following classes:
{class_descriptions_str}
The AI model predicted a certain instance of the dataset to belong to the class with label {predicted_label} with probability {predicted_score}.

The post-hoc feature attribution method used to explain the instance's predicted class is {attribution_method}. {attribution_method_description}

Table containing feature values, {attribution_method} values, feature descriptions and more:
{feature_attribution_table}

Can you come up with a plausible and fluent story as to why the model could have predicted this outcome, based on the most influential positive and most influential negative {attribution_method} values relative to class 1? Focus on the features with the highest absolute {attribution_method} values. Try to explain the most important feature values in this story, as well as potential interactions that fit the story. In the story you can refer to the actual feature values. No need though to enumerate individual features outside of the story. Limit your answer to 8 sentences.
"""

def make_shap_narrative_prompt(task_description, input_description, class_descriptions):
    if not task_description.startswith("predict"):
        raise Exception("Task description must begin with 'predict'")
    for _, class_desc in class_descriptions.items():
        if not class_desc.startswith(f"class"):
            raise Exception("Class descriptions must begin with 'class'")

    class_descriptions_str = ""
    for class_label, class_desc in class_descriptions.items():
      class_descriptions_str += f"- class label {class_label} represents the {class_desc}\n"

    shap_explanation = """The goal of SHAP is to explain the prediction of an instance by computing the contribution of each feature to the prediction. Each individual SHAP value is a measure of how much additional probability this feature adds or subtracts  in the predicted probability relative to the base level probability. This relative nature of the SHAP values might have unexpected consequences that you are to take into account. For example, features that should intuitively contribute in a positive way (and vice versa), can still have negative SHAP values if their value is below an average in the dataset.

The provided feature attribution table is generated to explain why the instance is predicted to be or not be of the target class 1. It includes every feature along with its value for that instance, the SHAP value assigned to it etc.
    """

    return narrative_prompt_template.format_map(Partial(
        task_description=task_description, 
        input_description=input_description, 
        class_descriptions_str=class_descriptions_str,
        attribution_method="SHAP",
        attribution_method_description=shap_explanation
    ))


evaluation_prompt_template = """An LLM was used to create a narrative to explain and interpret a prediction made by another smaller classifier model. The LLM was given an explanation of  the classifier task, the training data, and provided with the exact names of all the features and their meaning. Most importantly, the LLM was provided with a table that contains the feature values of that particular instance, the average feature values and their SHAP values which are a numeric measure of their importance. Here is some general info about the task:

The classifier model was used to {task_description}.
The input features to the model include data about {input_description}.
The target variable represents one of the following classes:
{class_descriptions_str}

The LLM returned the following narrative: {narrative}

The prompt also included the following table with the feature names and descriptions:
{feature_desc}

Your task is to extract {evaluation_task_description}

Make sure to use the exact names of the features as provided in the table, including capitalization.
"""

def make_rank_evaluation_prompt(task_description, input_description, class_descriptions, narrative, feature_desc):
    if not task_description.startswith("predict"):
        raise Exception("Task description must begin with 'predict'")
    for _, class_desc in class_descriptions.items():
        if not class_desc.startswith(f"class"):
            raise Exception("Class descriptions must begin with 'class'")

    class_descriptions_str = ""
    for class_label, class_desc in class_descriptions.items():
      class_descriptions_str += f"- class label {class_label} represents the {class_desc}\n"

    evaluation_task_description = """a rank for each feature indicating the importance and order of their occurence in the narrative. Please just provide the JSON dictionary and add nothing else to the answer. Example {{"0": "feature_name", "1": "feature_name", "2": "feature_name"}}."""

    return evaluation_prompt_template.format(
        task_description=task_description, 
        input_description=input_description, 
        class_descriptions_str=class_descriptions_str,
        narrative=narrative,
        evaluation_task_description=evaluation_task_description,
        feature_desc=feature_desc
    )
