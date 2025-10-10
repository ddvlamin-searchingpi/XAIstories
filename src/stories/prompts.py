class Partial(dict):
    def __missing__(self, key):
        # leave the placeholder in the string for future passes
        return '{' + key + '}'


def make_prompt(task_description, input_description, class_descriptions):
    if not task_description.startswith("predict"):
        raise Exception("Task description must begin with 'predict'")
    for _, class_desc in class_descriptions.items():
        if not class_desc.startswith(f"class"):
            raise Exception("Class descriptions must begin with 'class'")

    class_descriptions_str = ""
    for class_label, class_desc in class_descriptions.items():
      class_descriptions_str += f"- class label {class_label} represents the {class_desc}\n"

    prompt = """An AI model was used to {task_description}.
The input features to the model include data about {input_description}.
The target variable represents one of the following classes:
{class_descriptions_str}
The AI model predicted a certain instance of the dataset to belong to the class 
with label {predicted_label} with probability {predicted_score}.

The provided SHAP table was generated to explain this
outcome. It includes every feature along with its value for that instance, and the
SHAP value assigned to it.

The goal of SHAP is to explain the prediction of an instance by
computing the contribution of each feature to the prediction. The
SHAP explanation method computes Shapley values from coalitional game
theory. The feature values of a data instance act as players in a coalition.
Shapley values tell us how to fairly distribute the “payout”(=the prediction)
among the features. A player can be an individual feature value, e.g. for tabular
data. The scores in the table are sorted from most positive to most negative.

Can you come up with a plausible, fluent story as to why the model could have
predicted this outcome, based on the most influential positive and most influential
negative SHAP values? Focus on the features with the highest absolute
SHAP values. Try to explain the most important feature values in this story, as
well as potential interactions that fit the story. No need to enumerate individual
features outside of the story. Conclude with a short summary of why this
classification may have occurred. Limit your answer to 8 sentences.

Table containing feature values and SHAP values:
{shap_df}

Additional clarification of the features:
{feature_desc}
    """

    return prompt.format_map(Partial(task_description=task_description, input_description=input_description, class_descriptions_str=class_descriptions_str))
