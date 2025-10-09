import shap
import pandas as pd
import numpy as np
from pathlib import Path

import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#fixture for processing the data
@pytest.fixture(scope="module")
def model_and_test_data():
    data_path = Path(__file__).resolve().parent.parent / \
        "data" / "FIFA_2018_Statistics.csv"
    data = pd.read_csv(data_path)

    feature_names = [
        i for i in data.columns if data[i].dtype in [np.int64, np.int64]]

    x = data[feature_names]
    y = (data["Man of the Match"] == "Yes")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=5000)
    model.fit(x_train, y_train)

    return model, x_test


def test_additive_shap_values(model_and_test_data):
    model, x_test = model_and_test_data
 
    masker = shap.maskers.Independent(x_test)
    explainer = shap.LinearExplainer(model, masker=masker)
    shap_values = explainer.shap_values(x_test)

    scores = model.decision_function(x_test)
    print(scores[0])
    print(scores.mean())
    print(scores[0]-scores.mean())
    print(shap_values[0, :].sum())

    #check almost the same
    assert np.isclose(scores[0]-scores.mean(), shap_values[0, :].sum())


def test_linear_shap_values(model_and_test_data):
    model, x_test = model_and_test_data

    masker = shap.maskers.Independent(x_test)
    explainer = shap.LinearExplainer(model, masker=masker)
    shap_values = explainer.shap_values(x_test)

    for col in x_test.columns:
        idx = x_test.columns.get_loc(col)
        computed_shap = model.coef_[0][idx] * (x_test.iloc[0][col] - x_test[col].mean())
        assert np.isclose(computed_shap, shap_values[0][idx])


