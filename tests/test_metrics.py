import pytest
import pandas as pd
from stories.metrics import rank

def test_rank():
    true_rank_df = pd.DataFrame(
        {
            "Feature name": ["f1", "f2", "f3", "f4"],
            "Shap value": [0.5, -0.2, 0.1, -0.05],
        }
    )
    extracted_rank_df = pd.DataFrame(
        {
            "Feature name": ["f1", "f2", "f5", "f3"],
            "rank": [0, 1, 2, 3],
        }
    )

    tau, _ = rank(true_rank_df, extracted_rank_df)

    assert tau == pytest.approx(1)

    true_rank_df = pd.DataFrame(
        {
            "Feature name": ["f1", "f2", "f3", "f4"],
            "Shap value": [0.5, -0.2, 0.1, -0.05],
        }
    )
    extracted_rank_df = pd.DataFrame(
        {
            "Feature name": ["f3", "f2", "f5", "f1"],
            "rank": [0, 1, 2, 3],
        }
    )

    tau, _ = rank(true_rank_df, extracted_rank_df)

    assert tau == pytest.approx(-1)
