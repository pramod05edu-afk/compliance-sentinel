import numpy as np
import pandas as pd
import pytest

from src.fairness import four_fifths_ratio, group_fairness


def _frame():
    """Hand-checkable micro-dataset: dept A is treated fairly, dept B is not.

    A: 2 true violations, both flagged, no false alarms  -> TPR 1.0, FPR 0.0
    B: 1 true violation flagged + 2 of 3 innocents flagged -> TPR 1.0, FPR 0.667
    """
    return pd.DataFrame({
        "department":     ["A", "A", "A", "A", "B", "B", "B", "B"],
        "true_violation": [1,    1,   0,   0,   1,   0,   0,   0],
        "pred":           [1,    1,   0,   0,   1,   1,   1,   0],
    })


def test_group_fairness_rates_match_hand_computation():
    fair = group_fairness(_frame(), "pred").set_index("department")
    assert fair.loc["A", "FPR_false_alarm"] == 0.0
    assert fair.loc["B", "FPR_false_alarm"] == 0.667
    assert fair.loc["A", "TPR_recall"] == 1.0 == fair.loc["B", "TPR_recall"]
    assert fair.loc["B", "flag_rate"] == 0.75


def test_group_fairness_nan_when_a_group_has_no_positives():
    df = pd.DataFrame({"department": ["C", "C"],
                       "true_violation": [0, 0], "pred": [0, 1]})
    fair = group_fairness(df, "pred")
    assert np.isnan(fair.loc[0, "TPR_recall"])           # no crash, honest NaN


def test_four_fifths_ratio_flags_the_unfair_frame():
    fair = group_fairness(_frame(), "pred")
    ratio = four_fifths_ratio(fair)
    # favourable(A) = 1-0.5 = 0.5 ; favourable(B) = 1-0.75 = 0.25 -> 0.5 exact
    assert ratio == pytest.approx(0.5)
    assert ratio < 0.80                                   # disparate impact


def test_four_fifths_ratio_is_1_when_groups_equal():
    df = pd.DataFrame({"department": ["A", "A", "B", "B"],
                       "true_violation": [1, 0, 1, 0], "pred": [1, 0, 1, 0]})
    assert four_fifths_ratio(group_fairness(df, "pred")) == 1.0


def test_four_fifths_ratio_undefined_when_everyone_flagged():
    df = pd.DataFrame({"department": ["A", "B"],
                       "true_violation": [1, 0], "pred": [1, 1]})
    with pytest.raises(ValueError):
        four_fifths_ratio(group_fairness(df, "pred"))
