"""Per-group fairness audit (Week-2 Step 7, refactored and importable)."""
import numpy as np
import pandas as pd


def group_fairness(df: pd.DataFrame, pred_col: str,
                   truth_col: str = "true_violation",
                   group_col: str = "department") -> pd.DataFrame:
    """Confusion-matrix rates PER GROUP -- the whole method in one function."""
    rows = []
    for g, s in df.groupby(group_col):
        y, yh = s[truth_col], s[pred_col]
        tp = int(((y == 1) & (yh == 1)).sum())
        fp = int(((y == 0) & (yh == 1)).sum())
        tn = int(((y == 0) & (yh == 0)).sum())
        fn = int(((y == 1) & (yh == 0)).sum())
        rows.append({group_col: g, "n": len(s),
                     "true_violation_rate": round(float(y.mean()), 3),
                     "flag_rate": round(float(yh.mean()), 3),
                     "TPR_recall": round(tp / (tp + fn), 3) if tp + fn else np.nan,
                     "FPR_false_alarm": round(fp / (fp + tn), 3) if fp + tn else np.nan})
    return pd.DataFrame(rows)


def four_fifths_ratio(fair_df: pd.DataFrame, group_col: str = "department") -> float:
    """Disparate-impact test on the favourable outcome (= NOT being flagged).

    < 0.80 signals disparate impact (US EEOC rule of thumb).
    """
    fav = 1 - fair_df.set_index(group_col)["flag_rate"]
    if fav.max() == 0:
        raise ValueError("every group is always flagged -- ratio undefined")
    return float(fav.min() / fav.max())
