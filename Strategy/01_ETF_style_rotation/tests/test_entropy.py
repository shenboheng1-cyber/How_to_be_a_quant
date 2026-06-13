import numpy as np
import pandas as pd
import pytest

from src.macro.entropy import (minmax_normalize, entropy_weights,
                               window_score, rolling_category_scores, drop_invalid)


def test_minmax_direction():
    idx = pd.date_range("2024-01-07", periods=4, freq="W")
    df = pd.DataFrame({"pos": [1.0, 2, 3, 4], "neg": [1.0, 2, 3, 4]}, index=idx)
    norm = minmax_normalize(df, {"pos": 1, "neg": -1})
    assert norm["pos"].iloc[0] == 0 and norm["pos"].iloc[-1] == 1
    assert norm["neg"].iloc[0] == 1 and norm["neg"].iloc[-1] == 0


def test_entropy_weights_sum_and_info():
    idx = pd.date_range("2024-01-07", periods=52, freq="W")
    rng = np.random.default_rng(0)
    # a 高变异(信息多), b 近似常数(信息少)
    df = pd.DataFrame({"a": rng.uniform(0, 1, 52),
                       "b": 0.5 + 1e-6 * rng.standard_normal(52)}, index=idx)
    norm = minmax_normalize(df, {"a": 1, "b": 1})
    w = entropy_weights(norm)
    assert abs(w.sum() - 1) < 1e-10
    assert w["a"] > w["b"]


def test_constant_column_degenerates_gracefully():
    idx = pd.date_range("2024-01-07", periods=10, freq="W")
    df = pd.DataFrame({"c": [2.0] * 10}, index=idx)
    score = window_score(df, {"c": 1})
    assert np.isfinite(score)


def test_drop_invalid():
    idx = pd.date_range("2024-01-07", periods=10, freq="W")
    df = pd.DataFrame({"good": range(10),
                       "bad": [np.nan] * 5 + list(range(5))}, index=idx)
    out = drop_invalid(df, threshold=0.2)
    assert "good" in out.columns and "bad" not in out.columns


def test_rolling_scores_in_range_and_no_lookahead():
    idx = pd.date_range("2023-01-01", periods=80, freq="W")
    rng = np.random.default_rng(1)
    weekly = pd.DataFrame({"x1": rng.uniform(0, 10, 80),
                           "x2": rng.uniform(0, 5, 80)}, index=idx)
    scores = rolling_category_scores(weekly, {"x1": 1, "x2": -1},
                                     {"cat": ["x1", "x2"]}, window=52)
    s = scores["cat"]
    assert s.iloc[:51].isna().all()          # 窗口不足处必须为 NaN (防未来)
    valid = s.dropna()
    assert ((valid >= 0) & (valid <= 1 + 1e-9)).all()
