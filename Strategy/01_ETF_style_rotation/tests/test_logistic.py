import numpy as np
import pandas as pd

from src.prediction.logistic import build_feature_matrix, rolling_logistic_predict


def test_rolling_logistic_learns_signal():
    n = 200
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    rng = np.random.default_rng(0)
    macro = pd.DataFrame({f"m{i}": rng.standard_normal(n) for i in range(5)}, index=idx)
    fr = pd.Series(rng.standard_normal(n) * 0.01, index=idx)
    # 构造可学习信号: y_{t+1} 由 m0_t 决定
    label = (macro["m0"].shift(1) > 0).astype(float)
    X = build_feature_matrix(macro, fr, ar_lags=4)
    res = rolling_logistic_predict(label, X, window=104)
    acc = (res.dropna(subset=["actual"]).eval("pred == actual")).mean()
    assert acc > 0.9   # 强信号应几乎全对


def test_no_lookahead_index():
    n = 130
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    rng = np.random.default_rng(1)
    macro = pd.DataFrame({f"m{i}": rng.standard_normal(n) for i in range(5)}, index=idx)
    fr = pd.Series(rng.standard_normal(n), index=idx)
    label = (fr > 0).astype(float)
    X = build_feature_matrix(macro, fr, ar_lags=4)
    res = rolling_logistic_predict(label, X, window=104)
    # 第一条预测必须落在 第106周或之后 (104窗口 + 预测t+1)
    assert res.index.min() >= idx[105]
