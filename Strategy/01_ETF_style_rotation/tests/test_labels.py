import numpy as np
import pandas as pd

from src.prediction.labels import cumulative_label, single_week_label, trend_label


def make_fr():
    idx = pd.date_range("2024-01-05", periods=6, freq="W-FRI")
    return pd.DataFrame({
        "f1": [0.01, 0.02, 0.01, 0.03, -0.01, 0.02],
        "f2": [-0.01, -0.02, 0.0, -0.01, 0.01, -0.02],
        "f3": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }, index=idx)


def test_single_week():
    fr = make_fr()
    lab = single_week_label(fr)
    assert lab["f1"].iloc[0] == 1 and lab["f2"].iloc[0] == 0 and lab["f3"].iloc[0] == 0


def test_cumulative_window_nan_then_correct():
    fr = make_fr()
    lab = cumulative_label(fr, window=4)
    assert lab.iloc[:3].isna().all().all()
    # 第4行: 4周累计 f1=0.07, f2=-0.04, f3=0; 中位数=0 -> f1=1, f2=0, f3=0(不大于)
    assert lab["f1"].iloc[3] == 1
    assert lab["f2"].iloc[3] == 0
    assert lab["f3"].iloc[3] == 0


def test_trend():
    fr = make_fr()
    lab = trend_label(fr, window=4)
    # 第5行 f1: 当周-0.01, 4周均值=(0.02+0.01+0.03-0.01)/4=0.0125 -> 0
    assert lab["f1"].iloc[4] == 0
    # 第4行 f1: 当周0.03 > 均值0.0175 -> 1
    assert lab["f1"].iloc[3] == 1
