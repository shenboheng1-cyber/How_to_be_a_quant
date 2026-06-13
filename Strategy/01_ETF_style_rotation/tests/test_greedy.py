import numpy as np
import pandas as pd

from src.strategy.greedy import greedy_select, minmax


def test_first_pick_is_max_score():
    score = pd.Series({"a": 3.0, "b": 2.0, "c": 1.0})
    expo = pd.DataFrame(np.eye(3), index=["a", "b", "c"])
    sel, norm = greedy_select(score, expo, z=2, w_d=0.5)
    assert sel[0] == "a"
    assert abs(norm["a"] - 1) < 1e-12 and abs(norm["c"]) < 1e-12


def test_diversity_changes_second_pick():
    # b 得分略高但风格与 a 几乎相同; c 得分略低但风格差异大
    score = pd.Series({"a": 1.0, "b": 0.9, "c": 0.7})
    expo = pd.DataFrame({"x": [1.0, 0.99, -1.0], "y": [0.0, 0.0, 0.0]},
                        index=["a", "b", "c"])
    sel_div, _ = greedy_select(score, expo, z=2, w_d=0.5)
    assert sel_div == ["a", "c"]
    sel_no, _ = greedy_select(score, expo, z=2, w_d=0.0)
    assert sel_no == ["a", "b"]


def test_select_count_capped():
    score = pd.Series({"a": 1.0, "b": 0.5})
    expo = pd.DataFrame(np.eye(2), index=["a", "b"])
    sel, _ = greedy_select(score, expo, z=8)
    assert len(sel) == 2
