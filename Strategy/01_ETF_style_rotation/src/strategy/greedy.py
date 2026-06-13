"""贪心多样化指数选择 (报告 四.(二))。

(1) Norm_score = minmax(Style_score)
(2) 先选 Norm 最高者
(3) 其余: Combined_i = (1-w_d)*Norm_i + w_d*D_i, D_i = 指数i到所有已选指数
    在10维风格暴露空间的平均欧氏距离
(4) 循环到选满 Z 个

实现备注: 报告未说明 D_i 是否归一化。原始欧氏距离与 [0,1] 的 Norm_score
量纲不可比, 这里默认每轮把 D 也 min-max 到 [0,1] (normalize_distance=True),
对账时如有偏差可切换为 False。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if not np.isfinite(rng) or rng == 0:
        return pd.Series(1.0, index=s.index)
    return (s - s.min()) / rng


def avg_euclidean_distance(exposures: pd.DataFrame,
                           candidates: pd.Index,
                           selected: list) -> pd.Series:
    sel = exposures.loc[selected].values            # (m, 10)
    out = {}
    for i in candidates:
        v = exposures.loc[i].values
        out[i] = float(np.linalg.norm(sel - v, axis=1).mean())
    return pd.Series(out)


def greedy_select(style_score: pd.Series,
                  exposures: pd.DataFrame,
                  z: int = 8,
                  w_d: float = 0.5,
                  normalize_distance: bool = True):
    """返回 (selected: list[index_code], norm_score: pd.Series)。"""
    s = style_score.dropna()
    s = s[s.index.isin(exposures.index)]
    norm = minmax(s)
    if len(norm) == 0:
        return [], norm
    selected = [norm.idxmax()]
    while len(selected) < min(z, len(norm)):
        remaining = norm.index.difference(selected)
        D = avg_euclidean_distance(exposures, remaining, selected)
        if normalize_distance:
            D = minmax(D)
        combined = (1 - w_d) * norm.loc[remaining] + w_d * D
        selected.append(combined.idxmax())
    return selected, norm
