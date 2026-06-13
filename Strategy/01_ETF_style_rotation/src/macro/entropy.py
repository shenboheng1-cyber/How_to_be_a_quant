"""熵权法宏观综合得分 (报告 2.(二), 公式与步骤逐条对应)。

步骤:
(1) 日频->周频算术平均, 丢弃周度无效值>20%的指标
(2) 每个时点取过去52周窗口
(3) 按方向 min-max 归一
(4) P_tj = x'_tj / (Σ_t x'_tj + ε)
(5) E_j = -k Σ P ln(P+ε), k=1/lnT;  d_j = 1-E_j
(6) w_j = d_j/Σd_j;  score_k = Σ_j x'_Tj · w_j   (取窗口最后一行 x' 加权)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def to_weekly(daily: pd.Series, week_anchor: str = "W-SUN") -> pd.Series:
    """日频 -> 周频算术平均。索引为周期末自然日。"""
    return daily.resample(week_anchor).mean()


def drop_invalid(weekly: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
    """丢弃周度无效值占比 > threshold 的指标列。"""
    valid = [c for c in weekly.columns if weekly[c].isna().mean() <= threshold]
    return weekly[valid]


def minmax_normalize(window: pd.DataFrame, directions: dict[str, int]) -> pd.DataFrame:
    """对窗口内每个指标按方向做 min-max 归一。常数列归一为 0。"""
    out = {}
    for col in window.columns:
        x = window[col].astype(float)
        lo, hi = x.min(), x.max()
        rng = hi - lo
        if not np.isfinite(rng) or rng == 0:
            out[col] = pd.Series(0.0, index=window.index)
        elif directions[col] > 0:
            out[col] = (x - lo) / rng
        else:
            out[col] = (hi - x) / rng
    return pd.DataFrame(out, index=window.index)


def entropy_weights(norm: pd.DataFrame, eps: float = EPS) -> pd.Series:
    """对归一化窗口矩阵计算熵权 w_j (和为1)。信息量(变异)越大权重越大。"""
    T = len(norm)
    if T < 2:
        raise ValueError("窗口长度必须 >= 2")
    P = norm / (norm.sum(axis=0) + eps)
    k = 1.0 / np.log(T)
    E = -k * (P * np.log(P + eps)).sum(axis=0)
    d = 1.0 - E
    total = d.sum()
    if total <= 0:
        # 所有列均无信息(全常数), 退化为等权
        return pd.Series(1.0 / len(d), index=d.index)
    return d / total


def window_score(window: pd.DataFrame, directions: dict[str, int],
                 eps: float = EPS) -> float:
    """单个窗口的综合得分: 用窗口最后一行的归一值按熵权加权。"""
    norm = minmax_normalize(window, directions)
    w = entropy_weights(norm, eps)
    return float((norm.iloc[-1] * w).sum())


def rolling_category_scores(weekly: pd.DataFrame,
                            directions: dict[str, int],
                            categories: dict[str, list[str]],
                            window: int = 52,
                            invalid_ratio: float = 0.2) -> pd.DataFrame:
    """对每个类别滚动计算综合得分。

    weekly: index=周末日期, columns=全部指标
    directions: {指标: ±1}
    categories: {类别: [指标列表]}
    返回: index=周末日期, columns=类别, 前 window-1 周为 NaN
    """
    weekly = drop_invalid(weekly, invalid_ratio)
    result = {}
    for cat, cols in categories.items():
        cols = [c for c in cols if c in weekly.columns]
        if not cols:
            continue
        sub = weekly[cols].ffill()
        s = pd.Series(np.nan, index=weekly.index)
        for i in range(window, len(sub) + 1):
            win = sub.iloc[i - window:i].dropna(how="any")
            if len(win) < max(8, window // 4):       # 窗口有效行过少则跳过
                continue
            s.iloc[i - 1] = window_score(win, directions)
        result[cat] = s
    return pd.DataFrame(result)
