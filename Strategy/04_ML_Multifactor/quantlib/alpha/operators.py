# -*- coding: utf-8 -*-
"""
quantlib.alpha.operators —— 因子工厂的算子库
================================================================
所有算子都作用在【宽矩阵】上：index = 交易日(升序)，columns = 股票代码。
- 横截面算子（cs_*）：在每个交易日(行)内对所有股票操作。
- 时序算子（ts_*）：沿时间(列方向，每只股票)滚动，只用过去 → 无前视。

这是 WorldQuant「101 Formulaic Alphas」式因子表达式的底层积木。
组合这些算子就能批量生成上百个因子。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------- 横截面算子（按行/按日）----------
def rank(df: pd.DataFrame) -> pd.DataFrame:
    """横截面百分位排名 ∈[0,1]（最常用，天然抗异常值）。"""
    return df.rank(axis=1, pct=True)


def cs_demean(df: pd.DataFrame) -> pd.DataFrame:
    """横截面去均值。"""
    return df.sub(df.mean(axis=1), axis=0)


def cs_scale(df: pd.DataFrame, k: float = 1.0) -> pd.DataFrame:
    """缩放使每行绝对值之和=k（组合权重常用）。"""
    return df.div(df.abs().sum(axis=1), axis=0) * k


# ---------- 时序算子（沿时间滚动；只用过去）----------
def delay(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """d 天前的值。"""
    return df.shift(d)


def delta(df: pd.DataFrame, d: int) -> pd.DataFrame:
    """当前值 − d 天前的值。"""
    return df - df.shift(d)


def ts_sum(df, w):  return df.rolling(w, min_periods=max(2, w // 2)).sum()
def ts_mean(df, w): return df.rolling(w, min_periods=max(2, w // 2)).mean()
def ts_std(df, w):  return df.rolling(w, min_periods=max(2, w // 2)).std()
def ts_min(df, w):  return df.rolling(w, min_periods=max(2, w // 2)).min()
def ts_max(df, w):  return df.rolling(w, min_periods=max(2, w // 2)).max()


def ts_rank(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """当前值在过去 w 天窗口内的百分位秩 ∈[0,1]（pandas 内置 C 实现）。"""
    return df.rolling(w, min_periods=max(2, w // 2)).rank(pct=True)


def ts_skew(df, w):  return df.rolling(w, min_periods=max(3, w // 2)).skew()
def ts_kurt(df, w):  return df.rolling(w, min_periods=max(4, w // 2)).kurt()


def _roll_view(arr2d: np.ndarray, w: int):
    """沿时间轴(axis=0)的滚动窗口视图，shape=(T-w+1, N, w)。零拷贝。"""
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(arr2d, w, axis=0)


def ts_argmax(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """过去 w 天最大值出现在几天前（0=今天）。向量化。"""
    a = df.values.astype(np.float64)
    win = _roll_view(np.where(np.isnan(a), -np.inf, a), w)
    pos = (w - 1 - win.argmax(axis=2)).astype(float)
    pos[~np.isfinite(win).any(axis=2)] = np.nan        # 全 NaN 窗口置空
    out = np.full(a.shape, np.nan); out[w - 1:] = pos
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def ts_argmin(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """过去 w 天最小值出现在几天前。向量化。"""
    a = df.values.astype(np.float64)
    win = _roll_view(np.where(np.isnan(a), np.inf, a), w)
    pos = (w - 1 - win.argmin(axis=2)).astype(float)
    pos[~np.isfinite(win).any(axis=2)] = np.nan
    out = np.full(a.shape, np.nan); out[w - 1:] = pos
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def ts_corr(a: pd.DataFrame, b: pd.DataFrame, w: int) -> pd.DataFrame:
    """a、b 在过去 w 天的滚动相关。用滚动和公式向量化计算，快且稳。"""
    mp = min(w, max(2, w // 2))
    ma, mb = a.rolling(w, min_periods=mp).mean(), b.rolling(w, min_periods=mp).mean()
    cov = (a * b).rolling(w, min_periods=mp).mean() - ma * mb
    sa = a.rolling(w, min_periods=mp).std(ddof=0)
    sb = b.rolling(w, min_periods=mp).std(ddof=0)
    return (cov / (sa * sb)).replace([np.inf, -np.inf], np.nan)


def decay_linear(df: pd.DataFrame, w: int) -> pd.DataFrame:
    """线性衰减加权移动平均（近的权重大）。向量化、NaN 自适应归一。"""
    a = df.values.astype(np.float32)
    win = _roll_view(a, w)                              # (T-w+1, N, w)
    weights = np.arange(1, w + 1, dtype=np.float32)
    mask = ~np.isnan(win)
    wv = np.where(mask, win, 0.0)
    wsum = (wv * weights).sum(axis=2)
    wnorm = (mask * weights).sum(axis=2)
    res = wsum / np.where(wnorm == 0, np.nan, wnorm)
    out = np.full(a.shape, np.nan, dtype=float); out[w - 1:] = res
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def signedpower(df: pd.DataFrame, p: float) -> pd.DataFrame:
    """保号幂：sign(x)·|x|^p。"""
    return np.sign(df) * df.abs() ** p
