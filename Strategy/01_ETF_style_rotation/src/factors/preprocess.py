"""因子预处理: 去极值(3MAD) + 标准化 + 行业及市值中性化。截面操作, 防未来。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_mad(s: pd.Series, n_mad: float = 3.0) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or not np.isfinite(mad):
        return s
    lo, hi = med - n_mad * 1.4826 * mad, med + n_mad * 1.4826 * mad
    return s.clip(lo, hi)


def zscore(s: pd.Series, weights: pd.Series | None = None) -> pd.Series:
    """标准化。Barra口径: 均值用市值加权, 标准差用等权; weights=None 则全等权。"""
    if weights is not None:
        w = weights.reindex(s.index).fillna(0.0)
        mu = float((s * w).sum() / w.sum()) if w.sum() > 0 else s.mean()
    else:
        mu = s.mean()
    sd = s.std(ddof=1)
    return (s - mu) / sd if sd > 0 else s * 0.0


def neutralize(s: pd.Series, industry: pd.Series,
               log_mv: pd.Series | None = None) -> pd.Series:
    """对行业哑变量(+对数市值)回归取残差。"""
    df = pd.DataFrame({"y": s, "ind": industry})
    if log_mv is not None:
        df["mv"] = log_mv
    df = df.dropna()
    if df.empty:
        return s * np.nan
    dummies = pd.get_dummies(df["ind"], drop_first=False).astype(float)
    X = dummies.values
    if log_mv is not None:
        X = np.column_stack([X, df["mv"].values])
    y = df["y"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return pd.Series(resid, index=df.index).reindex(s.index)


def standardize_descriptor(raw: pd.Series, industry: pd.Series,
                           log_mv: pd.Series, mv_weight: pd.Series,
                           n_mad: float = 3.0,
                           do_neutralize: bool = True) -> pd.Series:
    """单个描述变量的完整截面处理流水线。"""
    x = winsorize_mad(raw, n_mad)
    x = zscore(x, mv_weight)
    if do_neutralize:
        x = neutralize(x, industry, log_mv)
        x = zscore(x)            # 中性化后再标准化一次
    return x
