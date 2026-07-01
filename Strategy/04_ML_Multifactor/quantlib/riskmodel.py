# -*- coding: utf-8 -*-
"""
quantlib.riskmodel —— Barra 式多因子风险模型（L5）
================================================================
每期横截面回归 个股收益 ~ 行业哑变量 + 风格因子，得到：
  - 因子收益 f_t（时序）→ 因子协方差 F（EWMA）
  - 特质收益 ε → 个股特质风险 d（EWMA of ε²）
组合风险 σ²(w) = (Xᵀw)ᵀ F (Xᵀw) + Σ d_i w_i²（因子结构，避免 N×N 大矩阵）。

PIT：优化某调仓日 t 时，F/d 只用 t 之前已实现的 f/ε。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def factor_returns(panel: pd.DataFrame, style_cols: list,
                   industry_col: str = "industry", fwd: str = "fwd_ret"):
    """逐调仓日 WLS(按√市值加权) 回归 fwd ~ 行业+风格。
    返回 (f_df: 日期×因子 因子收益, resid: 与panel对齐的特质收益, ind_levels)。"""
    ind_levels = sorted(panel[industry_col].dropna().unique())
    style = [c for c in style_cols]
    f_rows = {}
    resid = pd.Series(np.nan, index=panel.index)
    for dt, g in panel.groupby("trddt"):
        y = g[fwd].values
        m = ~np.isnan(y) & g[industry_col].notna().values
        if m.sum() < 50:
            continue
        gg = g[m]
        ind = pd.get_dummies(gg[industry_col]).reindex(columns=ind_levels, fill_value=0)
        Xs = gg[style].fillna(0.0)
        X = np.column_stack([ind.values.astype(float), Xs.values])
        cols = list(ind_levels) + style
        sw = np.sqrt(np.nan_to_num(gg["total_mktcap"].values, nan=0.0))
        sw = sw / (sw.sum() + 1e-12)
        sw = np.sqrt(sw)[:, None]
        beta, *_ = np.linalg.lstsq(X * sw, y[m] * sw[:, 0], rcond=None)
        f_rows[dt] = pd.Series(beta, index=cols)
        resid.loc[gg.index] = y[m] - X @ beta
    return pd.DataFrame(f_rows).T, resid, ind_levels


def factor_cov(f_df: pd.DataFrame, halflife: int = 24, ann: int = 12) -> pd.DataFrame:
    """因子协方差(EWMA, 年化)。"""
    w = 0.5 ** (np.arange(len(f_df))[::-1] / halflife)
    w = w / w.sum()
    fc = f_df.fillna(0.0)
    mu = np.average(fc.values, axis=0, weights=w)
    d = fc.values - mu
    cov = (d * w[:, None]).T @ d
    return pd.DataFrame(cov * ann, index=f_df.columns, columns=f_df.columns)


def specific_var(panel, resid: pd.Series, halflife: int = 12, ann: int = 12) -> pd.DataFrame:
    """个股特质方差(按时间 EWMA of ε², 年化)。返回 index=(stkcd) 截至每个调仓日的值用宽表。
    简化：返回 长表 (stkcd, trddt, specvar)，optimizer 取 t 时的最新值。"""
    df = pd.DataFrame({"stkcd": panel["stkcd"].values, "trddt": panel["trddt"].values,
                       "e2": (resid.values ** 2)})
    df = df.dropna().sort_values(["stkcd", "trddt"])
    df["specvar"] = (df.groupby("stkcd")["e2"]
                     .transform(lambda s: s.ewm(halflife=halflife, min_periods=3).mean())) * ann
    return df[["stkcd", "trddt", "specvar"]]
