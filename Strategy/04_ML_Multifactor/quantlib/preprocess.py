# -*- coding: utf-8 -*-
"""
quantlib.preprocess —— 因子预处理（去极值 / 中性化 / 标准化）
================================================================
全部是【横截面】操作：在每个调仓日 trddt 内部对所有股票做，期与期之间独立。
这点至关重要——绝不能跨时间标准化，否则会把"未来"的分布信息泄漏到过去。

标准三步管线 preprocess_factor()：
  ① winsorize_mad  去极值：中位数 ± k·MAD，比均值±3σ 更抗少数极端值
  ② neutralize     中性化：因子对 log(市值)[+行业] 回归，取残差
                   —— 目的：剥离规模暴露，回答"控制住市值后，这因子还有没有用"
  ③ standardize    标准化：zscore，使各因子量纲一致、可加权合成

行业中性化：本库暂无行业分类表，industry_col 预留接口，拿到数据后传入即可。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------- ① 去极值 ----------
def winsorize_mad(s: pd.Series, k: float = 3.0) -> pd.Series:
    """单个横截面的 MAD 去极值。mad 退化为 0 时回退到 1/99 分位裁剪。"""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        lo, hi = s.quantile(0.01), s.quantile(0.99)
    else:
        scale = 1.4826 * mad        # 1.4826 使 MAD 在正态下≈标准差
        lo, hi = med - k * scale, med + k * scale
    return s.clip(lo, hi)


# ---------- ③ 标准化 ----------
def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


# ---------- ② 中性化（单横截面的 OLS 取残差）----------
def _neutralize_xs(y: pd.Series, logsize: pd.Series,
                   industry: pd.Series | None = None) -> pd.Series:
    """对单个调仓日：y ~ const + logsize [+ 行业哑变量]，返回残差（NaN 原样保留）。"""
    valid = y.notna() & logsize.notna()
    if industry is not None:
        valid &= industry.notna()
    res = pd.Series(np.nan, index=y.index)
    if valid.sum() < 10:            # 样本太少不回归
        return res

    # 设计矩阵 X = [常数, log市值, (行业哑变量...)]
    n = int(valid.sum())
    parts = [np.ones((n, 1)), logsize[valid].values.reshape(-1, 1)]
    if industry is not None:
        dummies = pd.get_dummies(industry[valid], drop_first=True).values.astype(float)
        if dummies.size:
            parts.append(dummies)
    X = np.hstack(parts)

    yv = y[valid].values
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res.loc[valid] = yv - X @ beta   # 残差 = 因子中"市值/行业解释不了"的部分
    return res


# ---------- 完整管线 ----------
def preprocess_factor(panel: pd.DataFrame, factor: pd.Series,
                      size_col: str = "total_mktcap",
                      industry_col: str | None = None,
                      k: float = 3.0, do_neutralize: bool = True) -> pd.Series:
    """对一个原始因子做 去极值→中性化→标准化，按 trddt 分横截面。

    panel : 含 'trddt' 与 size_col（及可选 industry_col）的研究面板
    factor: 与 panel 行对齐的原始因子值（pd.Series）
    返回  : 预处理后的因子（与 panel 行对齐，可直接进 evaluate / 合成）
    """
    df = pd.DataFrame({"trddt": panel["trddt"].values, "f": factor.values})
    df["logsize"] = np.log(panel[size_col].values)
    if industry_col is not None:
        df["ind"] = panel[industry_col].values

    # ① 去极值（按日）
    df["f"] = df.groupby("trddt")["f"].transform(lambda s: winsorize_mad(s, k))

    # ② 中性化（按日回归取残差）。逐组算残差再按原索引拼回，避免 groupby.apply
    #    在单组/多组时返回形状不一致的坑。
    if do_neutralize:
        parts = []
        for _, g in df.groupby("trddt"):
            ind = g["ind"] if industry_col is not None else None
            parts.append(_neutralize_xs(g["f"], g["logsize"], ind))
        df["f"] = pd.concat(parts).reindex(df.index)

    # ③ 标准化（按日 zscore）
    df["f"] = df.groupby("trddt")["f"].transform(zscore)

    out = pd.Series(df["f"].values, index=panel.index, name=getattr(factor, "name", "factor"))
    return out


def orthogonalize(panel: pd.DataFrame, target: pd.Series,
                  others: dict) -> pd.Series:
    """逐横截面回归 target ~ const + others，返回【残差】。

    用途：把 target 里"能被 others 解释的部分"剥掉，残差=target 的独立增量信息。
    若残差仍有显著 IC，说明 target 不是 others 的伪装。

    target : 与 panel 对齐的因子（建议已 zscore）
    others : {名称: Series}，作为控制变量的因子（建议已 zscore）
    返回   : 残差 Series（与 panel 对齐）
    """
    cols = {"f": np.asarray(target)}
    for name, s in others.items():
        cols[name] = np.asarray(s)
    df = pd.DataFrame(cols)
    df["trddt"] = panel["trddt"].values
    onames = list(others.keys())

    parts = []
    for _, g in df.groupby("trddt"):
        y = g["f"]
        X = g[onames]
        valid = y.notna() & X.notna().all(axis=1)
        res = pd.Series(np.nan, index=g.index)
        if valid.sum() >= 10:
            Xm = np.column_stack([np.ones(valid.sum()), X[valid].values])
            yv = y[valid].values
            beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
            res.loc[valid] = yv - Xm @ beta
        parts.append(res)
    out = pd.concat(parts).reindex(df.index)
    return pd.Series(out.values, index=panel.index, name=f"{getattr(target,'name','f')}_orth")
