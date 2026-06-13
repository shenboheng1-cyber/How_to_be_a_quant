"""Phase2: 完整 Barra 截面 WLS 回归求因子收益。

r = country + X_ind·f_ind + X_sty·f_sty + eps
  - WLS 权重 = sqrt(流通市值)
  - 行业因子带市值加权约束: Σ_ind s_ind · f_ind = 0 (s_ind = 行业市值占比),
    消除与国家因子的共线性
每周做一次截面回归, f_sty 即 10 个风格因子的周度因子收益。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def weekly_cross_section_regression(returns: pd.Series,
                                    style_exposures: pd.DataFrame,
                                    industry: pd.Series,
                                    float_mv: pd.Series) -> pd.Series:
    """单周截面回归。返回 Series: ['country', 行业..., 风格...] 的因子收益。

    带约束的实现思路 (Menchero & Lee 风格):
      构造约束矩阵 C 将行业最后一列表示为其余行业的线性组合
      f_ind_K = -Σ_{j<K} (s_j/s_K) f_ind_j, 再做无约束 WLS。
    """
    df = pd.concat([returns.rename("r"), style_exposures,
                    industry.rename("ind"), float_mv.rename("mv")], axis=1).dropna()
    if len(df) < 200:
        return pd.Series(dtype=float)

    styles = list(style_exposures.columns)
    inds = sorted(df["ind"].unique())
    K = len(inds)
    D = pd.get_dummies(df["ind"])[inds].values.astype(float)     # (n, K)
    s = df.groupby("ind")["mv"].sum().reindex(inds).values
    s = s / s.sum()

    # 约束: f_K = -Σ_{j<K} s_j/s_K f_j  ->  行业设计矩阵降一维
    R = np.eye(K)[:, :-1]
    R[-1, :] = -s[:-1] / s[-1]
    D_c = D @ R                                                   # (n, K-1)

    X = np.column_stack([np.ones(len(df)), D_c, df[styles].values])
    y = df["r"].values
    w = np.sqrt(df["mv"].values)
    W = w / w.sum()

    Xw = X * np.sqrt(W)[:, None]
    yw = y * np.sqrt(W)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    f_country = beta[0]
    f_ind = R @ beta[1:K]                                         # 还原 K 个行业收益
    f_sty = beta[K:]
    out = {"country": f_country}
    out.update({f"ind_{i}": v for i, v in zip(inds, f_ind)})
    out.update({sname: v for sname, v in zip(styles, f_sty)})
    return pd.Series(out)


def build_factor_returns(weekly_returns: pd.DataFrame,
                         exposures_by_week: dict,
                         industry_by_week: dict,
                         float_mv_by_week: dict) -> pd.DataFrame:
    """逐周回归, 返回 index=周, columns=10风格因子 的因子收益。
    注意防未来: t+1 周收益对 t 周末的暴露回归, 因子收益记在 t+1。"""
    weeks = sorted(exposures_by_week.keys())
    ret_idx = list(weekly_returns.index)
    rows = {}
    for t in weeks:
        later = [d for d in ret_idx if d > t]
        if not later:
            continue
        t1 = later[0]
        res = weekly_cross_section_regression(
            weekly_returns.loc[t1], exposures_by_week[t],
            industry_by_week[t], float_mv_by_week[t])
        if res.empty:
            continue
        styles = [c for c in exposures_by_week[t].columns]
        rows[t1] = res.reindex(styles)
    return pd.DataFrame(rows).T.sort_index()
