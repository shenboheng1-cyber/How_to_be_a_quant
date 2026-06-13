"""Phase1: 分位多空法近似因子周度收益。

对每个合成风格因子, 在每个调仓周 t 用 t 时点(仅历史信息)的因子暴露排序,
Top q 等权 - Bottom q 等权, 持有至 t+1, 收益记为该因子 t+1 周的近似收益。
用于快速跑通全管线; Phase2 替换为完整 Barra 截面回归后对账。
"""
from __future__ import annotations

import pandas as pd


def quantile_long_short_return(exposure: pd.Series,
                               next_week_return: pd.Series,
                               q: float = 0.2) -> float:
    df = pd.DataFrame({"x": exposure, "r": next_week_return}).dropna()
    if len(df) < 50:
        return float("nan")
    n = max(int(len(df) * q), 1)
    df = df.sort_values("x")
    return float(df["r"].tail(n).mean() - df["r"].head(n).mean())


def build_factor_returns(exposures_by_week: dict,
                         weekly_stock_returns: pd.DataFrame,
                         q: float = 0.2) -> pd.DataFrame:
    """exposures_by_week: {周末日期: DataFrame(index=股票, columns=10因子)}
    weekly_stock_returns: index=周末日期, columns=股票 (t行 = t-1周末 到 t周末 的收益)
    返回: index=周末日期(t+1), columns=10因子。"""
    weeks = sorted(exposures_by_week.keys())
    ret_index = list(weekly_stock_returns.index)
    rows = {}
    for t in weeks:
        later = [d for d in ret_index if d > t]
        if not later:
            continue
        t1 = later[0]
        r_next = weekly_stock_returns.loc[t1]
        expo = exposures_by_week[t]
        rows[t1] = {fac: quantile_long_short_return(expo[fac], r_next, q)
                    for fac in expo.columns}
    return pd.DataFrame(rows).T.sort_index()
