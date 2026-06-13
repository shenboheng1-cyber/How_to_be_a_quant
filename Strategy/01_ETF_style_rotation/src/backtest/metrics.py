"""绩效指标: 年化收益/波动/Sharpe/Calmar/最大回撤/换手, 与逐年表 (对账 表7)。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(nav: pd.Series) -> float:
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def summarize(nav: pd.Series, turnover: pd.Series | None = None,
              rf: float = 0.0) -> dict:
    ret = nav.pct_change().dropna()
    n = len(ret)
    if n == 0:
        return {}
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (TRADING_DAYS / n) - 1.0
    ann_vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    mdd = max_drawdown(nav)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    out = {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
           "calmar": calmar, "max_drawdown": mdd,
           "cum_return": nav.iloc[-1] / nav.iloc[0] - 1.0}
    if turnover is not None and len(turnover):
        out["turnover"] = float(turnover.sum())
    return out


def yearly_table(nav: pd.Series, turnover: pd.Series | None = None) -> pd.DataFrame:
    rows = {"全区间": summarize(nav, turnover)}
    for year, sub in nav.groupby(nav.index.year):
        to = turnover[turnover.index.year == year] if turnover is not None else None
        rows[str(year)] = summarize(sub, to)
    return pd.DataFrame(rows).T
