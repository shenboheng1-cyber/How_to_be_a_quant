"""周度调仓回测引擎 (报告 四.(三)): 周末发信号, T+1 按 ETF 价格/净值执行,
单边成本 cost_rate, 初始资金 initial_capital。

会计规则:
- 持仓以"份额"记账, 逐日按价格估值
- 执行日: 目标权重 × 执行前组合净值 -> 目标市值; 成交额 = |目标市值-现持市值|;
  成本 = 成交额 × cost_rate, 从现金扣除
- 换手率(单边) = Σ卖出(或买入)市值 / 组合净值; 报告口径以年度加总展示
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(prices: pd.DataFrame,
                 target_weights: dict,
                 trading_days: pd.DatetimeIndex,
                 cost_rate: float = 0.0003,
                 initial_capital: float = 1e8,
                 execution_lag: int = 1) -> dict:
    """
    prices: index=交易日, columns=ETF代码, 值=执行/估值价格(复权净值)
    target_weights: {信号日(周末调仓日): pd.Series(etf->weight, 和<=1)}
    返回 {nav: 日度净值Series, turnover: 执行日单边换手Series, trades: 明细DataFrame}
    """
    days = trading_days[(trading_days >= prices.index.min()) &
                        (trading_days <= prices.index.max())]
    prices = prices.reindex(days).ffill()

    # 信号日 -> 执行日
    exec_map = {}
    for sig_date, w in target_weights.items():
        later = days[days > pd.Timestamp(sig_date)]
        if len(later) >= execution_lag:
            exec_map[later[execution_lag - 1]] = w.dropna()

    cash = initial_capital
    units = pd.Series(dtype=float)            # ETF份额
    nav_list, turn_list, trade_rows = [], [], []

    for d in days:
        px = prices.loc[d]
        port_val = cash + float((units * px.reindex(units.index)).sum()) if len(units) else cash

        if d in exec_map:
            w = exec_map[d]
            w = w[w > 0]
            cur_val = (units * px.reindex(units.index)).fillna(0.0) if len(units) else pd.Series(dtype=float)
            target_val = w * port_val
            all_codes = target_val.index.union(cur_val.index)
            cur_v = cur_val.reindex(all_codes).fillna(0.0)
            tgt_v = target_val.reindex(all_codes).fillna(0.0)
            delta = tgt_v - cur_v
            traded = delta.abs().sum()
            cost = traded * cost_rate
            sell_amt = (-delta[delta < 0]).sum()
            buy_amt = delta[delta > 0].sum()

            # 现金流: 卖出回流, 买入支出, 成本扣减
            cash = cash + sell_amt - buy_amt - cost
            new_units = {}
            for c in all_codes:
                p = px.get(c, np.nan)
                if tgt_v[c] > 0 and np.isfinite(p) and p > 0:
                    new_units[c] = tgt_v[c] / p
            units = pd.Series(new_units, dtype=float)
            turn_list.append((d, 0.5 * traded / port_val if port_val > 0 else 0.0))
            trade_rows.append({"date": d, "buy": buy_amt, "sell": sell_amt,
                               "cost": cost, "port_value_pre": port_val})
            port_val = cash + float((units * px.reindex(units.index)).sum())

        nav_list.append((d, port_val))

    nav = pd.Series(dict(nav_list)).sort_index()
    turnover = pd.Series(dict(turn_list)).sort_index() if turn_list else pd.Series(dtype=float)
    trades = pd.DataFrame(trade_rows)
    return {"nav": nav, "turnover": turnover, "trades": trades,
            "nav_norm": nav / initial_capital}
