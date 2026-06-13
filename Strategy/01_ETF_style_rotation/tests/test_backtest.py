import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.metrics import max_drawdown, summarize


def make_market():
    days = pd.date_range("2024-01-01", periods=20, freq="B")
    prices = pd.DataFrame({"E1": 1.0, "E2": 2.0}, index=days)  # 价格恒定
    return days, prices


def test_cost_accounting_flat_prices():
    days, prices = make_market()
    # 信号日 = 第1天, T+1 = 第2天执行, 全仓买入 E1
    tw = {days[0]: pd.Series({"E1": 1.0})}
    res = run_backtest(prices, tw, days, cost_rate=0.0003, initial_capital=1e8)
    nav = res["nav"]
    # 执行前净值=1e8; 买入1e8, 成本=1e8*0.0003=3万 -> 终值=1e8-3e4
    assert abs(nav.iloc[-1] - (1e8 - 1e8 * 0.0003)) < 1.0
    # 单边换手 = 0.5 * 1e8/1e8 = 0.5
    assert abs(res["turnover"].iloc[0] - 0.5) < 1e-9


def test_rebalance_switch_costs_both_sides():
    days, prices = make_market()
    tw = {days[0]: pd.Series({"E1": 1.0}),
          days[5]: pd.Series({"E2": 1.0})}
    res = run_backtest(prices, tw, days, cost_rate=0.0003, initial_capital=1e8)
    v1 = 1e8 * (1 - 0.0003)               # 第一次建仓后净值 (现金-3万, E1持仓1e8)
    v2 = v1 - (1e8 + v1) * 0.0003         # 卖出E1全部1e8 + 买入E2目标v1
    assert abs(res["nav"].iloc[-1] - v2) < 1.0


def test_value_tracks_price():
    days = pd.date_range("2024-01-01", periods=10, freq="B")
    px = pd.Series(np.linspace(1.0, 1.09, 10), index=days)
    prices = pd.DataFrame({"E1": px})
    tw = {days[0]: pd.Series({"E1": 1.0})}
    res = run_backtest(prices, tw, days, cost_rate=0.0, initial_capital=1e8)
    exec_px = px.iloc[1]
    expected = 1e8 / exec_px * px.iloc[-1]
    assert abs(res["nav"].iloc[-1] - expected) < 1.0


def test_metrics_basics():
    nav = pd.Series([1, 1.1, 1.05, 1.2],
                    index=pd.date_range("2024-01-01", periods=4, freq="B"))
    assert abs(max_drawdown(nav) - (1.05 / 1.1 - 1)) < 1e-12
    out = summarize(nav)
    assert out["cum_return"] > 0 and np.isfinite(out["sharpe"])
