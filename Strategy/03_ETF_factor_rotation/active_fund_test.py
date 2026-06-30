"""理想化测试:ETF-only vs ETF+主动基金(统一NAV口径,忽略赎回费,10bps)。
看加主动基金在【无摩擦】时能否提升——决定这个方向有没有上限可追。
注意:这是NAV口径(主动基金本就只有净值),且未计赎回费,是乐观上界。"""
from __future__ import annotations
import json, sqlite3
import numpy as np
import pandas as pd

from etf_factor_strategy.data import load_etf_universe, load_nav_prices, DEFAULT_DATA_DIR
from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_monthly_weights_v2,
    backtest_monthly_strategy, FACTOR_WEIGHTS_V2)
import hfq_common as H

START, END = "2018-01-02", "2026-06-05"


def active_codes():
    d = json.loads((DEFAULT_DATA_DIR / "bulk_universe.json").read_text(encoding="utf-8"))["data"]
    rows = []
    for c, it in d.items():
        n, t = str(it.get("name", "")), str(it.get("type", ""))
        if ("ETF" in n.upper()) or ("交易型开放式" in n) or ("联接" in n):
            continue
        if any(k in n + t for k in ["股票", "混合", "偏股", "灵活配置"]) and not any(k in n + t for k in ["债", "货币", "指数"]):
            rows.append({"fund_code": c, "fund_name": n, "fund_type": t or "混合型"})
    return pd.DataFrame(rows)


def metrics(eq):
    eq = eq.copy(); eq["date"] = pd.to_datetime(eq["date"])
    r = pd.Series(eq["strategy_return"].to_numpy(), index=eq["date"]).loc[START:END]
    return H.metrics(r), r


def run(prices, uni, label):
    prices = prices.dropna(axis=1, thresh=280)
    uni = uni[uni["fund_code"].isin(prices.columns)].copy()
    fac = compute_factor_panel(prices)
    sc = score_factors_with_weights(fac, FACTOR_WEIGHTS_V2, score_column="s")
    # 放宽同主题上限(主动基金多归入"其他/混合",否则被卡死)
    w = make_monthly_weights_v2(sc, prices, uni, top_n=20, max_per_theme=20, max_weight=0.12,
                                buffer_rank=35, volatility_target=0.18, cash_code="511880")
    eq, _ = backtest_monthly_strategy(prices, w, transaction_cost_bps=10.0, rebalance_lambda=0.4)
    m, r = metrics(eq)
    # 选中里主动基金占比
    held = w[w["fund_code"] != "511880"]["fund_code"].unique()
    act = set(active_codes()["fund_code"])
    n_act = sum(1 for c in held if c in act)
    print(f"{label:22} 年化{m['ann']*100:+.1f}% 波动{m['vol']*100:.1f}% Sharpe{m['sharpe']:.2f} "
          f"MDD{m['mdd']*100:.0f}% | 选中{len(held)}只,其中主动{n_act}只")
    return r


def main():
    etf = load_etf_universe(data_dir=DEFAULT_DATA_DIR)
    act = active_codes()
    print(f"ETF {len(etf)} + 主动 {len(act)}")

    print("\n拉 ETF NAV…")
    etf_px = load_nav_prices(etf["fund_code"].tolist(), start="2016-01-01", end=END)
    r_etf = run(etf_px, etf, "ETF-only(NAV)")

    print("拉 ETF+主动 NAV(较慢)…")
    allc = etf["fund_code"].tolist() + act["fund_code"].tolist()
    all_px = load_nav_prices(allc, start="2016-01-01", end=END)
    uni_all = pd.concat([etf, act], ignore_index=True).drop_duplicates("fund_code")
    r_all = run(all_px, uni_all, "ETF+主动(NAV,理想化)")

    rho = r_etf.corr(r_all.reindex(r_etf.index))
    print(f"\n两者相关 {rho:.2f}")
    print("注:NAV口径+未计赎回费,是乐观上界;月度持有主动基金真实赎回费~0.6-0.9%/往返会大幅拉低。")


if __name__ == "__main__":
    main()
