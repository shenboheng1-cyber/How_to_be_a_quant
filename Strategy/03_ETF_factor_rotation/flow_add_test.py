"""把资金流因子(场内份额20日增长,反向)加进 V3 打分,验证是否真提升。
V3基线 vs V3+flow:全期 + OOS段 + walk-forward(真样本外,逐年验证 flow 是否稳定加分)。"""
from __future__ import annotations
import sqlite3
import numpy as np
import pandas as pd

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_monthly_weights_v2,
    backtest_monthly_strategy, FACTOR_WEIGHTS_V2)
import hfq_common as H

DB = H.DEFAULT_DATA_DIR / "etf_share_ifind.db"
COST, LAM = 5.0, 0.4


def share_growth_long(px, w=20):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql_query("SELECT fund_code,date,share FROM share", con, parse_dates=["date"]); con.close()
    df["fund_code"] = df["fund_code"].astype(str).str.zfill(6)
    share = df.pivot_table(index="date", columns="fund_code", values="share").sort_index()
    share = share.reindex(index=px.index).ffill(limit=5).reindex(columns=px.columns)
    sg = (share / share.shift(w) - 1.0).clip(-0.9, 5.0)          # winsorize 极端
    long = sg.stack(future_stack=True).rename("share_growth_20").reset_index()
    long.columns = ["date", "fund_code", "share_growth_20"]
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    return long


def strat(fac, weights):
    sc = score_factors_with_weights(fac, weights, score_column="s")
    w = make_monthly_weights_v2(sc, PX, UNI, buffer_rank=35, volatility_target=0.18, **H.PORT)
    eq, _ = backtest_monthly_strategy(PX, w, transaction_cost_bps=COST, rebalance_lambda=LAM)
    return H.to_ret(eq)


def m(r, lo=None, hi=None):
    r = r.loc[lo:hi] if lo else r
    x = H.metrics(r)
    return f"{x['ann']*100:>5.1f}%/S{x['sharpe']:.2f}/MDD{x['mdd']*100:.0f}%"


def main():
    global PX, UNI
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    PX, _, _ = H.load_hfq(); PX = PX.loc[:H.END]
    UNI = uni[uni["fund_code"].isin(PX.columns)].copy()
    fac = compute_factor_panel(PX); fac["date"] = fac["date"].astype(str)
    fac = fac.merge(share_growth_long(PX), on=["date", "fund_code"], how="left")

    W_BASE = dict(FACTOR_WEIGHTS_V2)
    configs = {"V3 基线": W_BASE}
    for fw in (-0.15, -0.25, -0.35):
        configs[f"V3+flow({fw})"] = {**W_BASE, "share_growth_20": fw}

    rets = {k: strat(fac, w) for k, w in configs.items()}
    print("=" * 78)
    print("V3 + 资金流因子(反向)  [后复权市价, 含5bps, lam0.4]")
    print("=" * 78)
    print(f"{'配置':16}{'全期2018-26':>18}{'OOS 2021-26':>18}{'OOS 2023-26':>18}")
    for k, r in rets.items():
        print(f"{k:16}{m(r):>18}{m(r,'2021-01-01',H.END):>18}{m(r,'2023-01-01',H.END):>18}")

    # walk-forward: 逐年判断"加 flow"是否稳定优于基线(真样本外, flow权重不在test上调)
    print("\n=== Walk-forward: 每年 基线 vs +flow(-0.25), 各年 OOS 收益 ===")
    rb, rf = rets["V3 基线"], rets["V3+flow(-0.25)"]
    rows = []
    for y in range(2019, 2027):
        b = H.metrics(rb.loc[f"{y}-01-01":f"{y}-12-31"]); f = H.metrics(rf.loc[f"{y}-01-01":f"{y}-12-31"])
        rows.append((y, b["ann"], f["ann"], f["ann"] - b["ann"]))
    wins = sum(1 for _, b, f, d in rows if d > 0)
    for y, b, f, d in rows:
        print(f"  {y}: 基线{b*100:+6.1f}%  +flow{f*100:+6.1f}%  差{d*100:+5.1f}%  {'✓' if d>0 else '✗'}")
    print(f"  +flow 胜 {wins}/{len(rows)} 年")


if __name__ == "__main__":
    main()
