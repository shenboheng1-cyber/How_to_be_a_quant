"""增量因子组合回测: 在调松风险层基础上逐个加因子, 看 2023-24 前期收益 + 全期 Sharpe。
目标: 修复 2023-24 趴平(诊断: 主因是风险层把仓位躲进现金; 次为缺震荡市因子)。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from etf_factor_strategy.data import load_etf_universe, load_nav_prices, DEFAULT_DATA_DIR
from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_robust_monthly_weights,
    backtest_monthly_strategy,
)
from factor_research import factor_panels

TD = 252


def hs():
    con = sqlite3.connect(f"file:{DEFAULT_DATA_DIR}/idx_store.db?mode=ro", uri=True)
    d = pd.read_sql_query("SELECT date,close FROM index_daily WHERE symbol='CSI300' ORDER BY date", con)
    con.close(); d["date"] = pd.to_datetime(d["date"]); return d.set_index("date")["close"].astype(float)


def add_factor_cols(fac_long, panels, names):
    for nm in names:
        p = panels[nm].stack().reset_index()
        p.columns = ["date", "fund_code", nm]
        p["date"] = pd.to_datetime(p["date"]).dt.strftime("%Y-%m-%d")
        fac_long = fac_long.merge(p, on=["date", "fund_code"], how="left")
    return fac_long


def run(fac_long, weights, port, label, market):
    scored = score_factors_with_weights(fac_long, weights, score_column="s")
    w = make_robust_monthly_weights(scored, PX, UNI, market_nav=market, cash_code="511880", **port)
    eq, _ = backtest_monthly_strategy(PX, w, transaction_cost_bps=10.0)
    eq["date"] = pd.to_datetime(eq["date"])
    r = pd.Series(eq["strategy_return"].to_numpy(), index=eq["date"])
    cashw = w[w["fund_code"] == "511880"].groupby("date")["weight"].sum()
    return r, float(cashw.reindex(pd.to_datetime(w["date"].unique())).fillna(0).mean())


def seg(r, lo, hi):
    x = r.loc[lo:hi].dropna()
    if len(x) < 20: return None
    nav = (1 + x).cumprod(); ann = nav.iloc[-1] ** (TD / len(x)) - 1
    vol = x.std() * np.sqrt(TD); mdd = (nav / nav.cummax() - 1).min()
    return ann, ann / vol if vol else np.nan, float(mdd)


def main():
    global PX, UNI
    UNI = load_etf_universe(data_dir=DEFAULT_DATA_DIR)
    PX = load_nav_prices(UNI["fund_code"].tolist(), data_dir=DEFAULT_DATA_DIR,
                         start="2017-01-01", end="2026-06-05").dropna(axis=1, thresh=280)
    UNI = UNI[UNI["fund_code"].isin(PX.columns)].copy()
    market = hs()
    print("计算因子面板..."); panels, _ = factor_panels(PX)
    fac = compute_factor_panel(PX)
    fac = add_factor_cols(fac, panels, ["low_corr_120", "resid_mom_120", "dd_resilience_252"])

    B3 = {"combo_eff_accel": 0.45, "momentum_12_1": 0.35, "fund_hit_rate_20": 0.20}
    RISK = {"downside_vol_60d": -0.15, "max_drawdown_60d": 0.10}   # 下行波动口径
    PORT_ORIG = {"top_n": 20, "max_per_theme": 3, "max_weight": 0.12, "volatility_target": 0.18, "weak_market_exposure": 0.60}
    PORT_RELAX = {"top_n": 20, "max_per_theme": 3, "max_weight": 0.12, "volatility_target": 0.30, "weak_market_exposure": 1.0}

    variants = [
        ("S0 三因子·原风险层(现状)", {**B3, **RISK}, PORT_ORIG),
        ("S1 三因子·松风险层", {**B3, **RISK}, PORT_RELAX),
        ("S2 +low_corr(0.20)·松", {**B3, **RISK, "low_corr_120": 0.20}, PORT_RELAX),
        ("S3 ++resid_mom(0.20)·松", {**B3, **RISK, "low_corr_120": 0.20, "resid_mom_120": 0.20}, PORT_RELAX),
        ("S4 +++dd_resil(0.15)·松", {**B3, **RISK, "low_corr_120": 0.20, "resid_mom_120": 0.20, "dd_resilience_252": 0.15}, PORT_RELAX),
    ]
    print(f"\n{'方案':26}{'全期Sharpe':>11}{'全期年化':>10}{'MDD':>8}{'23-24年化':>11}{'2025年化':>10}{'均现金%':>9}")
    out = {}
    for label, wts, port in variants:
        r, cash = run(fac, wts, port, label, market)
        out[label] = r
        full = seg(r, "2020-02-04", "2026-06-05")
        s2324 = seg(r, "2023-01-01", "2024-12-31")
        s25 = seg(r, "2025-01-01", "2026-06-05")
        print(f"{label:26}{full[1]:>11.2f}{full[0]:>10.1%}{full[2]:>8.1%}"
              f"{s2324[0]:>11.1%}{s25[0]:>10.1%}{cash:>9.0%}")

    nav = pd.DataFrame({k: (1 + v.loc['2020-02-04':]).cumprod() for k, v in out.items()})
    od = ROOT / "outputs_factor_incremental"; od.mkdir(exist_ok=True)
    nav.to_csv(od / "incremental_nav.csv", encoding="utf-8-sig")
    print(f"\n输出 -> {od}")


if __name__ == "__main__":
    main()
