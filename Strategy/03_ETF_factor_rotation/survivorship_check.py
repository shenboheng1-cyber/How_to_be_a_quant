"""幸存者偏差量化: 把已清盘 ETF 并入 universe, 按 PIT 重跑动量策略, 对比。

清盘 ETF 识别: 用 delisted_etf_nav.parquet 中净值末日 < 2026-04 的(真停更, status标签不可靠)。
PIT 建模: 清盘 ETF 在其存活期有净值, 末日后净值停更 -> backtest 的 pct_change().fillna(0) 令其
冻结在末净值(≈拿回NAV现金), tradable 过滤在下次调仓把它踢成现金重投。无未来函数。
固定 walk-forward 选中的参数(base+risk_light+balanced), 对比 survivor-only vs +delisted。
"""
from __future__ import annotations
import sys, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from etf_factor_strategy.data import load_etf_universe, load_nav_prices, DEFAULT_DATA_DIR
from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_robust_monthly_weights,
    backtest_monthly_strategy,
)

BASE = {"combo_eff_accel": 0.45, "momentum_12_1": 0.35, "fund_hit_rate_20": 0.20,
        "vol_60d": -0.15, "max_drawdown_60d": 0.10}
PORT = {"top_n": 20, "max_per_theme": 3, "max_weight": 0.12,
        "volatility_target": 0.18, "weak_market_exposure": 0.60}
TD = 252


def hs300():
    con = sqlite3.connect(f"file:{DEFAULT_DATA_DIR/'idx_store.db'}?mode=ro", uri=True)
    d = pd.read_sql_query("SELECT date, close FROM index_daily WHERE symbol='CSI300' ORDER BY date", con)
    con.close(); d["date"] = pd.to_datetime(d["date"]); return d.set_index("date")["close"].astype(float)


def survivor_panel():
    uni = load_etf_universe(data_dir=DEFAULT_DATA_DIR)
    px = load_nav_prices(uni["fund_code"].tolist(), data_dir=DEFAULT_DATA_DIR, start="2017-01-01", end="2026-06-05")
    return uni, px


def delisted_panel():
    nav = pd.read_parquet(ROOT / "outputs_survivorship/delisted_etf_nav.parquet")
    nav["date"] = pd.to_datetime(nav["date"])
    last = nav.groupby("fund_code")["date"].max()
    dead = last[last < "2026-04-01"].index                       # 真停更
    nav = nav[nav["fund_code"].isin(dead) & (nav["date"] >= "2017-01-01")]
    px = nav.pivot_table(index="date", columns="fund_code", values="cum_nav").sort_index()
    b = pd.read_csv(ROOT / "outputs_survivorship/delisted_etf_basic.csv")
    b["fund_code"] = b["ts_code"].str.split(".").str[0]
    uni = b[b["fund_code"].isin(px.columns)][["fund_code", "name"]].rename(columns={"name": "fund_name"})
    uni["fund_type"] = "ETF"
    return uni, px


def run(uni, px, label, cost_bps=10):
    px = px.dropna(axis=1, thresh=280)
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    fac = compute_factor_panel(px)
    scored = score_factors_with_weights(fac, BASE, score_column="risk_adjusted_score")
    w = make_robust_monthly_weights(scored, px, uni, market_nav=hs300(), cash_code="511880", **PORT)
    eq, _ = backtest_monthly_strategy(px, w, transaction_cost_bps=cost_bps)
    eq["date"] = pd.to_datetime(eq["date"])
    r = pd.Series(eq["strategy_return"].to_numpy(), index=eq["date"])
    return r, w


def stats(r, lbl):
    r = r.dropna(); nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (TD / len(r)) - 1; vol = r.std(ddof=0) * np.sqrt(TD)
    mdd = (nav / nav.cummax() - 1).min()
    print(f"  {lbl:30} 年化{ann:+.2%} 波动{vol:.1%} Sharpe{ann/vol:.2f} MDD{mdd:.1%} Calmar{ann/abs(mdd):.2f}")
    return dict(ann=ann, sharpe=ann / vol, mdd=float(mdd))


def main():
    su, spx = survivor_panel()
    du, dpx = delisted_panel()
    print(f"survivor ETF {spx.shape[1]} | 清盘ETF并入 {dpx.shape[1]}")
    # 合并面板(列并集)
    cpx = spx.join(dpx, how="outer").sort_index()
    cuni = pd.concat([su[["fund_code", "fund_name", "fund_type"]] if "fund_name" in su else su,
                      du], ignore_index=True).drop_duplicates("fund_code")
    if "fund_name" not in su.columns:
        su = su.rename(columns={c: "fund_name" for c in su.columns if c == "fund_name"})
    cuni = pd.concat([su.assign(fund_name=su.get("fund_name", su["fund_code"])), du], ignore_index=True).drop_duplicates("fund_code")

    print("\n重跑 (survivor-only)..."); r_sv, w_sv = run(su, spx, "sv")
    print("重跑 (含清盘ETF)..."); r_full, w_full = run(cuni, cpx, "full")

    for tag, sl in [("全样本2020-2026", "2020-02-04"), ("样本外2023-2026", "2023-01-01")]:
        print(f"\n=== {tag} ===")
        a = stats(r_sv.loc[sl:], "survivor-only(原, 有偏差)")
        b = stats(r_full.loc[sl:], "含清盘ETF(修正)")
        print(f"  >>> Sharpe 偏差 {a['sharpe']-b['sharpe']:+.2f} (修正后 {b['sharpe']:.2f}); 年化偏差 {a['ann']-b['ann']:+.2%}")

    # 修正版有多少次/多少周期持有了清盘ETF
    dead_codes = set(dpx.columns)
    held_dead = w_full[w_full["fund_code"].isin(dead_codes)]
    print(f"\n修正版曾持有清盘ETF: {held_dead['fund_code'].nunique()} 只, "
          f"{held_dead['date'].nunique()} 个调仓月有持仓, 平均权重 {held_dead['weight'].mean():.1%}")
    out = ROOT / "outputs_survivorship"
    pd.DataFrame({"survivor_only": (1+r_sv).cumprod(), "with_delisted": (1+r_full).cumprod()}).to_csv(
        out / "survivorship_compare_nav.csv", encoding="utf-8-sig")
    print(f"\n输出 -> {out}")


if __name__ == "__main__":
    main()
