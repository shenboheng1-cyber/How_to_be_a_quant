"""5因子 walk-forward 验证 + 新因子×紧/松风险层对照。
候选 = 因子preset(3/4/5因子) × 风险层(紧/松) 共6个。每折用 train 指标选最优, 验证下一年, 拼接OOS。
诚实性: 因子权重/风险层都由 train 选, 不带后视。
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
    compute_factor_panel, score_factors_with_weights, make_robust_monthly_weights, backtest_monthly_strategy)
from factor_research import factor_panels
from incremental_factor_backtest import add_factor_cols

TD = 252
RISK = {"downside_vol_60d": -0.15, "max_drawdown_60d": 0.10}   # 下行波动口径
F3 = {"combo_eff_accel": 0.45, "momentum_12_1": 0.35, "fund_hit_rate_20": 0.20}
F4 = {**F3, "low_corr_120": 0.20}
F5 = {**F4, "resid_mom_120": 0.20}
TIGHT = {"top_n": 20, "max_per_theme": 3, "max_weight": 0.12, "volatility_target": 0.18, "weak_market_exposure": 0.60}
RELAX = {"top_n": 20, "max_per_theme": 3, "max_weight": 0.12, "volatility_target": 0.30, "weak_market_exposure": 1.0}
CANDS = {
    "f3_紧": ({**F3, **RISK}, TIGHT), "f3_松": ({**F3, **RISK}, RELAX),
    "f4_紧": ({**F4, **RISK}, TIGHT), "f4_松": ({**F4, **RISK}, RELAX),
    "f5_紧": ({**F5, **RISK}, TIGHT), "f5_松": ({**F5, **RISK}, RELAX),
}
FOLDS = [("2023", "2020-02-04", "2022-12-30", "2023-01-03", "2023-12-29"),
         ("2024", "2020-02-04", "2023-12-29", "2024-01-02", "2024-12-31"),
         ("2025", "2020-02-04", "2024-12-31", "2025-01-02", "2025-12-31"),
         ("2026", "2020-02-04", "2025-12-31", "2026-01-02", "2026-06-05")]


def hs():
    con = sqlite3.connect(f"file:{DEFAULT_DATA_DIR}/idx_store.db?mode=ro", uri=True)
    d = pd.read_sql_query("SELECT date,close FROM index_daily WHERE symbol='CSI300' ORDER BY date", con)
    con.close(); d["date"] = pd.to_datetime(d["date"]); return d.set_index("date")["close"].astype(float)


def metrics(nav):
    nav = nav.dropna() / nav.dropna().iloc[0]; r = nav.pct_change().fillna(0.0)
    ann = nav.iloc[-1] ** (TD / len(nav)) - 1; vol = r.std(ddof=0) * np.sqrt(TD)
    mdd = (nav / nav.cummax() - 1).min()
    return dict(ann=float(ann), sharpe=float(ann / vol) if vol else np.nan,
                mdd=float(mdd), calmar=float(ann / abs(mdd)) if mdd else np.nan)


def main():
    uni = load_etf_universe(data_dir=DEFAULT_DATA_DIR)
    px = load_nav_prices(uni["fund_code"].tolist(), data_dir=DEFAULT_DATA_DIR, start="2017-01-01", end="2026-06-05").dropna(axis=1, thresh=280)
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    market = hs()
    print("因子面板..."); panels, _ = factor_panels(px)
    fac = compute_factor_panel(px)
    fac = add_factor_cols(fac, panels, ["low_corr_120", "resid_mom_120", "dd_resilience_252"])

    # 每候选全期净值(只算一次)
    navs = {}
    for name, (wts, port) in CANDS.items():
        sc = score_factors_with_weights(fac, wts, score_column="s")
        w = make_robust_monthly_weights(sc, px, uni, market_nav=market, cash_code="511880", **port)
        eq, _ = backtest_monthly_strategy(px, w, transaction_cost_bps=10.0)
        navs[name] = pd.Series(eq["nav"].to_numpy(), index=pd.to_datetime(eq["date"]))
        print(f"  {name} done")

    # 对照(②): 全期直接对比
    print("\n=== ② 全期对照 (2020-2026) ===")
    print(f"{'候选':10}{'Sharpe':>8}{'年化':>8}{'MDD':>8}{'23-24年化':>11}")
    for name, nav in navs.items():
        m = metrics(nav.loc["2020-02-04":])
        s2324 = metrics(nav.loc["2023-01-01":"2024-12-31"])
        print(f"{name:10}{m['sharpe']:>8.2f}{m['ann']:>8.1%}{m['mdd']:>8.1%}{s2324['ann']:>11.1%}")

    # walk-forward(①): 每折 train 选, test 拼
    def sel_score(nav, s, e):
        m = metrics(nav.loc[s:e])
        return m  # 用 calmar+sharpe+ann - |mdd|
    chosen, curves = [], []
    for fold, ts, te, vs, ve in FOLDS:
        rows = []
        for name, nav in navs.items():
            m = sel_score(nav, ts, te)
            rows.append((name, m))
        df = pd.DataFrame({n: m for n, m in rows}).T
        df["score"] = (df["calmar"].rank(pct=True) + df["sharpe"].rank(pct=True)
                       + df["ann"].rank(pct=True) - df["mdd"].abs().rank(pct=True))
        best = df["score"].idxmax()
        chosen.append((fold, best))
        seg = navs[best].loc[vs:ve]
        curves.append(seg / seg.iloc[0])
    # 拼接
    stitched = []
    base = 1.0
    for c in curves:
        stitched.append(c * base); base = float((c * base).iloc[-1])
    oos = pd.concat(stitched)
    print("\n=== ① Walk-forward (每折 train 选, 拼接OOS 2023-2026) ===")
    print("各折选中:", ", ".join(f"{f}->{b}" for f, b in chosen))
    mo = metrics(oos)
    print(f"拼接OOS: Sharpe{mo['sharpe']:.2f} 年化{mo['ann']:.1%} MDD{mo['mdd']:.1%} Calmar{mo['calmar']:.2f}")
    o2324 = metrics(oos.loc[:"2024-12-31"])
    print(f"其中 2023-2024 段: 年化{o2324['ann']:.1%} Sharpe{o2324['sharpe']:.2f}")
    # 对照: 始终 f3_紧(原现状) 的同口径拼接
    base = 1.0; st3 = []
    for fold, ts, te, vs, ve in FOLDS:
        seg = navs["f3_紧"].loc[vs:ve]; c = seg / seg.iloc[0]; st3.append(c * base); base = float((c * base).iloc[-1])
    oos3 = pd.concat(st3); m3 = metrics(oos3); m3_2324 = metrics(oos3.loc[:"2024-12-31"])
    print(f"对照(始终f3_紧/现状): OOS Sharpe{m3['sharpe']:.2f} 年化{m3['ann']:.1%} MDD{m3['mdd']:.1%} | 23-24年化{m3_2324['ann']:.1%}")
    od = ROOT / "outputs_factor_incremental"; od.mkdir(exist_ok=True)
    pd.DataFrame({"wf_5factor_select": oos, "f3_baseline": oos3}).to_csv(od / "walk_forward_5factor.csv", encoding="utf-8-sig")
    print(f"\n输出 -> {od}")


if __name__ == "__main__":
    main()
