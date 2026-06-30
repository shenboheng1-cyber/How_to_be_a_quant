"""全区间(2018起) + 2023-2026 OOS + 去2025 指标, 供 README。"""
import sys, sqlite3, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/shenboheng/Documents/ClaudeCode/投顾策略组合/multi_asset_core")
from etf_factor_strategy.data import load_etf_universe, load_nav_prices, DEFAULT_DATA_DIR
from etf_factor_strategy.engine import compute_factor_panel, score_factors_with_weights, make_robust_monthly_weights, backtest_monthly_strategy
from factor_research import factor_panels
from incremental_factor_backtest import add_factor_cols
TD = 252
RISK = {"downside_vol_60d": -0.15, "max_drawdown_60d": 0.10}   # 下行波动口径
F3 = {"combo_eff_accel": 0.45, "momentum_12_1": 0.35, "fund_hit_rate_20": 0.20}
F5 = {**F3, "low_corr_120": 0.20, "resid_mom_120": 0.20}
TIGHT = {"top_n": 20, "max_per_theme": 3, "max_weight": 0.12, "volatility_target": 0.18, "weak_market_exposure": 0.60}


def hs():
    con = sqlite3.connect(f"file:{DEFAULT_DATA_DIR}/idx_store.db?mode=ro", uri=True)
    d = pd.read_sql_query("SELECT date,close FROM index_daily WHERE symbol='CSI300' ORDER BY date", con)
    con.close(); d["date"] = pd.to_datetime(d["date"]); return d.set_index("date")["close"].astype(float)


def strat(fac, wts, px, uni, mkt):
    sc = score_factors_with_weights(fac, wts, score_column="s")
    w = make_robust_monthly_weights(sc, px, uni, market_nav=mkt, cash_code="511880", **TIGHT)
    eq, _ = backtest_monthly_strategy(px, w, transaction_cost_bps=10.0)
    return pd.Series(eq["strategy_return"].to_numpy(), index=pd.to_datetime(eq["date"]))


def m(r, lo=None, hi=None):
    r = r.loc[lo:hi].dropna() if lo else r.dropna()
    if len(r) < 20: return None
    nav = (1 + r).cumprod(); ann = nav.iloc[-1] ** (TD / len(r)) - 1; vol = r.std() * np.sqrt(TD)
    return dict(ann=round(ann, 4), vol=round(vol, 4), sharpe=round(ann / vol, 2),
                mdd=round(float((nav / nav.cummax() - 1).min()), 4), calmar=round(ann / abs((nav/nav.cummax()-1).min()), 2),
                cum=round(nav.iloc[-1] - 1, 4))


def main():
    uni = load_etf_universe(data_dir=DEFAULT_DATA_DIR)
    px = load_nav_prices(uni["fund_code"].tolist(), data_dir=DEFAULT_DATA_DIR, start="2017-01-01", end="2026-06-05").dropna(axis=1, thresh=280)
    uni = uni[uni["fund_code"].isin(px.columns)].copy(); mkt = hs()
    panels, _ = factor_panels(px)
    fac = compute_factor_panel(px); fac = add_factor_cols(fac, panels, ["low_corr_120", "resid_mom_120", "dd_resilience_252"])
    r3 = strat(fac, {**F3, **RISK}, px, uni, mkt); r5 = strat(fac, {**F5, **RISK}, px, uni, mkt)
    from core_satellite.run_allocator import run_multi_asset_on_navstore
    ma = run_multi_asset_on_navstore("2018-01-01", "2026-06-05").returns
    h = mkt.pct_change()
    c = r3.dropna().index.intersection(ma.dropna().index).intersection(h.dropna().index)
    r3, r5, ma, h = r3.reindex(c).fillna(0), r5.reindex(c).fillna(0), ma.reindex(c).fillna(0), h.reindex(c).fillna(0)
    blend = 0.3 * r3 + 0.7 * ma
    no25 = lambda r: r[r.index.year != 2025]
    S = {"三因子": r3, "五因子(样本内)": r5, "30/70组合": blend, "multi_asset": ma, "沪深300": h}
    res = {}
    for k, v in S.items():
        res[k] = {"全区间18-26": m(v), "样本外23-26": m(v, "2023-01-01", "2026-06-05"), "去2025": m(no25(v))}
    print("起点", c.min().date(), "终点", c.max().date())
    print(json.dumps(res, ensure_ascii=False, indent=1))
    pd.DataFrame({k: (1 + v).cumprod() for k, v in S.items()}).to_csv(ROOT / "outputs_factor_incremental" / "five_strategy_daily_nav.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
