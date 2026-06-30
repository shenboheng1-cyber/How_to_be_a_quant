"""matplotlib 出图: 五因子/三因子/30-70/multi_asset/沪深300, 从最早有波动开始, 标OOS分界。"""
import sys, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang HK", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, "/Users/shenboheng/Documents/ClaudeCode/投顾策略组合/multi_asset_core")
from etf_factor_strategy.data import load_etf_universe, load_nav_prices, DEFAULT_DATA_DIR
from etf_factor_strategy.engine import compute_factor_panel, score_factors_with_weights, make_robust_monthly_weights, backtest_monthly_strategy
from factor_research import factor_panels
from incremental_factor_backtest import add_factor_cols
TD = 252
RISK = {"downside_vol_60d": -0.15, "max_drawdown_60d": 0.10}   # 用户口径: 下行波动
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
    # 从最早有波动的共同日开始
    c = r3.dropna().index.intersection(ma.dropna().index).intersection(h.dropna().index)
    r3, r5, ma, h = r3.reindex(c).fillna(0), r5.reindex(c).fillna(0), ma.reindex(c).fillna(0), h.reindex(c).fillna(0)
    blend = 0.3 * r3 + 0.7 * ma
    S = {"五因子(样本内)": r5, "三因子": r3, "30/70组合": blend, "multi_asset": ma, "沪深300": h}
    nav = {k: (1 + v).cumprod() for k, v in S.items()}

    def mets(r):
        r = r.dropna(); n = (1 + r).cumprod(); ann = n.iloc[-1] ** (TD / len(r)) - 1; vol = r.std() * np.sqrt(TD)
        return ann, ann / vol, float((n / n.cummax() - 1).min())
    fig, ax = plt.subplots(figsize=(13, 7))
    styles = {"五因子(样本内)": ("#ba7517", 1.6, "--"), "三因子": ("#c0392b", 1.6, "-"),
              "30/70组合": ("#7f4ab7", 2.6, "-"), "multi_asset": ("#1d9e75", 1.6, "-"), "沪深300": ("#999999", 1.2, "-")}
    for k, v in nav.items():
        a, s, m = mets(S[k])
        col, lw, ls = styles[k]
        ax.plot(v.index, v / v.iloc[0], label=f"{k}  年化{a:.1%} Sharpe{s:.2f} MDD{m:.1%}", color=col, lw=lw, ls=ls)
    ax.axvline(pd.Timestamp("2023-01-01"), color="#888", ls=":", lw=1)
    ax.text(pd.Timestamp("2023-01-05"), ax.get_ylim()[1]*0.97, " 动量样本外起点", color="#888", fontsize=9, va="top")
    ax.set_title(f"ETF策略对比 (从{c.min().date()}起, 全区间)", fontsize=13)
    ax.set_ylabel("净值(起点=1)"); ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "outputs_factor_incremental" / "five_strategy_full.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("起点:", c.min().date(), "| 终点:", c.max().date())
    print("SAVED", out)

    def seg(r, lo=None, hi=None):
        r = r.loc[lo:hi].dropna() if lo else r.dropna()
        n = (1 + r).cumprod(); ann = n.iloc[-1] ** (TD / len(r)) - 1; vol = r.std() * np.sqrt(TD)
        return f"年化{ann:.1%} Sharpe{ann/vol:.2f} MDD{(n/n.cummax()-1).min():.1%} 累计{n.iloc[-1]-1:.0%}"
    no25 = lambda r: r[r.index.year != 2025]
    print("\n指标(downside_vol口径):")
    for k, v in S.items():
        print(f"  {k:14} 全区间[{seg(v)}] | 23-26[{seg(v,'2023-01-01','2026-06-05')}] | 去25 Sharpe{(lambda r:(((1+r).cumprod().iloc[-1]**(TD/len(r))-1)/(r.std()*np.sqrt(TD))))(no25(v).dropna()):.2f}")


if __name__ == "__main__":
    main()
