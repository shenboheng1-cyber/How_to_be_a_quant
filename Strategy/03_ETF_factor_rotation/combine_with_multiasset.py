"""C 阶段: ETF 三因子动量策略 × multi_asset_core 组合验证。
配比网格 + block bootstrap 显著性 + 成本敏感性。

动量侧: 固定 walk-forward 选中的参数(base alpha + risk_light + balanced portfolio),
用其 engine 重生成日收益(可变成本; 因四折同参, 该曲线≈样本外拼接曲线)。
multi_asset 侧: 复用 core_satellite 引擎(nav_store 统一数据源)取日收益。
主区间用动量真样本外 2023-01~2026-06; 另报 2020+ 长样本做稳健性。
"""
from __future__ import annotations
import sys, json, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang HK", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

MOM_ROOT = Path("/Users/shenboheng/Documents/Codex/投顾策略量化平台/etf因子策略")
MA_ROOT = Path("/Users/shenboheng/Documents/ClaudeCode/投顾策略组合/multi_asset_core")
DATA_DIR = Path("/Users/shenboheng/Documents/ClaudeCode/dataset/基金深度分析")
sys.path.insert(0, str(MOM_ROOT)); sys.path.insert(0, str(MA_ROOT))

from etf_factor_strategy.data import load_etf_universe, load_nav_prices
from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_robust_monthly_weights,
    backtest_monthly_strategy,
)

BASE = {"combo_eff_accel": 0.45, "momentum_12_1": 0.35, "fund_hit_rate_20": 0.20,
        "vol_60d": -0.15, "max_drawdown_60d": 0.10}                       # base + risk_light
PORT = {"top_n": 20, "max_per_theme": 3, "max_weight": 0.12,
        "volatility_target": 0.18, "weak_market_exposure": 0.60}          # balanced
TD = 252


def hs300_nav():
    con = sqlite3.connect(f"file:{DATA_DIR/'idx_store.db'}?mode=ro", uri=True)
    d = pd.read_sql_query("SELECT date, close FROM index_daily WHERE symbol='CSI300' ORDER BY date", con)
    con.close(); d["date"] = pd.to_datetime(d["date"]); return d.set_index("date")["close"].astype(float)


def momentum_returns(cost_bps: float) -> pd.Series:
    uni = load_etf_universe(data_dir=DATA_DIR)
    px = load_nav_prices(uni["fund_code"].tolist(), data_dir=DATA_DIR,
                         start="2017-01-01", end="2026-06-05").dropna(axis=1, thresh=280)
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    fac = compute_factor_panel(px)
    scored = score_factors_with_weights(fac, BASE, score_column="risk_adjusted_score")
    w = make_robust_monthly_weights(scored, px, uni, market_nav=hs300_nav(), cash_code="511880", **PORT)
    eq, _ = backtest_monthly_strategy(px, w, transaction_cost_bps=cost_bps)
    s = pd.Series(eq["strategy_return"].to_numpy(), index=pd.to_datetime(eq["date"]))
    return s


def multiasset_returns() -> pd.Series:
    from core_satellite.run_allocator import run_multi_asset_on_navstore
    res = run_multi_asset_on_navstore("2018-01-01", "2026-06-05")
    return res.returns


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (TD / len(r)) - 1
    vol = r.std(ddof=0) * np.sqrt(TD)
    mdd = (nav / nav.cummax() - 1).min()
    return dict(ann=float(ann), vol=float(vol), sharpe=float(ann / vol) if vol else np.nan,
                mdd=float(mdd), calmar=float(ann / abs(mdd)) if mdd < 0 else np.nan)


def blend_grid(mom, ma, label):
    c = mom.dropna().index.intersection(ma.dropna().index)
    m, a = mom.reindex(c), ma.reindex(c)
    rows = []
    for w in np.round(np.arange(0, 1.0001, 0.1), 2):
        s = stats(w * m + (1 - w) * a)
        rows.append({"w_momentum": w, **s})
    g = pd.DataFrame(rows)
    best = g.loc[g["sharpe"].idxmax()]
    print(f"\n[{label}] 共同日 {len(c)} ({c.min().date()}~{c.max().date()}) | 相关 {m.corr(a):.3f}")
    print(g.round(3).to_string(index=False))
    print(f"  最优配比 w_momentum={best['w_momentum']:.1f} -> Sharpe {best['sharpe']:.2f} 年化 {best['ann']:.1%} MDD {best['mdd']:.1%}")
    return g, c, m, a, float(best["w_momentum"])


def block_bootstrap(m, a, w_opt, n=2000, block=21, seed=42):
    """联合 block bootstrap(保留两策略同期相关). 返回 blend Sharpe 分布 + P(blend>best single)."""
    rng = np.random.default_rng(seed)
    M, A = m.to_numpy(), a.to_numpy(); N = len(M)
    nblk = int(np.ceil(N / block))
    sh_blend, sh_best, win = [], [], 0
    for _ in range(n):
        starts = rng.integers(0, N - block + 1, size=nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:N]
        mm, aa = M[idx], A[idx]
        bl = w_opt * mm + (1 - w_opt) * aa
        def sh(x):
            v = x.std(ddof=0); return (x.mean() * TD) / (v * np.sqrt(TD)) if v > 0 else np.nan
        sb = sh(bl); single = max(sh(mm), sh(aa))
        sh_blend.append(sb); sh_best.append(single); win += int(sb > single)
    sh_blend = np.array(sh_blend)
    return dict(blend_sharpe_mean=float(np.nanmean(sh_blend)),
                blend_sharpe_p05=float(np.nanpercentile(sh_blend, 5)),
                blend_sharpe_p95=float(np.nanpercentile(sh_blend, 95)),
                prob_blend_beats_best_single=float(win / n))


def main():
    out = MOM_ROOT / "outputs_combined"; out.mkdir(exist_ok=True)
    print("重生成动量(多成本)..."); mom = {bp: momentum_returns(bp) for bp in (5, 10, 15, 20)}
    print("取 multi_asset 日收益..."); ma = multiasset_returns()
    hs = hs300_nav().pct_change()

    # 主分析: 动量真样本外 2023-2026, 现实成本 10bps
    OOS = "2023-01-01"
    mom10 = mom[10]
    g_oos, c_oos, m_oos, a_oos, w_oos = blend_grid(mom10.loc[OOS:], ma.loc[OOS:], "样本外2023-2026 · 动量10bps")
    g_oos.to_csv(out / "blend_grid_oos.csv", index=False)

    # 长样本 2020-2026 稳健性(动量含in-sample期)
    g_full, *_ = blend_grid(mom10.loc["2020-02-04":], ma.loc["2020-02-04":], "长样本2020-2026 · 动量10bps")
    g_full.to_csv(out / "blend_grid_full.csv", index=False)

    # 成本敏感性(样本外, 最优配比下)
    print("\n--- 成本敏感性 (样本外, w_momentum 各自最优) ---")
    cost_rows = []
    for bp in (5, 10, 15, 20):
        gg, cc, mm, aa, ww = blend_grid(mom[bp].loc[OOS:], ma.loc[OOS:], f"cost{bp}bps")
        b = stats(ww * mm + (1 - ww) * aa)
        cost_rows.append({"cost_bps": bp, "w_opt": ww, **b})
    pd.DataFrame(cost_rows).to_csv(out / "cost_sensitivity.csv", index=False)

    # bootstrap 显著性(样本外, 10bps, 最优配比)
    print("\n--- block bootstrap 显著性 (样本外 10bps, w=%.1f) ---" % w_oos)
    bs = block_bootstrap(m_oos, a_oos, w_oos)
    print(json.dumps(bs, ensure_ascii=False, indent=2))

    # 净值图 + 单独指标
    summ = {"动量(10bps)": stats(m_oos), "multi_asset": stats(a_oos),
            f"组合(w={w_oos:.1f})": stats(w_oos * m_oos + (1 - w_oos) * a_oos),
            "沪深300": stats(hs.reindex(c_oos).fillna(0))}
    pd.DataFrame(summ).T.to_csv(out / "combined_summary.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(12, 6))
    for lbl, r in [("动量(10bps)", m_oos), ("multi_asset", a_oos),
                   (f"组合w={w_oos:.1f}", w_oos * m_oos + (1 - w_oos) * a_oos)]:
        nav = (1 + r).cumprod(); ax.plot(nav.index, nav / nav.iloc[0], label=lbl, lw=2 if "组合" in lbl else 1.4)
    hsnav = (1 + hs.reindex(c_oos).fillna(0)).cumprod(); ax.plot(hsnav.index, hsnav / hsnav.iloc[0], label="沪深300", lw=1, color="#999")
    ax.legend(); ax.set_title("动量×multi_asset 组合 (样本外 2023-2026)"); ax.grid(alpha=0.3)
    fig.savefig(out / "combined_nav.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    (out / "bootstrap.json").write_text(json.dumps(bs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n样本外三方+组合:"); print(pd.DataFrame(summ).T.round(3).to_string())
    print(f"\n输出 -> {out}")


if __name__ == "__main__":
    main()
