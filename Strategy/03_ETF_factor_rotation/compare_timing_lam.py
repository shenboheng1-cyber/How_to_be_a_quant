"""2x2 对比：弱市择时(开/关) × 部分再平衡 lam(1.0/0.4)。
其余口径与 V2 一致(FACTOR_WEIGHTS_V2、top20、同主题3、单票12%、vol_target0.18、2018起、净5bps)。
用 make_robust_monthly_weights(无 hysteresis)统一构造，以纯净隔离 弱市择时×lam 两个变量。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_robust_monthly_weights,
    backtest_monthly_strategy, FACTOR_WEIGHTS_V2)
import hfq_common as H

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti TC", "STHeiti", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
COST = 5.0


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, amt, _ = H.load_hfq()
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    market = H.hs300()
    fac = compute_factor_panel(px)
    scored = score_factors_with_weights(fac, FACTOR_WEIGHTS_V2, score_column="risk_adjusted_score")

    configs = {
        "含弱市 + lam1.0（≈V1完整）": dict(timing=True, lam=1.0),
        "含弱市 + lam0.4（你问的）": dict(timing=True, lam=0.4),
        "去弱市 + lam1.0": dict(timing=False, lam=1.0),
        "去弱市 + lam0.4（≈V2）": dict(timing=False, lam=0.4),
    }

    rows, rets = [], {}
    for name, cfg in configs.items():
        w = make_robust_monthly_weights(
            scored, px, uni, top_n=20, max_per_theme=3, max_weight=0.12,
            volatility_target=0.18, cash_code="511880",
            market_nav=market if cfg["timing"] else None, weak_market_exposure=0.60)
        eq, eff = backtest_monthly_strategy(px, w, transaction_cost_bps=0.0, rebalance_lambda=cfg["lam"])
        g = eq.copy(); g["date"] = pd.to_datetime(g["date"]); g = g.set_index("date").loc[H.START:H.END]
        gr, tn = g["strategy_return"], g["turnover"]
        net5 = gr - tn * COST / 1e4
        rets[name] = net5
        m = H.metrics(net5)
        roll = net5.rolling(H.TD)
        rs = (roll.mean() / roll.std(ddof=0) * np.sqrt(H.TD)).dropna()
        rows.append({
            "配置": name, "年化": m["ann"], "Sharpe": m["sharpe"], "MDD": m["mdd"],
            "滚动S<0占比": (rs < 0).mean(), "年化换手": tn.sum() / (len(tn) / H.TD),
            "S@1亿": H.metrics(H.apply_impact(gr, tn, eff, px, amt, 1e8))["sharpe"],
            "S@5亿": H.metrics(H.apply_impact(gr, tn, eff, px, amt, 5e8))["sharpe"],
        })

    df = pd.DataFrame(rows)
    show = df.copy()
    for c in ["年化", "MDD", "滚动S<0占比"]:
        show[c] = (show[c] * 100).map(lambda x: f"{x:+.1f}%" if c == "年化" else f"{x:.1f}%")
    for c in ["Sharpe", "S@1亿", "S@5亿"]:
        show[c] = show[c].map(lambda x: f"{x:.2f}")
    show["年化换手"] = show["年化换手"].map(lambda x: f"{x:.1f}x")
    print("=" * 92)
    print(f"弱市择时 × lam 2x2 对比（后复权市价, 含{COST:.0f}bps, 2018-2026, FACTOR_WEIGHTS_V2）")
    print("=" * 92)
    print(show.to_string(index=False))
    df.to_csv(H.ROOT / "outputs_param_sweep/timing_lam_2x2.csv", index=False, encoding="utf-8-sig")

    # 净值图
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sty = {"含弱市 + lam1.0（≈V1完整）": ("#27ae60", 1.7), "含弱市 + lam0.4（你问的）": ("#c0392b", 2.4),
           "去弱市 + lam1.0": ("#95a5a6", 1.4), "去弱市 + lam0.4（≈V2）": ("#e67e22", 1.7)}
    for name, r in rets.items():
        nav = (1 + r).cumprod(); m = H.metrics(r)
        c, lw = sty[name]
        ax.plot(nav.index, nav.values, label=f"{name}  年化{m['ann']:.1%}/Sharpe{m['sharpe']:.2f}/MDD{m['mdd']:.1%}",
                color=c, lw=lw)
    ax.set_title("弱市择时 × lam 2×2 净值对比（后复权市价，含成本，2018-2026）")
    ax.set_ylabel("净值（起点=1）"); ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(H.ROOT / "figures/10_timing_lam.png", dpi=130); plt.close(fig)
    print("\n图 -> figures/10_timing_lam.png")


if __name__ == "__main__":
    main()
