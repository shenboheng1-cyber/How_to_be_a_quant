"""只改打分权重：删 fund_hit_rate_20，按 ICIR 把 0.20 分给 combo/momentum，风险项不变。
其余(组合/风险层/lam=0.4/成本)与 V2 完全一致。对比并出净值图。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_monthly_weights_v2,
    backtest_monthly_strategy, FACTOR_WEIGHTS_V2)
import hfq_common as H

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti TC", "STHeiti", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
LAM, COST = 0.4, 5.0

W_CUR = dict(FACTOR_WEIGHTS_V2)  # 含 hit_rate 0.20
W_NEW = {"combo_eff_accel": 0.58, "momentum_12_1": 0.42, "vol_60d": -0.15, "max_drawdown_60d": 0.10}


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, amt, _ = H.load_hfq()
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    market = H.hs300()
    fac = compute_factor_panel(px)

    def strat(weights):
        sc = score_factors_with_weights(fac, weights, score_column="s")
        w = make_monthly_weights_v2(sc, px, uni, buffer_rank=35, volatility_target=0.18, **H.PORT)
        eq, _ = backtest_monthly_strategy(px, w, transaction_cost_bps=COST, rebalance_lambda=LAM)
        return H.to_ret(eq)

    r_cur, r_new = strat(W_CUR), strat(W_NEW)
    # 基准
    wb = H.equal_weight_basket(px)
    eqb, _ = backtest_monthly_strategy(px, wb, transaction_cost_bps=0.0)
    r_bk = H.to_ret(eqb)
    hs = market.loc[H.START:H.END]; r_hs = hs.pct_change().dropna()

    rows = [H.metrics(r_cur, "V2 现状(含hit_rate)"), H.metrics(r_new, "V2 删hit_rate·按IC重配"),
            H.metrics(r_bk, "等权ETF篮子"), H.metrics(r_hs, "沪深300")]
    df = pd.DataFrame(rows)
    show = df.copy()
    for c in ["total", "ann", "vol", "mdd"]:
        show[c] = (show[c] * 100).map(lambda x: f"{x:+.1f}%")
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}")
    show["calmar"] = show["calmar"].map(lambda x: f"{x:.2f}")
    print("=" * 70)
    print(f"权重对比 (后复权市价, 含{COST:.0f}bps, lam={LAM}, 2018-2026)")
    print("  现状:", W_CUR)
    print("  新版:", W_NEW)
    print("=" * 70)
    print(show[["label", "total", "ann", "vol", "sharpe", "mdd", "calmar"]].to_string(index=False))

    # 分年度
    yr = pd.DataFrame({"现状": r_cur, "新版": r_new.reindex(r_cur.index).fillna(0)})
    ann = yr.groupby(yr.index.year).apply(lambda x: (1 + x).prod() - 1, include_groups=False)
    print("\n分年度收益 现状 vs 新版:")
    print((ann * 100).round(1).astype(str).add("%").to_string())

    # 净值图
    fig, ax = plt.subplots(figsize=(10, 5))
    for r, lab, c, lw in [(r_new, f"V2 删hit·重配 (年化{H.metrics(r_new)['ann']:.1%} Sharpe{H.metrics(r_new)['sharpe']:.2f})", "#c0392b", 2.2),
                          (r_cur, f"V2 现状 (年化{H.metrics(r_cur)['ann']:.1%} Sharpe{H.metrics(r_cur)['sharpe']:.2f})", "#e67e22", 1.6),
                          (r_bk, "等权ETF篮子", "#2980b9", 1.2), (r_hs, "沪深300", "#7f8c8d", 1.2)]:
        nav = (1 + r).cumprod()
        ax.plot(nav.index, nav.values, label=lab, color=c, lw=lw, alpha=0.9 if lw < 2 else 1)
    ax.set_title("删 fund_hit_rate_20 + 按 IC 重配权重 vs 现状（后复权市价，含成本，2018-2026）")
    ax.set_ylabel("净值（起点=1）"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(H.ROOT / "figures/08_reweight.png", dpi=130); plt.close(fig)
    print(f"\n图 -> figures/08_reweight.png")


if __name__ == "__main__":
    main()
