"""把 combo_eff_accel 拆成 efficiency_20d + fund_ret_accel_20_60 两个独立因子，
按各自 ICIR 分配 combo 原 0.45 预算（其余 V2 不变），回测对比。"""
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


def rank_ic(fac_me, ret_me):
    out = {}
    for d in fac_me.index:
        f, r = fac_me.loc[d], ret_me.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= 10:
            out[d] = f[m].rank().corr(r[m].rank())
    return pd.Series(out)


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, amt, _ = H.load_hfq()
    px = px.loc[:H.END]
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    market = H.hs300()
    fac = compute_factor_panel(px)
    fac["date"] = pd.to_datetime(fac["date"])

    # IC of the two components (2018-2026, monthly)
    me = [d for d in px.index.to_series().groupby(px.index.to_period("M")).max()
          if pd.Timestamp(H.START) <= d <= pd.Timestamp(H.END)]
    me_idx = pd.DatetimeIndex(me)
    fwd = px.reindex(me_idx).shift(-1) / px.reindex(me_idx) - 1.0
    icir = {}
    for f in ["efficiency_20d", "fund_ret_accel_20_60"]:
        fw = fac.pivot_table(index="date", columns="fund_code", values=f).reindex(me_idx)
        ic = rank_ic(fw, fwd)
        icir[f] = ic.mean() / ic.std(ddof=1)
        print(f"  {f}: IC均值 {ic.mean():+.3f}  ICIR {icir[f]:+.3f}")
    e, a = icir["efficiency_20d"], icir["fund_ret_accel_20_60"]
    wE, wA = 0.45 * e / (e + a), 0.45 * a / (e + a)
    print(f"  -> 按 ICIR 分配 combo 的 0.45: efficiency={wE:.3f}, accel={wA:.3f}")

    base_rest = {"momentum_12_1": 0.35, "fund_hit_rate_20": 0.20, "vol_60d": -0.15, "max_drawdown_60d": 0.10}
    weight_sets = {
        "现状(combo合成 0.45)": dict(FACTOR_WEIGHTS_V2),
        "拆分·按ICIR": {"efficiency_20d": round(wE, 3), "fund_ret_accel_20_60": round(wA, 3), **base_rest},
        "拆分·等权(0.225/0.225)": {"efficiency_20d": 0.225, "fund_ret_accel_20_60": 0.225, **base_rest},
    }

    def strat(W):
        sc = score_factors_with_weights(fac, W, score_column="s")
        w = make_monthly_weights_v2(sc, px, uni, buffer_rank=35, volatility_target=0.18, **H.PORT)
        eq, _ = backtest_monthly_strategy(px, w, transaction_cost_bps=COST, rebalance_lambda=LAM)
        return H.to_ret(eq)

    rets = {k: strat(W) for k, W in weight_sets.items()}
    wb = H.equal_weight_basket(px); eqb, _ = backtest_monthly_strategy(px, wb, transaction_cost_bps=0.0)
    rets["等权ETF篮子"] = H.to_ret(eqb)
    rets["沪深300"] = market.loc[H.START:H.END].pct_change().dropna()

    rows = [H.metrics(r, k) for k, r in rets.items()]
    df = pd.DataFrame(rows)
    show = df.copy()
    for c in ["total", "ann", "vol", "mdd"]:
        show[c] = (show[c] * 100).map(lambda x: f"{x:+.1f}%")
    show["sharpe"] = show["sharpe"].map(lambda x: f"{x:.2f}")
    show["calmar"] = show["calmar"].map(lambda x: f"{x:.2f}")
    print("\n" + "=" * 72)
    print(f"拆分 combo 对比 (后复权市价, 含{COST:.0f}bps, lam={LAM}, 2018-2026)")
    print("=" * 72)
    print(show[["label", "total", "ann", "vol", "sharpe", "mdd", "calmar"]].to_string(index=False))

    # 净值图
    fig, ax = plt.subplots(figsize=(10, 5))
    style = {"现状(combo合成 0.45)": ("#e67e22", 1.6), "拆分·按ICIR": ("#c0392b", 2.2),
             "拆分·等权(0.225/0.225)": ("#8e44ad", 1.6), "等权ETF篮子": ("#2980b9", 1.2), "沪深300": ("#7f8c8d", 1.2)}
    for k, r in rets.items():
        nav = (1 + r).cumprod(); m = H.metrics(r)
        lab = f"{k} (年化{m['ann']:.1%} Sharpe{m['sharpe']:.2f})" if k not in ("等权ETF篮子", "沪深300") else k
        c, lw = style[k]
        ax.plot(nav.index, nav.values, label=lab, color=c, lw=lw)
    ax.set_title("拆分 combo→efficiency+accel 并重配权重 vs 现状（后复权市价，含成本，2018-2026）")
    ax.set_ylabel("净值（起点=1）"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(H.ROOT / "figures/09_split.png", dpi=130); plt.close(fig)
    print("\n图 -> figures/09_split.png")


if __name__ == "__main__":
    main()
