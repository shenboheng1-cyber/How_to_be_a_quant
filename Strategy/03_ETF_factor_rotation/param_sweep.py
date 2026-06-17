"""可调参数敏感性：在 V2 基线上每次只改一个参数，看 Sharpe/回撤/换手/容量(1亿)。
基线：top_n20, max_per_theme3, max_weight0.12, vol_target0.18, buffer35, lam0.4, 5bps，后复权市价。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, backtest_monthly_strategy,
    make_monthly_weights_v2, FACTOR_WEIGHTS_V2)
import hfq_common as H

OUT = H.ROOT / "outputs_param_sweep"
OUT.mkdir(exist_ok=True)
BASE = dict(top_n=20, max_per_theme=3, max_weight=0.12, buffer_rank=35, volatility_target=0.18)
LAM = 0.4


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, amt, _ = H.load_hfq()
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    fac = compute_factor_panel(px)
    scored = score_factors_with_weights(fac, FACTOR_WEIGHTS_V2, score_column="risk_adjusted_score")

    def evalp(weight_kwargs, lam):
        w = make_monthly_weights_v2(scored, px, uni, **weight_kwargs)
        eq, eff = backtest_monthly_strategy(px, w, transaction_cost_bps=0.0, rebalance_lambda=lam)
        g = eq.copy(); g["date"] = pd.to_datetime(g["date"]); g = g.set_index("date").loc[H.START:H.END]
        gr, tn = g["strategy_return"], g["turnover"]
        net5 = H.metrics(gr - tn * 5 / 1e4)
        cap1 = H.metrics(H.apply_impact(gr, tn, eff, px, amt, 1e8))["sharpe"]
        return {"年化": net5["ann"], "Sharpe净5bps": net5["sharpe"], "MDD": net5["mdd"],
                "年化换手": tn.sum() / (len(tn) / H.TD), "Sharpe@1亿": cap1}

    rows = []
    def add(group, label, kw=None, lam=LAM):
        wk = {**BASE, **(kw or {})}
        r = evalp(wk, lam)
        rows.append({"参数组": group, "取值": label, **r})

    add("基线", "V2 默认")
    for v in [0.3, 0.4, 0.6, 1.0]:
        add("lam(部分再平衡)", f"lam={v}", lam=v)
    for v in [25, 35, 50, 999]:
        add("buffer_rank(滞后带)", f"buffer={v}", {"buffer_rank": v})
    for v in [15, 20, 25, 30]:
        add("top_n(持仓数)", f"top_n={v}", {"top_n": v})
    for v in [2, 3, 4]:
        add("max_per_theme(同主题上限)", f"≤{v}", {"max_per_theme": v})
    for v in [0.10, 0.12, 0.15, 0.20]:
        add("max_weight(单票上限)", f"{v:.0%}", {"max_weight": v})
    for v, lab in [(0.15, "0.15"), (0.18, "0.18"), (0.25, "0.25"), (None, "关闭")]:
        add("volatility_target(波动目标)", lab, {"volatility_target": v})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "param_sweep.csv", index=False, encoding="utf-8-sig")
    show = df.copy()
    show["年化"] = (show["年化"] * 100).map(lambda x: f"{x:+.1f}%")
    show["MDD"] = (show["MDD"] * 100).map(lambda x: f"{x:.1f}%")
    show["年化换手"] = show["年化换手"].map(lambda x: f"{x:.1f}x")
    for c in ["Sharpe净5bps", "Sharpe@1亿"]:
        show[c] = show[c].map(lambda x: f"{x:.2f}")
    print(show.to_string(index=False))
    print(f"\n输出 -> {OUT}")


if __name__ == "__main__":
    main()
