"""策略 V2 落地：去弱市择时(保留 vol-target) + hysteresis 降换手扩容量。
对比 V1(完整) vs V2(不同 buffer)，看 Sharpe / 换手 / 1亿&5亿容量，选定最优 buffer 并落地权重。"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from etf_factor_strategy.engine import compute_factor_panel, score_factors_with_weights, backtest_monthly_strategy
import hfq_common as H

OUT = H.ROOT / "outputs_strategy_v2"
OUT.mkdir(exist_ok=True)


def evaluate(w, px, amt, label, lam=1.0):
    eq, eff = H.backtest_smoothed(px, w, lam=lam, cost_bps=0.0)  # gross
    g = eq.copy(); g["date"] = pd.to_datetime(g["date"]); g = g.set_index("date").loc[H.START:H.END]
    g_ret, turn = g["strategy_return"], g["turnover"]
    m = H.metrics(g_ret, label)
    net5 = H.metrics(g_ret - turn * 5 / 1e4)
    reb = turn[turn > 1e-9]
    cap = {aum: H.metrics(H.apply_impact(g_ret, turn, eff, px, amt, aum))["sharpe"]
           for aum in (1e8, 5e8)}
    return {"label": label, "ann": m["ann"], "sharpe_gross": m["sharpe"], "sharpe_5bps": net5["sharpe"],
            "mdd": m["mdd"], "turn_mo": reb.mean(), "turn_yr": turn.sum() / (len(turn) / H.TD),
            "sharpe_1亿": cap[1e8], "sharpe_5亿": cap[5e8]}, w, eff


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, amt, prem = H.load_hfq()
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    market = H.hs300()
    print(f"HFQ池 {px.shape[1]} 只；计算因子面板…")
    fac = compute_factor_panel(px)
    scored = score_factors_with_weights(fac, H.FW, score_column="risk_adjusted_score")

    rows = []
    # V1 参考：完整策略（inv-vol + vol-target + 弱市择时）
    w_v1 = H.make_weights_flexible(scored, px, uni, market_nav=market, weighting="inv_vol",
                                   volatility_target=0.18, weak_market_exposure=0.60, **H.PORT)
    r, _, _ = evaluate(w_v1, px, amt, "V1 完整(弱市+vol目标)")
    rows.append(r)
    # V2a：仅去弱市择时（保留 vol-target），无 hysteresis（= buffer 20）
    w_a = H.make_weights_flexible(scored, px, uni, market_nav=None, weighting="inv_vol",
                                  volatility_target=0.18, **H.PORT)
    r, _, _ = evaluate(w_a, px, amt, "V2a 去弱市(无buffer)")
    rows.append(r)
    # V2：去弱市 + hysteresis buffer=35（名次滞后带）
    w_sel = H.make_weights_v2(scored, px, uni, buffer_rank=35, volatility_target=0.18, **H.PORT)
    r, _, _ = evaluate(w_sel, px, amt, "V2 buffer35·月度")
    rows.append(r)

    # 换手杠杆 A：降低调仓频率（月度→季度）
    for k, tag in [(3, "季度")]:
        wk = H.resample_weights(w_sel, k)
        rows.append(evaluate(wk, px, amt, f"V2·{tag}调仓(k={k})")[0])
    # 换手杠杆 B：部分再平衡（每月只移动 lam 朝目标，信号不变旧但交易额减小）
    for lam in [0.6, 0.4]:
        rows.append(evaluate(w_sel, px, amt, f"V2·月度·lam={lam}", lam=lam)[0])

    # 选优：1亿净 Sharpe 最优者（月度 full / 季度 / 平滑 中选）
    pool = {"V2·月度·full": (w_sel, 1.0, 1),
            "V2·季度调仓(k=3)": (H.resample_weights(w_sel, 3), 1.0, 3),
            "V2·月度·lam=0.6": (w_sel, 0.6, 1),
            "V2·月度·lam=0.4": (w_sel, 0.4, 1)}
    rows.append(evaluate(w_sel, px, amt, "V2·月度·full")[0])
    best = None
    for lbl, (wv, lam, k) in pool.items():
        rr = next((x for x in rows if x["label"] == lbl), None)
        if rr and (best is None or rr["sharpe_1亿"] > best[0]["sharpe_1亿"]):
            best = (rr, wv, lam, k)

    df = pd.DataFrame(rows)
    show = df.copy()
    for c in ["ann", "mdd", "turn_mo"]:
        show[c] = (show[c] * 100).map(lambda x: f"{x:+.1f}%" if c != "mdd" else f"{x:.1f}%")
    for c in ["sharpe_gross", "sharpe_5bps", "sharpe_1亿", "sharpe_5亿"]:
        show[c] = show[c].map(lambda x: f"{x:.2f}")
    show["turn_yr"] = show["turn_yr"].map(lambda x: f"{x:.1f}x")
    print("\n" + "=" * 100)
    print("V1 vs V2  (gross/净指标皆市价口径; sharpe_1亿/5亿 = 平方根冲击模型后净 Sharpe)")
    print("=" * 100)
    print(show[["label", "ann", "sharpe_gross", "sharpe_5bps", "mdd", "turn_mo", "turn_yr",
                "sharpe_1亿", "sharpe_5亿"]].to_string(index=False))

    best_r, best_w, best_lam, best_k = best
    freq = {1: "月度", 3: "季度"}[best_k]
    print(f"\n选定 V2: buffer=35 + {freq}调仓 + lam={best_lam}（1亿容量净 Sharpe 最优 {best_r['sharpe_1亿']:.2f}）")
    print(f"  vs V1: 换手 {df.iloc[0]['turn_yr']:.1f}x→{best_r['turn_yr']:.1f}x/年 | "
          f"1亿净Sharpe {df.iloc[0]['sharpe_1亿']:.2f}→{best_r['sharpe_1亿']:.2f}, "
          f"5亿 {df.iloc[0]['sharpe_5亿']:.2f}→{best_r['sharpe_5亿']:.2f}")

    # 落地最优 V2 权重 + 摘要
    best_w.merge(uni, on="fund_code", how="left").to_csv(OUT / "v2_rebalance_weights.csv", index=False, encoding="utf-8-sig")
    df.to_csv(OUT / "v1_vs_v2.csv", index=False, encoding="utf-8-sig")
    (OUT / "v2_summary.json").write_text(json.dumps({
        "version": "v2", "basis": "close_hfq",
        "changes": ["去弱市择时(market_nav=None)", "保留 vol-target 0.18",
                    "hysteresis buffer_rank=35", f"调仓频率={freq}(每{best_k}月)", f"部分再平衡 lam={best_lam}"],
        "factor_weights": H.FW,
        "portfolio": {**H.PORT, "buffer_rank": 35, "volatility_target": 0.18,
                      "rebalance_every_months": best_k, "rebalance_lambda": best_lam},
        "metrics_vs_v1": {"v1": rows[0], "v2_selected": best_r},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出 -> {OUT}")


if __name__ == "__main__":
    main()
