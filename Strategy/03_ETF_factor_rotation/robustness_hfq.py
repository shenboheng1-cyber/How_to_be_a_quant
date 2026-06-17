"""后复权市价口径下的稳健性全套：消融归因 / 分年度+去最强年 / 滚动Sharpe /
block-bootstrap Sharpe CI / 成本sweep / 换手压力 / 等权篮子基准 / 冲击成本容量。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, backtest_monthly_strategy)
import hfq_common as H

OUT = H.ROOT / "outputs_robustness_hfq"
OUT.mkdir(exist_ok=True)
np.random.seed(20260617)


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, amt, prem = H.load_hfq()
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    market = H.hs300()
    print(f"HFQ池 {px.shape[1]} 只 | 区间 {H.START}~{H.END}")

    print("计算因子面板(一次)…")
    fac = compute_factor_panel(px)
    scored5 = score_factors_with_weights(fac, H.FW, score_column="risk_adjusted_score")   # 含风险项
    scored3 = score_factors_with_weights(fac, H.FW3, score_column="risk_adjusted_score")  # 纯3因子

    def run(scored, **kw):
        w = H.make_weights_flexible(scored, px, uni, market_nav=kw.pop("market", None), **kw)
        eq, eff = backtest_monthly_strategy(px, w, transaction_cost_bps=0.0)  # gross
        return H.to_ret(eq), eq, eff, w

    # ===== 1. 消融 / 归因 =====
    print("\n[1] 消融/归因…")
    variants = {
        "A0 完整策略(inv-vol+vol目标+弱市)": dict(scored=scored5, weighting="inv_vol",
            volatility_target=0.18, market="m", weak_market_exposure=0.60, **H.PORT),
        "A1 改等权(去inv-vol)":             dict(scored=scored5, weighting="equal",
            volatility_target=0.18, market="m", weak_market_exposure=0.60, **H.PORT),
        "A2 去弱市择时":                    dict(scored=scored5, weighting="inv_vol",
            volatility_target=0.18, market=None, **H.PORT),
        "A3 去波动目标":                    dict(scored=scored5, weighting="inv_vol",
            volatility_target=None, market="m", weak_market_exposure=0.60, **H.PORT),
        "A4 全程满仓(去择时+去vol目标)":     dict(scored=scored5, weighting="inv_vol",
            volatility_target=None, market=None, **H.PORT),
        "A5 去风险因子(纯3因子打分)":        dict(scored=scored3, weighting="inv_vol",
            volatility_target=0.18, market="m", weak_market_exposure=0.60, **H.PORT),
        "A6 纯选股alpha(3因子+等权+满仓)":   dict(scored=scored3, weighting="equal",
            volatility_target=None, market=None, **H.PORT),
        "A7 剔除防御主题(债/金/货币)":       dict(scored=scored5, weighting="inv_vol",
            volatility_target=0.18, market="m", weak_market_exposure=0.60,
            exclude_themes=H.DEFENSIVE_THEMES, **H.PORT),
    }
    abl_rows, base_ret, base_eq, base_eff, base_w = [], None, None, None, None
    for name, kw in variants.items():
        kw = dict(kw)
        if kw.get("market") == "m":
            kw["market"] = market
        r, eq, eff, w = run(**kw)
        abl_rows.append(H.metrics(r, name))
        if name.startswith("A0"):
            base_ret, base_eq, base_eff, base_w = r, eq, eff, w
    print(H.fmt(abl_rows)[["label", "total", "ann", "vol", "sharpe", "mdd", "calmar"]].to_string(index=False))
    H.fmt(abl_rows).to_csv(OUT / "ablation.csv", index=False, encoding="utf-8-sig")

    # 基准：等权篮子 + 沪深300
    print("\n[7] 更强基准：全市场等权 ETF 篮子…")
    wb = H.equal_weight_basket(px)
    eqb, _ = backtest_monthly_strategy(px, wb, transaction_cost_bps=0.0)
    basket_ret = H.to_ret(eqb)
    hs_ret = market.loc[H.START:H.END].pct_change().dropna()
    bench_rows = [H.metrics(base_ret, "完整策略(市价,gross)"),
                  H.metrics(basket_ret, "等权ETF篮子"),
                  H.metrics(hs_ret, "沪深300")]
    print(H.fmt(bench_rows)[["label", "total", "ann", "vol", "sharpe", "mdd", "calmar"]].to_string(index=False))

    # ===== 2. 分年度 + 去最强年 =====
    print("\n[2] 分年度收益 + 去最强年…")
    yr = pd.DataFrame({"strat": base_ret, "basket": basket_ret.reindex(base_ret.index).fillna(0),
                       "hs300": hs_ret.reindex(base_ret.index).fillna(0)})
    ann_tbl = []
    for y, g in yr.groupby(yr.index.year):
        row = {"year": y}
        for col in ["strat", "basket", "hs300"]:
            n = (1 + g[col]).prod()
            row[col] = n - 1
        nav = (1 + g["strat"]).cumprod()
        row["strat_mdd"] = float((nav / nav.cummax() - 1).min())
        row["strat_sharpe"] = (g["strat"].mean() / g["strat"].std(ddof=0) * np.sqrt(H.TD)) if g["strat"].std() else np.nan
        ann_tbl.append(row)
    at = pd.DataFrame(ann_tbl)
    show = at.copy()
    for c in ["strat", "basket", "hs300", "strat_mdd"]:
        show[c] = (show[c] * 100).map(lambda x: f"{x:+.1f}%")
    show["strat_sharpe"] = show["strat_sharpe"].map(lambda x: f"{x:.2f}")
    print(show.to_string(index=False))
    at.to_csv(OUT / "annual_returns.csv", index=False, encoding="utf-8-sig")

    best_y = int(at.loc[at["strat"].idxmax(), "year"])
    full = H.metrics(base_ret, "全期")
    ex = H.metrics(base_ret[base_ret.index.year != best_y], f"去最强年{best_y}")
    print(f"\n  全期: 年化 {full['ann']:+.1%} Sharpe {full['sharpe']:.2f}")
    print(f"  去最强年({best_y}, 当年 {at['strat'].max():+.1%}): 年化 {ex['ann']:+.1%} Sharpe {ex['sharpe']:.2f}")

    # ===== 3. 12个月滚动 Sharpe =====
    print("\n[3] 12个月滚动 Sharpe…")
    roll = base_ret.rolling(H.TD)
    rs = (roll.mean() / roll.std(ddof=0) * np.sqrt(H.TD)).dropna()
    print(f"  滚动Sharpe: 中位 {rs.median():.2f} | <0 占比 {(rs < 0).mean():.1%} | "
          f"最低 {rs.min():.2f}({rs.idxmin().date()}) | 最高 {rs.max():.2f}")
    rs.to_csv(OUT / "rolling_sharpe_12m.csv", encoding="utf-8-sig")

    # ===== 4. Block-bootstrap Sharpe CI =====
    print("\n[4] Block-bootstrap Sharpe 置信区间(按月分块,L=21,B=3000)…")
    r = base_ret.dropna().to_numpy()
    L, B, n = 21, 3000, len(base_ret.dropna())
    nb = int(np.ceil(n / L))
    starts_max = len(r) - L
    sh = np.empty(B)
    for b in range(B):
        st = np.random.randint(0, starts_max + 1, size=nb)
        samp = np.concatenate([r[s:s + L] for s in st])[:n]
        sd = samp.std(ddof=0)
        sh[b] = (samp.mean() / sd * np.sqrt(H.TD)) if sd else np.nan
    lo, med, hi = np.nanpercentile(sh, [2.5, 50, 97.5])
    print(f"  点估计 Sharpe {full['sharpe']:.2f} | bootstrap 中位 {med:.2f} | 95% CI [{lo:.2f}, {hi:.2f}] | "
          f"P(Sharpe>1) {np.nanmean(sh > 1):.1%}")
    pd.Series(sh, name="boot_sharpe").to_csv(OUT / "bootstrap_sharpe.csv", index=False, encoding="utf-8-sig")

    # ===== 5. 成本敏感性 sweep =====
    print("\n[5] 成本敏感性 sweep…")
    gross = base_eq.copy(); gross["date"] = pd.to_datetime(gross["date"])
    gross = gross.set_index("date").loc[H.START:H.END]
    g_ret, turn = gross["strategy_return"], gross["turnover"]
    sweep = []
    for bps in [0, 5, 10, 15, 20, 30, 50]:
        net = (g_ret - turn * bps / 1e4)
        m = H.metrics(net, f"{bps}bps")
        sweep.append({"cost_bps": bps, "ann": m["ann"], "sharpe": m["sharpe"], "mdd": m["mdd"]})
    sw = pd.DataFrame(sweep)
    sw_show = sw.copy()
    sw_show["ann"] = (sw_show["ann"] * 100).map(lambda x: f"{x:+.1f}%")
    sw_show["mdd"] = (sw_show["mdd"] * 100).map(lambda x: f"{x:.1f}%")
    sw_show["sharpe"] = sw_show["sharpe"].map(lambda x: f"{x:.2f}")
    print(sw_show.to_string(index=False))
    sw.to_csv(OUT / "cost_sweep.csv", index=False, encoding="utf-8-sig")

    # ===== 6. 换手压力 =====
    print("\n[6] 换手压力…")
    reb = turn[turn > 1e-9]
    ann_turn = turn.sum() / (len(turn) / H.TD)
    print(f"  调仓次数 {len(reb)} | 单次换手: 均 {reb.mean():.1%} 中位 {reb.median():.1%} "
          f"最大 {reb.max():.1%} | 年化双边换手 {ann_turn:.1f}x")

    # ===== 8. 冲击成本模型(平方根律) + 容量 =====
    print("\n[8] 冲击成本模型(平方根律) + 容量曲线…")
    rets = px.pct_change(fill_method=None)
    sigma = rets.rolling(60, min_periods=40).std()                    # 个券日波动(分数)
    adv = amt.reindex(columns=px.columns).rolling(20, min_periods=10).mean()  # 近20日均额(元)
    dW = base_eff.diff().abs().reindex(px.index).fillna(0.0)          # 逐日逐券权重变化
    idx = base_eff.index
    sigma_a = sigma.reindex(idx).reindex(columns=base_eff.columns)
    adv_a = adv.reindex(idx).reindex(columns=base_eff.columns)
    IMPACT_COEF = 0.5
    cap_rows = []
    for aum in [1e7, 5e7, 1e8, 5e8, 1e9, 3e9]:
        part = (aum * dW) / adv_a.replace(0, np.nan)                  # 参与率 Q/ADV
        impact_frac = IMPACT_COEF * sigma_a * np.sqrt(part.clip(lower=0))
        impact_cost_t = (dW * impact_frac).sum(axis=1).reindex(g_ret.index).fillna(0.0)
        net = g_ret - turn * 5 / 1e4 - impact_cost_t                  # 5bps佣金/价差 + 冲击
        m = H.metrics(net, f"AUM {aum/1e8:.1f}亿")
        cap_rows.append({"AUM": f"{aum/1e8:.1f}亿", "年化": m["ann"], "Sharpe": m["sharpe"],
                         "MDD": m["mdd"], "年化冲击拖累": float(impact_cost_t.sum() / (len(turn) / H.TD))})
    cap = pd.DataFrame(cap_rows)
    cap_show = cap.copy()
    cap_show["年化"] = (cap_show["年化"] * 100).map(lambda x: f"{x:+.1f}%")
    cap_show["MDD"] = (cap_show["MDD"] * 100).map(lambda x: f"{x:.1f}%")
    cap_show["年化冲击拖累"] = (cap_show["年化冲击拖累"] * 100).map(lambda x: f"-{x:.2f}%")
    cap_show["Sharpe"] = cap_show["Sharpe"].map(lambda x: f"{x:.2f}")
    print("  (平方根律 impact = 0.5·σ_daily·√(订单额/ADV); 5bps 佣金/价差固定)")
    print(cap_show.to_string(index=False))
    cap.to_csv(OUT / "impact_capacity.csv", index=False, encoding="utf-8-sig")

    print(f"\n输出 -> {OUT}")


if __name__ == "__main__":
    main()
