# -*- coding: utf-8 -*-
"""
研究脚本 29_te_cap_grid —— 中证1000指增：TE预算 × 个股主动权重上限 网格调参
================================================================
基线 29_csi1000_product.py 的 active_cap=0.02 / te=0.03 固定。本脚本做网格：
  te ∈ {0.02,0.03,0.04} × active_cap ∈ {0.015,0.025,0.04}  共9组，
逐月对每组调用一次优化器，扣换手后按【对真实中证1000的净IR】择优。

不修改 quantlib/ 任何文件。optimize_enhanced 直接复用 quantlib.optimizer。

用法：/opt/anaconda3/bin/python research/29_te_cap_grid.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, backtest, fundamentals, altdata,
                      riskmodel, optimizer)
from quantlib.factors import classic

FREQ, C, PPY = "M", 0.003, 12
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]
LO, HI = 800, 1800

TE_GRID = [0.02, 0.03, 0.04]
CAP_GRID = [0.015, 0.025, 0.04]
COMBOS = [(te, cap) for te in TE_GRID for cap in CAP_GRID]


def main():
    t0 = time.time()
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    pred = pd.read_parquet("results/lgb_oos_pred.parquet"); pred["trddt"] = pred["trddt"].astype("datetime64[ns]")
    panel = panel.merge(pred, on=["stkcd", "trddt"], how="left").rename(columns={"lgb": "alpha"})
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    panel["caprank"] = panel.groupby("trddt")["total_mktcap"].rank(ascending=False, method="first")
    i1000 = backtest.load_benchmark("000852", FREQ)
    print(f"准备完成 {time.time()-t0:.0f}s，逐月优化 9 组 ...", flush=True)

    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")

    rows = []
    wprev = {c: None for c in COMBOS}          # 每组各自的上期权重
    for dt in sorted(panel[panel["alpha"].notna()]["trddt"].unique()):
        m = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna() &
                  (panel["caprank"] > LO) & (panel["caprank"] <= HI)].copy()
        if len(m) < 200: continue
        b = (m["total_mktcap"] / m["total_mktcap"].sum()).values
        Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values
        rec = {"dt": dt, "bench": float(np.nansum(b * fwd)), "i1000": i1000.get(dt, np.nan)}
        for (te, cap) in COMBOS:
            w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, d,
                                            active_cap=cap, te=te, style_band=0.10)
            ws = pd.Series(w, index=m["stkcd"].values)
            wp = wprev[(te, cap)]
            to = 0.0 if wp is None else 0.5 * ws.subtract(wp, fill_value=0).abs().sum()
            rec[f"opt_{te}_{cap}"] = float(np.nansum(w * fwd)); rec[f"to_{te}_{cap}"] = to
            wprev[(te, cap)] = ws
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("dt")

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 90, "\n中证1000指增 TE×active_cap 网格(OOS,扣0.3%换手) —— 对真实中证1000\n", "=" * 90, sep="")
    print(f"{'te':>6}{'cap':>7}{'超额%':>9}{'净IR':>8}{'TE%':>8}{'超额MDD%':>11}{'换手x':>8}")
    results_tbl = []
    for (te, cap) in COMBOS:
        port = R[f"opt_{te}_{cap}"] - R[f"to_{te}_{cap}"] * C    # 已扣换手成本
        exI = (port - R["i1000"]).dropna()                      # 对真实中证1000
        navx = (1 + exI).cumprod()
        ex_ann = exI.mean() * PPY
        ir = exI.mean() / exI.std(ddof=1) * np.sqrt(PPY)
        te_real = exI.std(ddof=1) * np.sqrt(PPY)
        mdd = (navx / navx.cummax() - 1).min()
        turn = R[f"to_{te}_{cap}"].mean() * PPY
        results_tbl.append({"te": te, "cap": cap, "ex": ex_ann, "ir": ir,
                            "te_real": te_real, "mdd": mdd, "turn": turn})
        print(f"{te:>6.2f}{cap:>7.3f}{ex_ann*100:>9.2f}{ir:>8.2f}{te_real*100:>8.2f}"
              f"{mdd*100:>11.2f}{turn:>8.1f}")

    best = max(results_tbl, key=lambda r: r["ir"])              # 按净IR择优
    print("\n" + "=" * 90)
    print(f"最优(按对真实中证1000净IR): te={best['te']} active_cap={best['cap']}")
    print(f"  超额{best['ex']:.2%}  净IR{best['ir']:.2f}  跟踪误差{best['te_real']:.2%}  "
          f"超额回撤{best['mdd']:.2%}  年化换手{best['turn']:.1f}x")
    print(f"\n完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
