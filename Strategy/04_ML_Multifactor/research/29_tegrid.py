# -*- coding: utf-8 -*-
"""
研究脚本 29_tegrid —— 中证1000指增：TE预算 × 个股主动权重上限 网格调参
================================================================
基线 29_csi1000_product.py 固定 te=0.03 / active_cap=0.02。
本脚本网格 te ∈ {0.02,0.03,0.04} × active_cap ∈ {0.015,0.025,0.04}，
每组合独立回测（独立追踪 wprev 以正确计算换手），按【对真实中证1000净IR】选优。
不改动 quantlib/ 任何共享文件。

用法：/opt/anaconda3/bin/python research/29_tegrid.py
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
    print(f"准备完成 {time.time()-t0:.0f}s，逐月优化网格 ...", flush=True)

    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")

    combos = [(te, cap) for te in TE_GRID for cap in CAP_GRID]
    rows = []
    wprev = {c: None for c in combos}          # 每组合独立追踪上期权重
    dts = sorted(panel[panel["alpha"].notna()]["trddt"].unique())
    for di, dt in enumerate(dts):
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
        for (te, cap) in combos:
            w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, d,
                                            active_cap=cap, te=te, style_band=0.10)
            ws = pd.Series(w, index=m["stkcd"].values)
            pv = wprev[(te, cap)]
            to = 0.0 if pv is None else 0.5 * ws.subtract(pv, fill_value=0).abs().sum()
            rec[f"opt_{te}_{cap}"] = float(np.nansum(w * fwd))
            rec[f"to_{te}_{cap}"] = to
            wprev[(te, cap)] = ws
        rows.append(rec)
        if (di + 1) % 12 == 0:
            print(f"  {dt.date()} 进度 {di+1}/{len(dts)}  累计 {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows).set_index("dt")

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 90, "\n中证1000指增 TE×active_cap 网格 (OOS, 扣换手) —— 对真实中证1000\n", "=" * 90, sep="")
    print(f"{'te':>5}{'cap':>7}{'净超额%':>9}{'净IR':>7}{'跟踪误差%':>11}{'超额回撤%':>11}{'换手x':>8}{'毛超额%':>9}")
    results = []
    for (te, cap) in combos:
        port_gross = R[f"opt_{te}_{cap}"]
        port = port_gross - R[f"to_{te}_{cap}"] * C
        exI = (port - R["i1000"]).dropna()                 # 净（扣成本）对真实中证1000
        exI_g = (port_gross - R["i1000"]).dropna()         # 毛
        navx = (1 + exI).cumprod()
        ann_ex = exI.mean() * PPY
        ir = exI.mean() / exI.std(ddof=1) * np.sqrt(PPY)
        te_real = exI.std(ddof=1) * np.sqrt(PPY)
        exdd = (navx / navx.cummax() - 1).min()
        turn = R[f"to_{te}_{cap}"].mean() * PPY
        ann_ex_g = exI_g.mean() * PPY
        results.append({"te": te, "cap": cap, "net_ex": ann_ex, "ir": ir, "te_real": te_real,
                        "exdd": exdd, "turn": turn, "gross_ex": ann_ex_g})
        print(f"{te:>5.2f}{cap:>7.3f}{ann_ex*100:>9.2f}{ir:>7.2f}{te_real*100:>11.2f}"
              f"{exdd*100:>11.2f}{turn:>8.1f}{ann_ex_g*100:>9.2f}")

    best = max(results, key=lambda r: r["ir"])
    print("\n" + "=" * 90)
    print(f"最优（净IR 最高）: te={best['te']}, active_cap={best['cap']}")
    print(f"  净超额 {best['net_ex']*100:.2f}%  净IR {best['ir']:.2f}  跟踪误差 {best['te_real']*100:.2f}%  "
          f"超额回撤 {best['exdd']*100:.2f}%  换手 {best['turn']:.1f}x  毛超额 {best['gross_ex']*100:.2f}%")
    print(f"\n基线(te=0.03,cap=0.02): 超额6.8% / IR1.00 / TE5% / 超额回撤-3.9% / 换手9.3x")
    print(f"完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
