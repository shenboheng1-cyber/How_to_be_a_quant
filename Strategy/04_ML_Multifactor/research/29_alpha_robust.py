# -*- coding: utf-8 -*-
"""
研究脚本 29_alpha_robust —— 中证1000指增 · 杠杆=【alpha 横截面收缩/稳健化】
================================================================
基线 29_csi1000_product 直接用 lgb 原始预测当 alpha。LGB 预测尾部
(min/max ≈ ±11σ) 可能过拟合。本脚本在每个调仓日对 lgb 做横截面稳健化,
对比三种 alpha 输入(TE=3% 口径):
  (0) raw   —— 原始 lgb (基线)
  (a) rank  —— 横截面排名分位 (rank/N - 0.5)
  (b) wz    —— zscore 后 winsorize 到 ±2.5
其余流程(基准/风险模型/优化器/换手成本)与 29 完全一致。
**不修改 quantlib/**,优化器直接复用 optimizer.optimize_enhanced。

用法：/opt/anaconda3/bin/python research/29_alpha_robust.py
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
TE = 0.03                      # 本实验只跑 TE=3% 口径
WINZ = 2.5                     # winsorize 阈值 (zscore 单位)


def robustify(alpha_raw):
    """对一组横截面 alpha (1d array) 返回三种稳健化版本 dict。
    NaN 已在调用前过滤。返回的 alpha 均做了标准化(0均值/单位方差),
    使三种方案在优化器目标尺度上可比 —— 优化器只关心 alpha 的横截面排序/相对大小。"""
    a = np.asarray(alpha_raw, dtype=float)
    n = len(a)
    out = {}
    # (0) raw —— 仅做标准化以对齐尺度
    out["raw"] = (a - a.mean()) / (a.std(ddof=0) + 1e-12)
    # (a) rank —— 横截面排名分位居中, 再标准化
    r = pd.Series(a).rank(method="average").values  # 1..n
    rk = r / (n + 1) - 0.5                           # ∈(-0.5,0.5)
    out["rank"] = (rk - rk.mean()) / (rk.std(ddof=0) + 1e-12)
    # (b) wz —— zscore 后 winsorize 到 ±WINZ, 再重标准化
    z = (a - a.mean()) / (a.std(ddof=0) + 1e-12)
    z = np.clip(z, -WINZ, WINZ)
    out["wz"] = (z - z.mean()) / (z.std(ddof=0) + 1e-12)
    return out


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
    print(f"准备完成 {time.time()-t0:.0f}s，逐月优化 ...", flush=True)

    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")

    VARIANTS = ["raw", "rank", "wz"]
    rows = {}
    wprev = {v: None for v in VARIANTS}
    out_rows = []
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
        alphas = robustify(m["alpha"].values)
        for v in VARIANTS:
            w = optimizer.optimize_enhanced(alphas[v], b, Xind.values, Xs, F, d,
                                            active_cap=0.02, te=TE, style_band=0.10)
            ws = pd.Series(w, index=m["stkcd"].values)
            to = 0.0 if wprev[v] is None else 0.5 * ws.subtract(wprev[v], fill_value=0).abs().sum()
            rec[f"{v}"] = float(np.nansum(w * fwd)); rec[f"to_{v}"] = to
            wprev[v] = ws
        out_rows.append(rec)
    R = pd.DataFrame(out_rows).set_index("dt")

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 84, "\n中证1000指增 · alpha 横截面稳健化对比 (TE=3%, OOS, 扣换手) — 对真实中证1000\n", "=" * 84, sep="")
    summary = {}
    for v in VARIANTS:
        port = R[v] - R[f"to_{v}"] * C
        port_gross = R[v]
        exI = (port - R["i1000"]).dropna()
        exI_gross = (port_gross - R["i1000"]).dropna()
        navx = (1 + exI).cumprod()
        ann_ex = exI.mean() * PPY
        ann_ex_gross = exI_gross.mean() * PPY
        ir = exI.mean() / exI.std(ddof=1) * np.sqrt(PPY)
        te_real = exI.std(ddof=1) * np.sqrt(PPY)
        exdd = (navx / navx.cummax() - 1).min()
        turn = R[f"to_{v}"].mean() * PPY
        win = (exI > 0).mean()
        summary[v] = dict(ex=ann_ex, ex_gross=ann_ex_gross, ir=ir, te=te_real,
                          exdd=exdd, turn=turn, win=win)
        label = {"raw": "(0) raw 原始lgb [基线]", "rank": "(a) rank 排名分位", "wz": "(b) wz winsor±2.5"}[v]
        print(f"\n--- {label} ---")
        print(f"  净超额(扣成本){ann_ex:.2%}  毛超额{ann_ex_gross:.2%}  IR{ir:.2f}  "
              f"跟踪误差{te_real:.2%}  超额回撤{exdd:.2%}  年化换手{turn:.1f}x  月胜率{win:.0%}")

    # 选优(按净超额IR)
    best = max(VARIANTS, key=lambda v: summary[v]["ir"])
    print("\n" + "=" * 84)
    print(f"最优方案: {best}  (按 IR)")
    print("=" * 84)
    print(f"完成 {time.time()-t0:.0f}s")
    return summary, best


if __name__ == "__main__":
    main()
