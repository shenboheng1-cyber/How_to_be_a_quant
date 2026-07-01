# -*- coding: utf-8 -*-
"""
研究脚本 29_alphashrink —— 中证1000指增：alpha 横截面收缩/稳健化
================================================================
基线 29 直接用 lgb OOS 预测当 alpha。LGB 尾部可能过拟合，本脚本在每个
调仓日对 lgb 做横截面稳健化，对比三种 alpha 喂给同一优化器：
  raw  : 原始 lgb（基线对照）
  rank : 横截面排名分位 rank/N（再居中到 0 均值，纯序信息）
  zwin : 横截面 zscore 后 winsorize 到 ±2.5

优化器 optimize_enhanced 内联在本脚本（不改 quantlib/）。
报告 TE=3% 口径对真实中证1000的：超额% / IR / 跟踪误差% / 超额回撤% / 年化换手x / 净超额%。

用法：/opt/anaconda3/bin/python research/29_alphashrink.py
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
TE = 0.03
VARIANTS = ["raw", "rank", "zwin"]


def optimize_enhanced(alpha, b, X_ind, X_style, F, d,
                      active_cap=0.02, te=0.05, style_band=0.10):
    """内联副本（与 quantlib.optimizer.optimize_enhanced 同）。"""
    import cvxpy as cp
    n = len(alpha)
    X = np.hstack([X_ind, X_style])
    w = cp.Variable(n)
    a = w - b
    afe = X.T @ a
    te2 = cp.quad_form(afe, cp.psd_wrap(F)) + cp.sum(cp.multiply(np.maximum(d, 1e-8), cp.square(a)))
    cons = [cp.sum(w) == 1, w >= 0, w <= b + active_cap,
            X_ind.T @ a == 0,
            cp.abs(X_style.T @ a) <= style_band,
            te2 <= te ** 2]
    prob = cp.Problem(cp.Maximize(alpha @ w), cons)
    for solver in (cp.CLARABEL, cp.SCS, cp.ECOS):
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None and not np.isnan(w.value).any():
                return np.clip(w.value, 0, None)
        except Exception:
            continue
    return b


def shrink_alpha(raw, kind):
    """横截面稳健化。raw:(n,) 当期 lgb 值。返回同长度变换后 alpha。"""
    raw = np.asarray(raw, dtype=float)
    n = len(raw)
    if kind == "raw":
        return raw
    if kind == "rank":
        # rank/N 分位，居中到 0 均值（纯序信息，去掉幅度）
        r = pd.Series(raw).rank(method="average").values
        return r / n - 0.5
    if kind == "zwin":
        mu, sd = np.mean(raw), np.std(raw)
        if sd < 1e-12:
            return np.zeros(n)
        z = (raw - mu) / sd
        return np.clip(z, -2.5, 2.5)
    raise ValueError(kind)


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
    print(f"准备完成 {time.time()-t0:.0f}s，逐月优化（{len(VARIANTS)} 个 alpha 变体）...", flush=True)

    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")

    rows = []
    wprev = {v: None for v in VARIANTS}
    for dt in sorted(panel[panel["alpha"].notna()]["trddt"].unique()):
        m = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna() &
                  (panel["caprank"] > LO) & (panel["caprank"] <= HI)].copy()
        if len(m) < 200: continue
        b = (m["total_mktcap"] / m["total_mktcap"].sum()).values
        Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        dd = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values
        rawalpha = m["alpha"].values
        rec = {"dt": dt, "bench": float(np.nansum(b * fwd)), "i1000": i1000.get(dt, np.nan)}
        for v in VARIANTS:
            a = shrink_alpha(rawalpha, v)
            w = optimize_enhanced(a, b, Xind.values, Xs, F, dd,
                                  active_cap=0.02, te=TE, style_band=0.10)
            ws = pd.Series(w, index=m["stkcd"].values)
            to = 0.0 if wprev[v] is None else 0.5 * ws.subtract(wprev[v], fill_value=0).abs().sum()
            rec[f"ret_{v}"] = float(np.nansum(w * fwd)); rec[f"to_{v}"] = to
            wprev[v] = ws
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("dt")

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 88, "\n中证1000 指增 alpha 横截面稳健化对比（OOS, TE=3%, 扣换手） —— 对真实中证1000\n", "=" * 88, sep="")
    summary = {}
    for v in VARIANTS:
        gross = R[f"ret_{v}"]
        port = gross - R[f"to_{v}"] * C
        exI = (port - R["i1000"]).dropna()
        exI_gross = (gross - R["i1000"]).dropna()
        navx = (1 + exI).cumprod()
        excess = exI.mean() * PPY
        excess_gross = exI_gross.mean() * PPY
        te_real = exI.std(ddof=1) * np.sqrt(PPY)
        ir = exI.mean() / exI.std(ddof=1) * np.sqrt(PPY)
        mdd = (navx / navx.cummax() - 1).min()
        to_ann = R[f"to_{v}"].mean() * PPY
        summary[v] = dict(excess_gross=excess_gross, excess=excess, ir=ir, te=te_real,
                          mdd=mdd, to=to_ann, winrate=(exI > 0).mean())
        print(f"\n--- alpha = {v} ---")
        print(f"  对真实中证1000: 毛超额{excess_gross:.1%}  净超额{excess:.1%}  IR{ir:.2f}  "
              f"跟踪误差{te_real:.1%}  超额回撤{mdd:.1%}  月胜率{(exI>0).mean():.0%}  年化换手{to_ann:.1f}x")
    # 选净超额最高者
    best = max(summary, key=lambda v: summary[v]["excess"])
    print("\n" + "=" * 88)
    print(f"最优 alpha 稳健化方案: {best}")
    s = summary[best]
    print(f"  净超额{s['excess']:.2%}  IR{s['ir']:.3f}  TE{s['te']:.2%}  超额回撤{s['mdd']:.2%}  换手{s['to']:.2f}x")
    print(f"\n完成 {time.time()-t0:.0f}s")
    return summary, best


if __name__ == "__main__":
    main()
