# -*- coding: utf-8 -*-
"""
研究脚本 29_turnpen —— 中证1000 指增 + 【换手惩罚】杠杆
================================================================
复制 29_csi1000_product，把 optimize_enhanced 内联改造，目标函数加
  - gamma * ||w - w_prev||_1
逐月把上期权重 w_prev 按本期股票池对齐(缺失填0,首月=基准b)传入。
扫 gamma ∈ {0.0, 0.001, 0.005, 0.02}，选【扣成本后净超额最高】者。
TE=3% 口径报告。

用法：/opt/anaconda3/bin/python research/29_turnpen.py
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
TE = 0.03                                    # 本任务只看 TE=3%
GAMMAS = [0.0, 0.001, 0.005, 0.02]           # 换手惩罚强度扫描


def optimize_enhanced_tp(alpha, b, X_ind, X_style, F, d, w_prev,
                         active_cap=0.02, te=0.05, style_band=0.10, gamma=0.0):
    """指数增强 + 换手惩罚。
    max  alpha·w - gamma·||w - w_prev||_1
    s.t. Σw=1, 0≤w≤b+active_cap, 行业中性 X_ind·(w-b)=0,
         |X_style·(w-b)|≤band, (w-b)ᵀΣ(w-b)≤te².
    w_prev:(n,) 上期权重已按本期 universe 对齐(缺失填0)。"""
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
    obj = alpha @ w
    if gamma > 0 and w_prev is not None:
        obj = obj - gamma * cp.norm1(w - w_prev)
    prob = cp.Problem(cp.Maximize(obj), cons)
    # CLARABEL ~0.6s/解；SCS 慢 30x 且常 inaccurate，故只用 CLARABEL，失败回退基准
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if w.value is not None and not np.isnan(w.value).any():
            return np.clip(w.value, 0, None)
    except Exception:
        pass
    return b


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

    # 每个 gamma 维护自己的 wprev（按 stkcd 索引的 Series）
    rows = []
    wprev = {g: None for g in GAMMAS}
    dts = sorted(panel[panel["alpha"].notna()]["trddt"].unique())
    for dt in dts:
        m = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna() &
                  (panel["caprank"] > LO) & (panel["caprank"] <= HI)].copy()
        if len(m) < 200: continue
        b = (m["total_mktcap"] / m["total_mktcap"].sum()).values
        Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values
        stk = m["stkcd"].values
        rec = {"dt": dt, "bench": float(np.nansum(b * fwd)), "i1000": i1000.get(dt, np.nan)}
        for g in GAMMAS:
            # 把上期权重按本期 universe 对齐：首月用基准 b，否则 reindex 缺失填0
            if wprev[g] is None:
                wp = b.copy()
            else:
                wp = wprev[g].reindex(stk).fillna(0.0).values
            w = optimize_enhanced_tp(m["alpha"].values, b, Xind.values, Xs, F, d, wp,
                                     active_cap=0.02, te=TE, style_band=0.10, gamma=g)
            ws = pd.Series(w, index=stk)
            # 换手按 上期权重(对齐后)与本期权重的差：0.5*Σ|w - wp|
            to = 0.5 * np.abs(w - wp).sum() if wprev[g] is not None else 0.0
            rec[f"opt{g}"] = float(np.nansum(w * fwd)); rec[f"to{g}"] = to
            wprev[g] = ws
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("dt")

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 84, "\n中证1000 指增 + 换手惩罚 (OOS, TE=3%, 扫 gamma) —— 对真实中证1000\n", "=" * 84, sep="")
    summary = {}
    for g in GAMMAS:
        port = R[f"opt{g}"] - R[f"to{g}"] * C            # 扣成本后净组合收益
        gross = R[f"opt{g}"]
        exI = (port - R["i1000"]).dropna()               # 净超额(对真实中证1000)
        exI_gross = (gross - R["i1000"]).dropna()        # 毛超额
        navx = (1 + exI).cumprod()
        net_exc = exI.mean() * PPY
        ir = exI.mean() / exI.std(ddof=1) * np.sqrt(PPY)
        te_real = exI.std(ddof=1) * np.sqrt(PPY)
        mdd = (navx / navx.cummax() - 1).min()
        turn = R[f"to{g}"].mean() * PPY
        summary[g] = dict(net_exc=net_exc, gross_exc=exI_gross.mean() * PPY, ir=ir,
                          te=te_real, mdd=mdd, turn=turn, win=(exI > 0).mean())
        print(f"\n--- gamma={g} ---")
        print(f"  净超额{net_exc:.2%}  毛超额{exI_gross.mean()*PPY:.2%}  IR{ir:.2f}  "
              f"跟踪误差{te_real:.2%}  超额回撤{mdd:.2%}  年化换手{turn:.1f}x  月胜率{(exI>0).mean():.0%}")

    best_g = max(GAMMAS, key=lambda g: summary[g]["net_exc"])
    s = summary[best_g]
    print("\n" + "=" * 84)
    print(f"最优 gamma = {best_g}  (按扣成本后净超额最高)")
    print(f"  净超额 {s['net_exc']:.2%} | 毛超额 {s['gross_exc']:.2%} | IR {s['ir']:.2f} | "
          f"跟踪误差 {s['te']:.2%} | 超额回撤 {s['mdd']:.2%} | 年化换手 {s['turn']:.1f}x")
    base = summary[0.0]
    print(f"  对比 gamma=0 基线: 净超额 {base['net_exc']:.2%} -> {s['net_exc']:.2%}, "
          f"换手 {base['turn']:.1f}x -> {s['turn']:.1f}x, IR {base['ir']:.2f} -> {s['ir']:.2f}")
    print(f"\n完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
