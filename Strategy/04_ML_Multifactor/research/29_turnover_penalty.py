# -*- coding: utf-8 -*-
"""
研究脚本 29_turnover_penalty —— 中证1000指增 + 【换手惩罚】杠杆
================================================================
基线 29 的优化器版换手 9.3x 偏高、吃成本。本脚本在目标函数加换手惩罚项
  max alpha·w - gamma·||w - w_prev||_1
逐月把上期权重 w_prev(按本期股票池 stkcd 对齐,缺失填0;首月=基准b)传入。
扫 gamma ∈ {0.0, 0.001, 0.005, 0.02},选【扣成本后净超额最高】者。
只跑 TE=3% 口径。

用法：/opt/anaconda3/bin/python research/29_turnover_penalty.py
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
GAMMAS = [0.0, 0.001, 0.005, 0.02]


def optimize_enhanced_to(alpha, b, X_ind, X_style, F, d, w_prev,
                         active_cap=0.02, te=0.05, style_band=0.10, gamma=0.0):
    """指数增强 + 换手惩罚。max alpha·w - gamma·||w - w_prev||_1
    s.t. Σw=1, 0≤w≤b+active_cap, 行业中性, 风格暴露受控, 跟踪误差预算。
    w_prev:(n,) 上期权重已对齐本期股票池(缺失填0)。"""
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
    if gamma > 0:
        obj = obj - gamma * cp.norm1(w - w_prev)
    prob = cp.Problem(cp.Maximize(obj), cons)
    for solver in (cp.CLARABEL, cp.SCS, cp.ECOS):
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None and not np.isnan(w.value).any():
                return np.clip(w.value, 0, None)
        except Exception:
            continue
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
    print(f"准备完成 {time.time()-t0:.0f}s，逐月优化(扫 gamma) ...", flush=True)

    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")

    rows = []
    wprev = {g: None for g in GAMMAS}    # 每个 gamma 维护自己的上期权重(Series stkcd->w)
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
        stk = m["stkcd"].values
        rec = {"dt": dt, "bench": float(np.nansum(b * fwd)), "i1000": i1000.get(dt, np.nan)}
        for g in GAMMAS:
            # 对齐上期权重到本期股票池;首月用基准 b
            if wprev[g] is None:
                wp = b.copy()
            else:
                wp = pd.Series(wprev[g]).reindex(stk).fillna(0.0).values
            w = optimize_enhanced_to(m["alpha"].values, b, Xind.values, Xs, F, d, wp,
                                     active_cap=0.02, te=TE, style_band=0.10, gamma=g)
            ws = pd.Series(w, index=stk)
            # 换手 = 0.5 * 与对齐后上期权重的 L1 距离(首月不计)
            to = 0.0 if wprev[g] is None else 0.5 * np.abs(w - wp).sum()
            rec[f"opt{g}"] = float(np.nansum(w * fwd)); rec[f"to{g}"] = to
            wprev[g] = ws
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("dt")

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 84, "\n中证1000指增 + 换手惩罚(TE=3%, OOS) —— gamma 扫描\n", "=" * 84, sep="")
    summary = []
    for g in GAMMAS:
        port = R[f"opt{g}"] - R[f"to{g}"] * C          # 扣成本后净组合
        exI = (port - R["i1000"]).dropna()              # 对真实中证1000
        navx = (1 + exI).cumprod()
        net_ex = exI.mean() * PPY
        ir = exI.mean() / exI.std(ddof=1) * np.sqrt(PPY)
        te_real = exI.std(ddof=1) * np.sqrt(PPY)
        ex_mdd = (navx / navx.cummax() - 1).min()
        turn = R[f"to{g}"].mean() * PPY
        # 毛超额(不扣成本)
        port_gross = R[f"opt{g}"]
        gross_ex = (port_gross - R["i1000"]).dropna().mean() * PPY
        summary.append(dict(gamma=g, net_ex=net_ex, gross_ex=gross_ex, ir=ir,
                            te=te_real, ex_mdd=ex_mdd, turn=turn))
        print(f"\n--- gamma={g} ---")
        print(f"  对真实中证1000: 毛超额{gross_ex:.2%}  净超额{net_ex:.2%}  IR{ir:.2f}  "
              f"跟踪误差{te_real:.2%}  超额回撤{ex_mdd:.2%}  年化换手{turn:.1f}x")

    S = pd.DataFrame(summary)
    best = S.loc[S["net_ex"].idxmax()]
    print("\n" + "=" * 84)
    print(f"最优 gamma = {best['gamma']}  (扣成本后净超额最高 = {best['net_ex']:.2%})")
    print(f"  净超额 {best['net_ex']:.2%}  IR {best['ir']:.2f}  跟踪误差 {best['te']:.2%}  "
          f"超额回撤 {best['ex_mdd']:.2%}  年化换手 {best['turn']:.1f}x")
    print(f"  基线(gamma=0)换手 {S.loc[S['gamma']==0.0,'turn'].iloc[0]:.1f}x → "
          f"最优换手 {best['turn']:.1f}x")
    print(f"\n完成 {time.time()-t0:.0f}s")

    # 存汇总
    S.to_csv("results/29_turnover_penalty_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
