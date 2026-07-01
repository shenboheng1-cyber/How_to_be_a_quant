# -*- coding: utf-8 -*-
"""Timing probe: how long is one optimize_enhanced solve, and the data-prep."""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, backtest, altdata, riskmodel)
from quantlib.factors import classic
import cvxpy as cp

FREQ = "M"
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]
LO, HI = 800, 1800

def pr(*a):
    print(*a, flush=True)

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
pr(f"data prep done {time.time()-t0:.0f}s")

tf = time.time()
f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")
pr(f"riskmodel done {time.time()-tf:.0f}s")

dts = sorted(panel[panel["alpha"].notna()]["trddt"].unique())
pr(f"n months with alpha = {len(dts)}")

# time one solve at the last available date
dt = dts[len(dts)//2]
m = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna() &
          (panel["caprank"] > LO) & (panel["caprank"] <= HI)].copy()
pr(f"sample month {pd.Timestamp(dt).date()}: n stocks = {len(m)}")
b = (m["total_mktcap"] / m["total_mktcap"].sum()).values
Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
Xs = m[style_cols].fillna(0.0).values
d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
alpha = m["alpha"].values
n = len(alpha)

def solve_once(active_cap, te, solver):
    X = np.hstack([Xind.values, Xs])
    w = cp.Variable(n)
    a = w - b
    afe = X.T @ a
    te2 = cp.quad_form(afe, cp.psd_wrap(F)) + cp.sum(cp.multiply(np.maximum(d,1e-8), cp.square(a)))
    cons = [cp.sum(w)==1, w>=0, w<=b+active_cap, Xind.values.T@a==0,
            cp.abs(Xs.T@a)<=0.10, te2<=te**2]
    prob = cp.Problem(cp.Maximize(alpha@w), cons)
    ts = time.time()
    prob.solve(solver=solver, verbose=False)
    el = time.time()-ts
    return el, prob.status, (None if w.value is None else float(alpha@w.value))

for solver in (cp.CLARABEL, cp.SCS):
    el, st, obj = solve_once(0.025, 0.03, solver)
    pr(f"  solver={solver} time={el:.2f}s status={st} obj={obj}")

pr(f"total {time.time()-t0:.0f}s")
