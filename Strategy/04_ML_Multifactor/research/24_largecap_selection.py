# -*- coding: utf-8 -*-
"""
研究脚本 24 —— 大盘专属选因子的中证800指增（OOS，冲 IR）
================================================================
洞察：全样本/全市场选出的因子偏小盘(max_ret/专利/调研)，在中证800大盘未必最优。
做法：用【中证800成分内】的 trailing IC 做 OOS 选择 → 大盘专属 alpha → 指增。
看 IR 能否比全局选择的 0.27 提升（质量因子在大盘应更突出）。

用法：/opt/anaconda3/bin/python research/24_largecap_selection.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, evaluate, backtest,
                      fundamentals, events, altdata, riskmodel, optimizer)
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
WARM = 36
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]


def build_pool(panel):
    pool = {}
    for name, (fn, _) in classic.REGISTRY.items():
        pool[name] = fn
    for mod in (fundamentals, events, altdata):
        for name, (fn, _) in getattr(mod, "REGISTRY", {}).items():
            pool[name] = fn
    try:
        from quantlib.factors import behavioral
        for name, (fn, _) in behavioral.REGISTRY.items():
            pool["b_" + name] = fn
    except Exception:
        pass
    return pool


def main():
    t0 = time.time()
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel = fundamentals.attach(panel); panel = events.attach_events(panel); panel = altdata.attach_altfactors(panel)
    pool = build_pool(panel)
    names = []
    for name, fn in pool.items():
        try:
            z = preprocess.preprocess_factor(panel, fn(panel), industry_col="industry", do_neutralize=True)
            if z.notna().mean() < 0.3:
                continue
            panel["_z_" + name] = z.values; names.append(name)
        except Exception:
            pass

    # 中证800 成分 as-of 标记
    con = data.connect()
    bw = con.sql(f"""SELECT CAST(Enddt AS DATE) dt, Stkcd stkcd, TRY_CAST(Weight AS DOUBLE) w
                     FROM '{data.DATA_ROOT}/raw/IDX_Smprat.parquet' WHERE Indexcd='000906'""").df()
    con.close(); bw["dt"] = bw["dt"].astype("datetime64[ns]"); bw = bw.dropna()
    snaps = np.sort(bw["dt"].unique())
    panel["in800"] = False
    for t in sorted(panel["trddt"].unique()):
        s = snaps[snaps <= np.datetime64(t)]
        if not len(s): continue
        cons = set(bw[bw["dt"] == s[-1]]["stkcd"])
        panel.loc[(panel["trddt"] == t) & panel["stkcd"].isin(cons), "in800"] = True
    big = panel[panel["in800"]]
    # 大盘专属 IC 矩阵
    IC = pd.DataFrame({n: evaluate.compute_ic(big, big["_z_" + n]) for n in names}).sort_index()
    print(f"候选 {len(names)}，中证800覆盖 {panel['in800'].mean():.0%}，大盘IC完成 {time.time()-t0:.0f}s", flush=True)

    # OOS 逐期贪心选择（用大盘 IC），形成大盘 alpha（只在 in800 上）
    alpha = pd.Series(np.nan, index=panel.index)
    sel_log = {}
    for t in sorted(panel["trddt"].unique()):
        win = IC.loc[IC.index < t].dropna(axis=1, how="all")
        if len(win) < WARM: continue
        signs = np.sign(win.mean()); aligned = win.mul(signs, axis=1).fillna(0.0)
        cands, selected, best = list(aligned.columns), [], -np.inf
        cur = pd.Series(0.0, index=aligned.index)
        while cands and len(selected) < 12:
            scs = {c: ((cur + aligned[c]).mean() / (cur + aligned[c]).std()) for c in cands}
            c = max(scs, key=scs.get)
            if scs[c] <= best + 1e-4 and len(selected) >= 3: break
            best = scs[c]; selected.append(c); cands.remove(c); cur = cur + aligned[c]
        idx = panel.index[(panel["trddt"] == t) & panel["in800"]]
        if not len(idx): continue
        comp = sum(signs[c] * panel.loc[idx, "_z_" + c] for c in selected) / len(selected)
        alpha.loc[idx] = comp.values; sel_log[t] = selected
    panel["alpha"] = alpha
    freq = pd.Series([f for v in sel_log.values() for f in v]).value_counts()
    print("大盘OOS最常被选：" + ", ".join(f"{k}×{v}" for k, v in freq.head(12).items()))

    # 指增（中证800）
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")
    TES = [0.02, 0.03, 0.05]
    erows = []
    for dt in sorted(panel[panel["alpha"].notna()]["trddt"].unique()):
        s = snaps[snaps <= np.datetime64(dt)]
        if not len(s): continue
        cons = bw[bw["dt"] == s[-1]][["stkcd", "w"]]
        g = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna()]
        m = g.merge(cons, on="stkcd", how="inner")
        if len(m) < 200: continue
        b = (m["w"] / m["w"].sum()).values
        Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        dd = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values; rec = {"dt": dt, "bench": float(np.nansum(b * fwd))}
        for te in TES:
            w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, dd, active_cap=0.025, te=te, style_band=0.10)
            rec[f"opt{te}"] = float(np.nansum(w * fwd))
        erows.append(rec)
    R = pd.DataFrame(erows).set_index("dt")
    print("\n中证800指增 — 大盘专属OOS alpha（对比全局选择 IR0.27）：")
    for te in TES:
        ex = (R[f"opt{te}"] - R["bench"]).dropna()
        ir = ex.mean() / ex.std(ddof=1) * np.sqrt(12); nav = (1 + ex).cumprod()
        print(f"   TE={te:.0%}: 超额{ex.mean()*12:.2%} IR{ir:.2f} 跟踪误差{ex.std(ddof=1)*np.sqrt(12):.1%} "
              f"超额回撤{(nav/nav.cummax()-1).min():.1%}")
    print(f"\n完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
