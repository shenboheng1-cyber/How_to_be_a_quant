# -*- coding: utf-8 -*-
"""
研究脚本 23 —— OOS 逐期因子选择（去选择偏差）+ 质量因子，冲 IR
================================================================
两处升级：
1) 因子选择改 OOS：每个调仓日只用【过去】数据贪心选正交核(最大化过去窗口的
   合成IC的ICIR)，再用选出的因子组当期 alpha → 无任何前视选择偏差。
   选择准则用"合成IC序列的ICIR"(等权合成IC的均值/标准差，正交因子能降方差→自然奖励正交)。
2) 候选池加入质量因子(GP/A、毛利率、低杠杆、周转率等)，看大盘指增 IR 能否上 0.5。

用法：/opt/anaconda3/bin/python research/23_oos_selection_quality.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, evaluate, backtest,
                      fundamentals, events, altdata, riskmodel, optimizer)
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
C = 0.003
WARM = 36                                                   # 选择预热月数
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

    # 预处理所有候选 + 各自月度IC
    names = []
    for name, fn in pool.items():
        try:
            z = preprocess.preprocess_factor(panel, fn(panel), industry_col="industry", do_neutralize=True)
            if z.notna().mean() < 0.3:
                continue
            panel["_z_" + name] = z.values
            names.append(name)
        except Exception:
            pass
    IC = pd.DataFrame({n: evaluate.compute_ic(panel, panel["_z_" + n]) for n in names}).sort_index()
    print(f"候选 {len(names)} 个(含质量)，预处理+IC完成 {time.time()-t0:.0f}s", flush=True)

    # ---- OOS 逐期贪心选择 ----
    months = sorted(panel["trddt"].unique())
    alpha = pd.Series(np.nan, index=panel.index)
    sel_log = {}
    for t in months:
        win = IC.loc[IC.index < t].dropna(axis=1, how="all")
        if len(win) < WARM:
            continue
        signs = np.sign(win.mean())
        aligned = win.mul(signs, axis=1).fillna(0.0)
        cands, selected, best = list(aligned.columns), [], -np.inf
        cur = pd.Series(0.0, index=aligned.index)
        while cands and len(selected) < 12:
            scs = {c: ((cur + aligned[c]).mean() / (cur + aligned[c]).std()) for c in cands}
            c = max(scs, key=scs.get)
            if scs[c] <= best + 1e-4 and len(selected) >= 3:
                break
            best = scs[c]; selected.append(c); cands.remove(c); cur = cur + aligned[c]
        idx = panel.index[panel["trddt"] == t]
        comp = sum(signs[c] * panel.loc[idx, "_z_" + c] for c in selected) / len(selected)
        alpha.loc[idx] = comp.values
        sel_log[t] = selected
    panel["alpha"] = alpha
    oos = panel[panel["alpha"].notna()]
    icr = evaluate.ic_summary(evaluate.compute_ic(oos, oos["alpha"]))
    print(f"\nOOS alpha 区间 {oos['trddt'].min().date()}~{oos['trddt'].max().date()}  "
          f"RankIC {icr['IC均值']}  ICIR {icr['ICIR']}  t {icr['t值']}  (对比 in-sample核 ICIR 1.223)")
    # 选择频率
    freq = pd.Series([f for v in sel_log.values() for f in v]).value_counts()
    avgk = np.mean([len(v) for v in sel_log.values()])
    print(f"平均核大小 {avgk:.1f}；最常被选(OOS)：")
    print("  " + ", ".join(f"{k}×{v}" for k, v in freq.head(14).items()))
    qual = [n for n in freq.index if n in ("f_gross_prof", "f_gross_margin", "f_low_lev", "f_asset_turn", "f_roe", "f_roa")]
    print(f"  质量因子入选情况：" + ", ".join(f"{q}×{freq[q]}" for q in qual) if qual else "  质量因子未入选")

    # ---- 部署 ----
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    pd.set_option("display.unicode.east_asian_width", True)

    # A) 市场中性 IM
    rows, prev = [], set()
    for dt, g in oos.dropna(subset=["fwd_ret"]).groupby("trddt"):
        top = g.nlargest(max(1, len(g) // 10), "alpha"); cur = set(top["stkcd"])
        to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows.append({"dt": dt, "g": top["fwd_ret"].mean(), "to": to}); prev = cur
    L = pd.DataFrame(rows).set_index("dt"); long_net = L["g"] - L["to"] * C
    i1000 = backtest.load_benchmark("000852", FREQ).reindex(L.index)
    d = pd.concat([long_net, i1000], axis=1).dropna(); beta = float(np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0])
    print("\nA) 市场中性 IM对冲(OOS alpha)：")
    for basis in [0.0, 0.03, 0.06]:
        m = backtest.metrics(long_net - beta * i1000 - beta * basis / 12, FREQ)
        print(f"   贴水{basis:.0%}: 净年化{m['年化']:.1%} 夏普{m['夏普']:.2f} 回撤{m['最大回撤']:.0%}")

    # B) 中证800 指增
    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")
    con = data.connect()
    bw = con.sql(f"""SELECT CAST(Enddt AS DATE) dt, Stkcd stkcd, TRY_CAST(Weight AS DOUBLE) w
                     FROM '{data.DATA_ROOT}/raw/IDX_Smprat.parquet' WHERE Indexcd='000906'""").df()
    con.close(); bw["dt"] = bw["dt"].astype("datetime64[ns]"); bw = bw.dropna()
    snaps = np.sort(bw["dt"].unique())
    TES = [0.03, 0.05]
    erows = []
    for dt in sorted(panel[panel["alpha"].notna()]["trddt"].unique()):
        snap = snaps[snaps <= np.datetime64(dt)]
        if not len(snap): continue
        cons = bw[bw["dt"] == snap[-1]][["stkcd", "w"]]
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
            w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, dd, active_cap=0.02, te=te, style_band=0.10)
            rec[f"opt{te}"] = float(np.nansum(w * fwd))
        erows.append(rec)
    R = pd.DataFrame(erows).set_index("dt")
    print("\nB) 中证800 指增(OOS alpha)：")
    for te in TES:
        ex = (R[f"opt{te}"] - R["bench"]).dropna()
        ir = ex.mean() / ex.std(ddof=1) * np.sqrt(12); nav = (1 + ex).cumprod()
        print(f"   TE={te:.0%}: 超额{ex.mean()*12:.2%} IR{ir:.2f} 跟踪误差{ex.std(ddof=1)*np.sqrt(12):.1%} "
              f"超额回撤{(nav/nav.cummax()-1).min():.1%}")
    print(f"\n完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
