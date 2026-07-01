# -*- coding: utf-8 -*-
"""
研究脚本 25 —— 最佳几版策略的完整指标（OOS 诚实口径）
================================================================
用 research/23 的 OOS 逐期选择 alpha(无选择偏差)，重算 research/22 的表：
  V1 多头·全A / V2 市场中性 IM对冲贴水3% / V3 中证800指增 TE=3%
指标：年化/波动/夏普/最大回撤/卡玛/超额/胜率，并存净值。

用法：/opt/anaconda3/bin/python research/25_oos_best_versions.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, evaluate, backtest,
                      fundamentals, events, altdata, riskmodel, optimizer)
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
C, WARM, PPY = 0.003, 36, 12
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]


def build_pool(panel):
    pool = {}
    for n, (fn, _) in classic.REGISTRY.items():
        pool[n] = fn
    for mod in (fundamentals, events, altdata):
        for n, (fn, _) in getattr(mod, "REGISTRY", {}).items():
            pool[n] = fn
    try:
        from quantlib.factors import behavioral
        for n, (fn, _) in behavioral.REGISTRY.items():
            pool["b_" + n] = fn
    except Exception:
        pass
    return pool


def metrics_full(r, bench=None):
    r = r.dropna(); n = len(r)
    ann = (1 + r).prod() ** (PPY / n) - 1
    vol = r.std(ddof=1) * np.sqrt(PPY)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(PPY)
    nav = (1 + r).cumprod(); mdd = (nav / nav.cummax() - 1).min()
    calmar = ann / abs(mdd) if mdd else np.nan
    if bench is not None:
        b = bench.reindex(r.index); ex = (r - b).dropna()
        exann = (1 + r).prod() ** (PPY / n) - (1 + b.reindex(r.index)).prod() ** (PPY / n)
        win = (ex > 0).mean()
    else:
        exann = ann; win = (r > 0).mean()
    return {"年化收益": ann, "年化波动": vol, "夏普": sharpe, "最大回撤": mdd,
            "卡玛": calmar, "超额": exann, "胜率": win}


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
            if z.notna().mean() < 0.3: continue
            panel["_z_" + name] = z.values; names.append(name)
        except Exception:
            pass
    IC = pd.DataFrame({n: evaluate.compute_ic(panel, panel["_z_" + n]) for n in names}).sort_index()
    # OOS 逐期贪心
    alpha = pd.Series(np.nan, index=panel.index)
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
        idx = panel.index[panel["trddt"] == t]
        panel.loc[idx, "alpha"] = (sum(signs[c] * panel.loc[idx, "_z_" + c] for c in selected) / len(selected)).values
    oos = panel[panel["alpha"].notna()]
    print(f"OOS alpha {oos['trddt'].min().date()}~{oos['trddt'].max().date()}，{time.time()-t0:.0f}s", flush=True)
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    i500 = backtest.load_benchmark("000905", FREQ); i800 = backtest.load_benchmark("000906", FREQ)
    i1000 = backtest.load_benchmark("000852", FREQ)

    # V1 多头·全A
    rows, prev = [], set()
    for dt, g in oos.dropna(subset=["fwd_ret"]).groupby("trddt"):
        top = g.nlargest(max(1, len(g) // 10), "alpha"); cur = set(top["stkcd"])
        to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows.append({"dt": dt, "g": top["fwd_ret"].mean(), "to": to}); prev = cur
    L = pd.DataFrame(rows).set_index("dt"); v1 = L["g"] - L["to"] * C
    # V2 市场中性 IM 贴水3%
    d = pd.concat([v1, i1000], axis=1).dropna(); beta = float(np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0])
    v2 = (v1 - beta * i1000.reindex(v1.index) - beta * 0.03 / 12).dropna()
    # V3 中证800 指增 TE=3%
    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")
    con = data.connect()
    bw = con.sql(f"""SELECT CAST(Enddt AS DATE) dt, Stkcd stkcd, TRY_CAST(Weight AS DOUBLE) w
                     FROM '{data.DATA_ROOT}/raw/IDX_Smprat.parquet' WHERE Indexcd='000906'""").df()
    con.close(); bw["dt"] = bw["dt"].astype("datetime64[ns]"); bw = bw.dropna()
    snaps = np.sort(bw["dt"].unique()); erows, wprev = [], None
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
        w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, dd, active_cap=0.02, te=0.03, style_band=0.10)
        ws = pd.Series(w, index=m["stkcd"].values)
        to = 0.0 if wprev is None else 0.5 * ws.subtract(wprev, fill_value=0).abs().sum()
        erows.append({"dt": dt, "r": float(np.nansum(w * m["fwd_ret"].values)) - to * C,
                      "bench": float(np.nansum(b * m["fwd_ret"].values))}); wprev = ws
    E = pd.DataFrame(erows).set_index("dt"); v3 = E["r"]

    res = {"V1 多头·全A(OOS)": metrics_full(v1, i500),
           "V2 市场中性·IM贴水3%(OOS)": metrics_full(v2),
           "V3 中证800指增 TE3%(OOS)": metrics_full(v3, i800.reindex(v3.index))}
    out = pd.DataFrame(res).T
    for c in ["年化收益", "年化波动", "最大回撤", "超额"]:
        out[c] = (out[c] * 100).round(1).astype(str) + "%"
    out["夏普"] = out["夏普"].round(2); out["卡玛"] = out["卡玛"].round(2)
    out["胜率"] = (out["胜率"] * 100).round(0).astype(int).astype(str) + "%"
    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 84, "\n最佳几版策略 完整指标（OOS诚实口径，扣0.3%单边换手）\n", "=" * 84, sep="")
    print(out.to_string()); out.to_csv("results/25_oos_best_versions.csv", encoding="utf-8-sig")
    NAV = pd.DataFrame({"V1多头全A": (1 + v1).cumprod(), "V2市场中性": (1 + v2).cumprod(),
                        "V3中证800指增": (1 + v3).cumprod(),
                        "中证500": (1 + i500.reindex(v1.index)).cumprod(),
                        "中证800": (1 + i800.reindex(v1.index)).cumprod()})
    NAV.to_csv("results/25_oos_nav.csv", encoding="utf-8-sig")
    print(f"\n净值存 results/25_oos_nav.csv；完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
