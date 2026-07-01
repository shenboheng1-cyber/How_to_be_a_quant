# -*- coding: utf-8 -*-
"""
研究脚本 19 —— 中证800 指数增强（基准相对优化）
================================================================
对齐中证800(000906)行业/风格暴露 + 跟踪误差预算的优化器，对比 naive 倾斜。
看 IR / 跟踪误差 / 信息比 能否真升。

用法：/opt/anaconda3/bin/python research/19_enhanced_index.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import data, universe, preprocess, backtest, altdata, riskmodel, optimizer
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]
BENCH = "000906"   # 中证800（成分权重齐全；中证500/300成分不在IDX_Smprat）


def load_bench_weights():
    con = data.connect()
    df = con.sql(f"""SELECT CAST(Enddt AS DATE) AS dt, Stkcd AS stkcd, TRY_CAST(Weight AS DOUBLE) AS w
                     FROM '{data.DATA_ROOT}/raw/IDX_Smprat.parquet' WHERE Indexcd='{BENCH}'""").df()
    con.close()
    df["dt"] = df["dt"].astype("datetime64[ns]")
    return df.dropna()


def main():
    t0 = time.time()
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    A = [preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), industry_col="industry", do_neutralize=True)
         for k in ["reversal", "low_turnover", "low_vol", "illiquidity", "ep"]]
    panel["alpha"] = pd.concat(A, axis=1).mean(axis=1).values

    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid)
    sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")
    bw = load_bench_weights()
    bench_ret = backtest.load_benchmark(BENCH, FREQ)
    print(f"准备完成 {time.time()-t0:.0f}s，开始逐月优化 ...", flush=True)

    TES = [0.03, 0.05, 0.08]                                 # 跟踪误差预算扫描
    snaps = np.sort(bw["dt"].unique())
    dates = sorted(panel["trddt"].unique())
    rows = []
    for dt in dates[24:]:
        snap = snaps[snaps <= np.datetime64(dt)]
        if len(snap) == 0: continue
        cons = bw[bw["dt"] == snap[-1]][["stkcd", "w"]]
        g = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna()]
        m = g.merge(cons, on="stkcd", how="inner")          # 基准成分 ∩ 可投
        if len(m) < 200: continue
        b = (m["w"] / m["w"].sum()).values                   # 归一基准权重
        Xind = pd.get_dummies(m["industry"]).astype(float)
        cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values
        rec = {"dt": dt, "bench": float(np.nansum(b * fwd))}
        for te in TES:
            w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, d,
                                            active_cap=0.02, te=te, style_band=0.10)
            rec[f"opt{te}"] = float(np.nansum(w * fwd))
        nv = m.nlargest(max(1, len(m) // 3), "alpha")        # naive 倾斜:成分内 alpha 前30% 等权
        rec["naive"] = nv["fwd_ret"].mean()
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("dt")

    def rep(col):
        ex = (R[col] - R["bench"]).dropna()
        te = ex.std(ddof=1) * np.sqrt(12)
        ir = ex.mean() / ex.std(ddof=1) * np.sqrt(12)
        nav = (1 + ex).cumprod()
        return {"年化超额": round(ex.mean() * 12, 4), "跟踪误差": round(te, 3),
                "信息比IR": round(ir, 2), "超额最大回撤": round((nav / nav.cummax() - 1).min(), 3)}

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 64, f"\n中证800 指数增强（样本外，对标{BENCH}）\n", "=" * 64, sep="")
    cols = {"naive倾斜(成分内top30%)": rep("naive")}
    for te in TES:
        cols[f"优化器 TE预算={te:.0%}"] = rep(f"opt{te}")
    out = pd.DataFrame(cols).T
    print(out.to_string())
    print(f"\n基准中证800 年化: {((1+R['bench']).prod()**(12/len(R))-1):.1%}")
    out.to_csv("results/19_enhanced_index.csv", encoding="utf-8-sig")
    print(f"完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
