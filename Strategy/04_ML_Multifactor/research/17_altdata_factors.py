# -*- coding: utf-8 -*-
"""
研究脚本 17 —— 另类数据因子(机构调研/专利) + 行业中性化验证
================================================================
用法：python research/17_altdata_factors.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import data, universe, preprocess, evaluate, altdata
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
EXIST = ["reversal", "low_turnover", "size", "illiquidity"]


def main():
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel = altdata.attach_altfactors(panel)
    print(f"行业覆盖 {panel['industry'].notna().mean():.1%} | 行业数 {panel['industry'].nunique()} | "
          f"调研非0 {(panel.research90>0).mean():.1%} | 专利非0 {(panel.patent365>0).mean():.1%}\n")

    # 新因子 IC + 与价量正交性（行业+市值中性）
    proc = {}
    for nm, (fn, cn) in altdata.REGISTRY.items():
        proc[nm] = preprocess.preprocess_factor(panel, fn(panel), industry_col="industry", do_neutralize=True)
    for k in EXIST:
        proc[k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), industry_col="industry", do_neutralize=True)
    names = list(altdata.REGISTRY)
    df = pd.DataFrame({k: proc[k].values for k in names + EXIST}); df["dt"] = panel["trddt"].values
    corr = df.dropna().groupby("dt")[names + EXIST].corr().groupby(level=1).mean()
    pd.set_option("display.unicode.east_asian_width", True)
    rows = []
    for nm in names:
        ic = evaluate.ic_summary(evaluate.compute_ic(panel, proc[nm]))
        rows.append({"因子": altdata.REGISTRY[nm][1], "RankIC": ic["IC均值"], "ICIR": ic["ICIR"],
                     "t值": ic["t值"], "max|corr价量|": round(corr.loc[nm, EXIST].abs().max(), 2)})
    print("=" * 60, "\n另类因子(行业+市值中性)\n", "=" * 60, sep="")
    print(pd.DataFrame(rows).to_string(index=False))

    # 验证行业中性化：某因子 size-only vs size+行业 中性后, 行业内残差均值应≈0
    raw = classic.REGISTRY["reversal"][0](panel)
    f_size = preprocess.preprocess_factor(panel, raw, do_neutralize=True)                       # 仅市值
    f_ind = preprocess.preprocess_factor(panel, raw, industry_col="industry", do_neutralize=True)  # +行业
    tmp = pd.DataFrame({"ind": panel["industry"].values, "fs": f_size.values, "fi": f_ind.values}).dropna()
    # 各行业均值的离散度：行业中性后应更接近0
    disp_size = tmp.groupby("ind")["fs"].mean().abs().mean()
    disp_ind = tmp.groupby("ind")["fi"].mean().abs().mean()
    print(f"\n行业中性化验证(反转因子各行业均值的平均|偏离|)：")
    print(f"  仅市值中性: {disp_size:.4f}   行业+市值中性: {disp_ind:.4f}  (后者应明显更小=行业暴露被剥离)")


if __name__ == "__main__":
    main()
