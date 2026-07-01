# -*- coding: utf-8 -*-
"""
研究脚本 16b —— 因子择时（多样化/正交因子集）
================================================================
验证诊断：因子动量在【同质因子】上无效，但在【多样/正交因子】(价量+基本面+行为)上
是否有用？同质集见 research/16。

用法：python research/16b_timing_diverse.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import data, universe, evaluate, fundamentals, factor_timing as ft
from quantlib.factors import classic, behavioral

FREQ, START, END = "M", "2015-01-01", "2025-12-31"


def main():
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = fundamentals.attach(panel)

    # 多样化因子：价量(4) + 基本面(5,与价量正交) + 行为(1)
    funcs = {
        "reversal": classic.REGISTRY["reversal"][0], "low_turnover": classic.REGISTRY["low_turnover"][0],
        "low_vol": classic.REGISTRY["low_vol"][0], "illiquidity": classic.REGISTRY["illiquidity"][0],
        "f_ep": fundamentals.ep, "f_bp": fundamentals.bp, "f_roe": fundamentals.roe,
        "f_accruals": fundamentals.accruals, "f_cfp": fundamentals.cfp,
        "w52_high": behavioral.REGISTRY["w52_high"][0],
    }
    print(f"多样化因子 {len(funcs)} 个，计算 IC 面板 ...")
    fvals, ic_df = ft.factor_panel(panel, funcs, neutralize=True)

    # 因子间平均相关(确认更多样)
    fv = pd.DataFrame({k: v.values for k, v in fvals.items()})
    fv["dt"] = panel["trddt"].values
    corr = fv.dropna().groupby("dt")[list(funcs)].corr().groupby(level=1).mean()
    avg_abs = corr.abs().values[np.triu_indices(len(funcs), 1)].mean()
    print(f"因子间平均绝对相关 = {avg_abs:.3f}（越低越多样）")

    oos_start = ic_df.index[24]
    rows = {}
    for scheme in ["equal", "static", "expanding", "mom", "icir"]:
        w = ft.weights(ic_df, scheme, window=12)
        comp = ft.composite(panel, fvals, w)
        mask = (panel["trddt"] >= oos_start).values
        sub = panel[mask].reset_index(drop=True)
        f = pd.Series(comp[mask].values)
        ic = evaluate.ic_summary(evaluate.compute_ic(sub, f))
        ls = evaluate.quantile_summary(evaluate.quantile_returns(sub, f, 10)).loc["多空(QN-Q1)"]
        rows[scheme] = {"RankIC": ic["IC均值"], "ICIR": ic["ICIR"], "t值": ic["t值"], "多空夏普": ls["夏普"]}

    name = {"equal": "等权(baseline)", "static": "静态IC(偷看)", "expanding": "扩张IC",
            "mom": "因子动量(近12m)", "icir": "IC_IR(近12m)"}
    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 64, f"\n多样化因子集 — 因子择时（样本外 {oos_start.date()} 起）\n", "=" * 64, sep="")
    print(pd.DataFrame(rows).T.rename(index=name).to_string())
    base = rows["equal"]["ICIR"]
    print(f"\n判定(vs 等权 ICIR={base}):")
    for s in ["mom", "icir"]:
        v = rows[s]["ICIR"]
        print(f"  {name[s]:<16} ICIR={v}  {'✓跑赢等权!' if v > base else '✗仍没跑赢'}")
    pd.DataFrame(rows).T.to_csv("results/16b_timing_diverse.csv", encoding="utf-8-sig")
    print("\n已保存")


if __name__ == "__main__":
    main()
