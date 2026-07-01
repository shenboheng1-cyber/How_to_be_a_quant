# -*- coding: utf-8 -*-
"""
研究脚本 16 —— 因子择时（动态因子权重 vs 静态 baseline）
================================================================
对一组因子做动态赋权(因子动量/IC_IR)，与 等权/静态IC/扩张IC 对比。
铁律：择时必须样本外跑赢静态 baseline 才算有用。

用法：python research/16_factor_timing.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import data, universe, evaluate, factor_timing as ft
from quantlib.factors import classic, behavioral

FREQ, START, END = "M", "2015-01-01", "2025-12-31"


def main():
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)

    funcs = {k: classic.REGISTRY[k][0] for k in
             ["reversal", "low_turnover", "low_vol", "max_ret", "illiquidity",
              "size", "ep", "bp", "momentum", "sp"]}
    funcs["w52_high"] = behavioral.REGISTRY["w52_high"][0]
    print(f"因子 {len(funcs)} 个，计算 IC 面板 ...")
    fvals, ic_df = ft.factor_panel(panel, funcs, neutralize=True)

    # 共同样本外窗口：跳过前 24 期(让扩张/滚动都已定义)
    oos_start = ic_df.index[24]
    rows = {}
    for scheme in ["equal", "static", "expanding", "mom", "icir"]:
        w = ft.weights(ic_df, scheme, window=12)
        comp = ft.composite(panel, fvals, w)
        mask = (panel["trddt"] >= oos_start).values
        sub = panel[mask].reset_index(drop=True)
        f = pd.Series(comp[mask].values)
        ic = evaluate.ic_summary(evaluate.compute_ic(sub, f))
        qs = evaluate.quantile_summary(evaluate.quantile_returns(sub, f, 10))
        ls = qs.loc["多空(QN-Q1)"]
        # 因子权重换手(择时churn代价的代理)
        wt = w[w.index >= oos_start]
        turn = wt.diff().abs().sum(axis=1).mean()
        rows[scheme] = {"RankIC": ic["IC均值"], "ICIR": ic["ICIR"], "t值": ic["t值"],
                        "多空年化": round(ls["年化收益"], 4), "多空夏普": ls["夏普"],
                        "权重月换手": round(turn, 2)}

    name = {"equal": "等权(baseline)", "static": "静态IC(偷看,上限)", "expanding": "扩张IC",
            "mom": "因子动量(近12m IC)", "icir": "IC_IR(近12m)"}
    pd.set_option("display.unicode.east_asian_width", True)
    out = pd.DataFrame(rows).T.rename(index=name)
    print("\n" + "=" * 74, f"\n因子择时对比（样本外 {oos_start.date()} 起）\n", "=" * 74, sep="")
    print(out.to_string())
    base = rows["equal"]["ICIR"]
    print(f"\n判定(vs 等权 baseline ICIR={base}):")
    for s in ["expanding", "mom", "icir"]:
        v = rows[s]["ICIR"]
        print(f"  {name[s]:<20} ICIR={v}  {'✓跑赢' if v > base else '✗没跑赢'}静态")
    out.to_csv("results/16_factor_timing.csv", encoding="utf-8-sig")
    print("\n已保存 results/16_factor_timing.csv")


if __name__ == "__main__":
    main()
