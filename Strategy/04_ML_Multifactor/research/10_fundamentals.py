# -*- coding: utf-8 -*-
"""
研究脚本 10 —— 基本面因子：IC + 与价量的正交性
================================================================
PIT(法定披露截止日)对齐的年报基本面因子，检验有效性 + 与现有价量因子的正交性。

用法：python research/10_fundamentals.py
"""
import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import data, universe, preprocess, evaluate, fundamentals
from quantlib.alpha import factory
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
EXISTING = ["reversal", "low_turnover", "illiquidity", "size", "low_vol", "max_ret"]


def main():
    t = time.time()
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = fundamentals.attach(panel)
    cov = panel["net_profit"].notna().mean()
    print(f"PIT 基本面覆盖率: {cov:.1%} | {time.time()-t:.0f}s\n")

    proc = {}
    for nm, (fn, cn) in fundamentals.REGISTRY.items():
        proc[nm] = preprocess.preprocess_factor(panel, fn(panel), do_neutralize=True)
    for k in EXISTING:
        proc[k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=True)
    names = list(fundamentals.REGISTRY.keys())

    rows = []
    for nm in names:
        res = evaluate.evaluate_factor(panel, proc[nm], n_groups=10, freq=FREQ)
        ic = res["ic_summary"]; ls = res["quantile_summary"].loc["多空(QN-Q1)"]
        rows.append({"因子": fundamentals.REGISTRY[nm][1], "code": nm, "RankIC": ic["IC均值"],
                     "ICIR": ic["ICIR"], "t值": ic["t值"], "多空夏普": ls["夏普"]})
    tbl = pd.DataFrame(rows)

    df = pd.DataFrame({k: proc[k].values for k in names + EXISTING})
    df["dt"] = panel["trddt"].values
    df = df.dropna()
    corr = df.groupby("dt")[names + EXISTING].corr().groupby(level=1).mean()
    tbl["max|corr价量|"] = tbl["code"].map(corr.loc[names, EXISTING].abs().max(axis=1)).round(2)
    tbl = tbl.reindex(tbl["ICIR"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    pd.set_option("display.unicode.east_asian_width", True)
    print("=" * 76, "\n基本面因子(年报PIT) — IC 与正交性\n", "=" * 76, sep="")
    print(tbl.to_string(index=False))
    print("\n多重检验:")
    for k, v in factory.multiple_testing_summary(tbl["t值"].dropna()).items():
        print(f"  {k:<20} {v}")
    gem = tbl[(tbl["t值"].abs() > 2) & (tbl["max|corr价量|"] < 0.3)]
    print(f"\n★ 显著(|t|>2) 且 与价量正交(max|corr|<0.3): {len(gem)} 个")
    print(gem.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    tbl.to_csv("results/10_fundamentals.csv", index=False, encoding="utf-8-sig")
    print("\n已保存 results/10_fundamentals.csv")


if __name__ == "__main__":
    main()
