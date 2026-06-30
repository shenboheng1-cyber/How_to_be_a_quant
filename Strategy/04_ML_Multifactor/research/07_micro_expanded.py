# -*- coding: utf-8 -*-
"""
研究脚本 07 —— 扩展版微观结构因子目录：批量 IC + 正交性 + 多重检验
================================================================
读取多智能体产出的因子目录(results/micro_catalog.json)，批量构建并检验，
重点筛出"显著 且 与价量正交"的因子——L3 真正能用的增量。

用法：python research/07_micro_expanded.py
"""
import sys, os, json, warnings, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import data, universe, preprocess, evaluate, microstructure
from quantlib.alpha import factory
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
EXISTING = ["reversal", "low_turnover", "illiquidity", "size", "low_vol", "max_ret"]


def main():
    catalog = json.load(open("results/micro_catalog.json"))["factors"]
    print(f"因子目录 {len(catalog)} 个")

    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    t = time.time()
    micro = microstructure.load_specs(catalog, FREQ, START, END)
    panel = panel.merge(micro, on=["stkcd", "trddt"], how="left")
    print(f"构建因子 {time.time()-t:.0f}s，并入面板\n")

    # 预处理：新因子(带sign) + 现有价量因子
    proc, meta = {}, {}
    for s in catalog:
        nm = s["name"]
        if nm not in panel.columns:
            continue
        proc[nm] = preprocess.preprocess_factor(panel, panel[nm] * s["sign"], do_neutralize=True)
        meta[nm] = (s.get("theme", ""), s.get("hypothesis", ""))
    for k in EXISTING:
        proc[k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=True)

    names = [s["name"] for s in catalog if s["name"] in proc]

    # IC / 分层
    rows = []
    for nm in names:
        res = evaluate.evaluate_factor(panel, proc[nm], n_groups=10, freq=FREQ)
        ic = res["ic_summary"]; ls = res["quantile_summary"].loc["多空(QN-Q1)"]
        rows.append({"因子": nm, "theme": meta[nm][0], "RankIC": ic["IC均值"],
                     "ICIR": ic["ICIR"], "t值": ic["t值"], "多空夏普": ls["夏普"]})
    tbl = pd.DataFrame(rows)

    # 与现有价量因子最大|相关|
    df = pd.DataFrame({k: proc[k].values for k in names + EXISTING})
    df["dt"] = panel["trddt"].values
    df = df.dropna()
    corr = df.groupby("dt")[names + EXISTING].corr().groupby(level=1).mean()
    maxcorr = corr.loc[names, EXISTING].abs().max(axis=1)
    tbl["max|corr价量|"] = tbl["因子"].map(maxcorr).round(2)
    tbl = tbl.reindex(tbl["ICIR"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_rows", 100)
    print("=" * 84, "\n扩展微观结构因子 — 全部(按|ICIR|)\n", "=" * 84, sep="")
    print(tbl.to_string(index=False))

    print("\n多重检验:")
    for k, v in factory.multiple_testing_summary(tbl["t值"].dropna()).items():
        print(f"  {k:<22} {v}")

    # 关键：显著 且 正交
    gem = tbl[(tbl["t值"].abs() > 3) & (tbl["max|corr价量|"] < 0.4)]
    print(f"\n★ 显著(|t|>3) 且 与价量正交(max|corr|<0.4) 的因子: {len(gem)} 个")
    print(gem.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    tbl.to_csv("results/07_micro_expanded.csv", index=False, encoding="utf-8-sig")
    print("\n已保存 results/07_micro_expanded.csv")


if __name__ == "__main__":
    main()
