# -*- coding: utf-8 -*-
"""
研究脚本 04 —— 国泰君安 191 价量因子：批量评估 + 多重检验
================================================================
把 GTJA 191 因子（quantlib/alpha/gtja191.py）全部跑过同一条流水线，
统计有效数、IC 分布、Top 因子，并做多重检验校正。

用法：python research/04_gtja191.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd
from quantlib import data, universe
from quantlib.alpha import matrices, gtja191, factory

FREQ, START, END = "M", "2015-01-01", "2025-12-31"


def main():
    t = time.time()
    print("加载宽矩阵 ...")
    M = matrices.load_matrices(START, END)
    reg = gtja191.build_registry(M)
    print(f"  GTJA 因子 {len(reg)} 个 | 矩阵 {M.close.shape} | 不支持 {len(gtja191.UNSUPPORTED)} 个")

    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)

    print("批量评估（191 个，含较慢算子，约几分钟）...")
    tbl = factory.evaluate_alphas(panel, reg, M, freq=FREQ, do_neutralize=True, verbose=True)
    n_valid = tbl["ICIR"].notna().sum()
    print(f"\n评估完成 {time.time()-t:.0f}s | 有效计算 {n_valid}/{len(reg)} 个\n")

    pd.set_option("display.unicode.east_asian_width", True)
    print("=" * 64, "\nGTJA Top 15 因子（按 |ICIR|）\n", "=" * 64, sep="")
    print(tbl.head(15).to_string(index=False))

    print("\n" + "=" * 64, "\n多重检验校正\n", "=" * 64, sep="")
    for k, v in factory.multiple_testing_summary(tbl["t值"].dropna()).items():
        print(f"  {k:<22} {v}")

    os.makedirs("results", exist_ok=True)
    tbl.to_csv("results/04_gtja191_summary.csv", index=False, encoding="utf-8-sig")
    print("\n已保存 results/04_gtja191_summary.csv")


if __name__ == "__main__":
    main()
