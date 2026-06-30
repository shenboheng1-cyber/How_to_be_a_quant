# -*- coding: utf-8 -*-
"""
研究脚本 03 —— 因子工厂：批量生成+检验 95 个因子 + 多重检验校正
================================================================
系统化生成上百个因子，全部跑同一条流水线，并诚实回答：
扣掉多重检验后，真正扛得住的有几个？

用法：python research/03_alpha_factory.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from quantlib import data, universe
from quantlib.alpha import matrices, alphas, factory

FREQ, START, END = "M", "2015-01-01", "2025-12-31"


def main():
    t = time.time()
    print("加载宽矩阵 ...")
    M = matrices.load_matrices(START, END)
    reg = alphas.build_registry(M)
    print(f"  因子工厂生成 {len(reg)} 个因子 | 矩阵 {M.close.shape} | {time.time()-t:.0f}s")

    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)

    print("批量评估 ...")
    t = time.time()
    tbl = factory.evaluate_alphas(panel, reg, M, freq=FREQ, do_neutralize=True)
    print(f"  评估完成 {time.time()-t:.0f}s\n")

    pd.set_option("display.unicode.east_asian_width", True)
    print("=" * 70, "\nTop 15 因子（按 |ICIR|）\n", "=" * 70, sep="")
    print(tbl.head(15).to_string(index=False))

    print("\n" + "=" * 70, "\n多重检验校正（关键：诚实区分真信号与巧合）\n", "=" * 70, sep="")
    mt = factory.multiple_testing_summary(tbl["t值"].dropna())
    for k, v in mt.items():
        print(f"  {k:<22} {v}")

    os.makedirs("results", exist_ok=True)
    tbl.to_csv("results/03_alpha_factory_summary.csv", index=False, encoding="utf-8-sig")
    print("\n已保存 results/03_alpha_factory_summary.csv")


if __name__ == "__main__":
    main()
