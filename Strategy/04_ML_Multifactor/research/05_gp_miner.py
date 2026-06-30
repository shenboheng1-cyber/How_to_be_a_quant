# -*- coding: utf-8 -*-
"""
研究脚本 05 —— 遗传规划因子挖掘
================================================================
让算法在算子树空间里自动进化价量因子，训练期选优、样本外验证。
名人堂只收"训练强 + 与已选去相关"的公式，并报告其样本外 ICIR。

用法：python research/05_gp_miner.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pandas as pd
from quantlib import data, universe
from quantlib.alpha import matrices, gp_miner


def main():
    t = time.time()
    print("加载宽矩阵 + 训练/样本外面板 ...")
    M = matrices.load_matrices("2015-01-01", "2025-12-31")
    train = universe.filter_universe(
        data.load_research_panel("M", "2015-01-01", "2020-12-31"), verbose=False)
    oos = universe.filter_universe(
        data.load_research_panel("M", "2021-01-01", "2025-12-31"), verbose=False)
    print(f"  训练 {train.trddt.nunique()} 期 | 样本外 {oos.trddt.nunique()} 期 | {time.time()-t:.0f}s")

    print("开始进化 ...")
    hof = gp_miner.evolve(M, train, oos, pop_size=50, generations=8,
                          parsimony=0.002, hof_size=15, max_corr=0.7, seed=42)

    df = pd.DataFrame(hof, columns=["进化出的公式", "训练ICIR", "样本外ICIR", "复杂度"])
    df["OOS稳健"] = ((df["训练ICIR"] * df["样本外ICIR"] > 0) &
                     (df["样本外ICIR"].abs() > 0.2)).map({True: "✓", False: ""})
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.max_colwidth", 60)
    print(f"\n进化完成 {time.time()-t:.0f}s\n")
    print("=" * 80, "\n名人堂（训练选优 → 去相关 → 样本外验证）\n", "=" * 80, sep="")
    print(df.to_string(index=False))
    n_robust = (df["OOS稳健"] == "✓").sum()
    print(f"\n样本外稳健(训练OOS同号且|OOS ICIR|>0.2)的因子: {n_robust}/{len(df)}")

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/05_gp_factors.csv", index=False, encoding="utf-8-sig")
    print("已保存 results/05_gp_factors.csv")


if __name__ == "__main__":
    main()
