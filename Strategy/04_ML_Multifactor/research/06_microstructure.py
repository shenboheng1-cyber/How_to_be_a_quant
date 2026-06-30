# -*- coding: utf-8 -*-
"""
研究脚本 06 —— 高频微观结构因子：有效性 + 与价量因子的正交性
================================================================
对订单流/价差/VPIN/跳跃类因子做 IC/分层，并【与现有价量因子算相关性】，
回答最关键的问题：它们是真·新信息，还是又一个流动性代理？

用法：python research/06_microstructure.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import data, universe, preprocess, evaluate, microstructure
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"


def main():
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    micro = microstructure.load_micro_features(FREQ, START, END)
    panel = panel.merge(micro, on=["stkcd", "trddt"], how="left")
    print(f"研究样本 {panel.shape[0]} 行，已并入微观结构因子\n")

    # 微观结构因子（带方向与经济假设）
    micro_factors = {
        "ofi":        (panel["ofi"],         "订单流不平衡(净买)"),
        "ofi_big":    (panel["ofi_big"],     "大单不平衡(主力方向)"),
        "ofi_small":  (-panel["ofi_small"],  "小单不平衡(散户,反向)"),
        "spread":     (panel["spread"],      "有效价差(流动性溢价)"),
        "vpin":       (panel["vpin"],        "知情交易概率VPIN"),
        "rskew":      (-panel["rskew"],      "已实现偏度(彩票,反向)"),
        "downside":   (panel["downside"],    "下行半方差占比"),
        "jump_freq":  (panel["jump_freq"],   "跳跃频率"),
        "sjv":        (-panel["sjv"],        "符号跳跃(反向)"),
    }
    # 对照用的现有价量因子
    existing = ["reversal", "low_turnover", "illiquidity", "size", "low_vol", "max_ret"]

    # 预处理所有因子（市值中性化）
    proc = {}
    for k, (raw, _) in micro_factors.items():
        proc[k] = preprocess.preprocess_factor(panel, raw, do_neutralize=True)
    for k in existing:
        proc[k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=True)

    # ---- IC / 分层 ----
    rows = []
    for k, (_, cn) in micro_factors.items():
        res = evaluate.evaluate_factor(panel, proc[k], n_groups=10, freq=FREQ)
        ic = res["ic_summary"]; ls = res["quantile_summary"].loc["多空(QN-Q1)"]
        rows.append({"因子": cn, "代码": k, "RankIC": ic["IC均值"], "ICIR": ic["ICIR"],
                     "t值": ic["t值"], "多空夏普": ls["夏普"]})
    tbl = pd.DataFrame(rows).reindex(
        pd.DataFrame(rows)["ICIR"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    pd.set_option("display.unicode.east_asian_width", True)
    print("=" * 70, "\n微观结构因子 有效性（市值中性化）\n", "=" * 70, sep="")
    print(tbl.to_string(index=False))

    # ---- 与现有价量因子的相关性（正交性证明）----
    allk = list(micro_factors) + existing
    df = pd.DataFrame({k: proc[k].values for k in allk})
    df["dt"] = panel["trddt"].values
    df = df.dropna()
    corr = df.groupby("dt")[allk].corr().groupby(level=1).mean().loc[list(micro_factors), existing]
    print("\n" + "=" * 70, "\n微观结构因子 × 现有价量因子 平均横截面相关\n", "=" * 70, sep="")
    print(corr.round(2).to_string())
    print("\n每个微观因子与所有价量因子的最大|相关|（越低越正交）:")
    print(corr.abs().max(axis=1).round(2).to_string())

    os.makedirs("results", exist_ok=True)
    tbl.to_csv("results/06_micro_summary.csv", index=False, encoding="utf-8-sig")
    corr.to_csv("results/06_micro_corr.csv", encoding="utf-8-sig")
    print("\n已保存 results/06_micro_summary.csv 和 06_micro_corr.csv")


if __name__ == "__main__":
    main()
