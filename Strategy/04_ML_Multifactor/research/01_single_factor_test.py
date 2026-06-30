# -*- coding: utf-8 -*-
"""
研究脚本 01 —— 经典因子库单因子检验（L1 baseline）
================================================================
把 quantlib 引擎跑通整条流水线，对每个经典因子输出 IC 与分层回测指标，
形成一张横向对比表 —— 这是后续所有工作的 baseline。

用法：python research/01_single_factor_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from quantlib import data, universe, preprocess, evaluate
from quantlib.factors import classic

FREQ = "M"
START, END = "2015-01-01", "2025-12-31"


def main():
    print("加载研究面板 ...")
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.add_universe(panel, min_list_days=120, verbose=True)
    panel = panel[panel["in_universe"]].reset_index(drop=True)
    print(f"研究样本：{len(panel):,} 行，{panel.trddt.nunique()} 期\n")

    rows = []
    for name, (fn, cn) in classic.REGISTRY.items():
        raw = fn(panel)
        f = preprocess.preprocess_factor(panel, raw, size_col="total_mktcap",
                                         do_neutralize=True)
        res = evaluate.evaluate_factor(panel, f, n_groups=10, freq=FREQ)
        ic = res["ic_summary"]
        ls = res["quantile_summary"].loc["多空(QN-Q1)"]
        rows.append({
            "因子": cn, "代码": name,
            "RankIC": ic["IC均值"], "ICIR": ic["ICIR"], "t值": ic["t值"],
            "IC>0占比": ic["IC>0占比"],
            "多空年化": round(ls["年化收益"], 4),
            "多空夏普": ls["夏普"], "多空回撤": ls["最大回撤"],
        })

    tbl = pd.DataFrame(rows).sort_values("ICIR", key=lambda s: s.abs(), ascending=False)
    pd.set_option("display.unicode.east_asian_width", True)
    print("=" * 90)
    print("经典因子单因子检验结果（市值中性化，月频，2015-2025，全A股）")
    print("=" * 90)
    print(tbl.to_string(index=False))

    os.makedirs("results", exist_ok=True)
    tbl.to_csv("results/01_single_factor_summary.csv", index=False, encoding="utf-8-sig")
    print("\n已保存 results/01_single_factor_summary.csv")


if __name__ == "__main__":
    main()
