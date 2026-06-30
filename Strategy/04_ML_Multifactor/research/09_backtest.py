# -*- coding: utf-8 -*-
"""
研究脚本 09 —— L4：带成本的组合回测，对标中证500
================================================================
用 L3 的 ML 合成信号构建组合，扣真实交易成本，对标中证500，给出净业绩 + 成本敏感性。

用法：DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib python research/09_backtest.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from quantlib import data, ml, backtest

FREQ, START, END = "M", "2015-01-01", "2025-12-31"


def main():
    t0 = time.time()
    df = pd.read_parquet("results/08_features.parquet")
    feat = [c for c in df.columns if c not in ("stkcd", "trddt", "y")]
    X, y, dates = df[feat].values, df["y"].values, df["trddt"].values
    print(f"特征 {X.shape}，复现 ML 信号(walk-forward)...")
    pred = ml.walk_forward_predict(X, y, dates, ml.lgb_model(), init=36, embargo=1, step=3)
    oos = ~np.isnan(pred)

    base = data.load_research_panel(FREQ, START, END)
    base = base.merge(df[["stkcd", "trddt"]], on=["stkcd", "trddt"], how="right").reset_index(drop=True)
    panel_oos = base[oos].reset_index(drop=True)
    sig = pred[oos]
    bench = backtest.load_benchmark("000905", FREQ, START, END)   # 中证500

    bt = backtest.backtest(panel_oos, sig, bench, cost=0.0)       # 先算毛+换手
    print(f"回测期 {len(bt)} 个月 | {time.time()-t0:.0f}s\n")

    C = 0.003                                                     # 双边千3
    long_net = bt["long_g"] - bt["to_l"] * C
    ls_net = bt["ls_g"] - bt["to_ls"] * C
    long_excess = long_net - bt["bench"]

    pd.set_option("display.unicode.east_asian_width", True)
    print("=" * 66, "\nL4 净业绩（扣双边千3，对标中证500）\n", "=" * 66, sep="")
    res = pd.DataFrame({
        "多头(净)": backtest.metrics(long_net, FREQ),
        "多空(净)": backtest.metrics(ls_net, FREQ),
        "中证500": backtest.metrics(bt["bench"], FREQ),
    }).T
    print(res.to_string())
    print(f"\n多头超额 年化: {long_excess.mean()*12:.2%} | 信息比率IR: {backtest.info_ratio(long_excess, FREQ)}")
    print(f"年化换手(多头): {bt['to_l'].mean()*12:.1f}x | (多空): {bt['to_ls'].mean()*12:.1f}x")

    print("\n成本敏感性（多头 top decile 净夏普 / 净年化）：")
    print(f"{'双边成本':>8} {'净年化':>8} {'净夏普':>8} {'对500超额':>10}")
    for c in [0.0, 0.001, 0.002, 0.003, 0.005, 0.008]:
        ln = bt["long_g"] - bt["to_l"] * c
        m = backtest.metrics(ln, FREQ)
        ex = (ln - bt["bench"]).mean() * 12
        print(f"{c*100:>7.1f}% {m['年化']:>8.1%} {m['夏普']:>8.2f} {ex:>9.1%}")

    # 容量粗估
    med_cap = bt["med_cap"].median()
    print(f"\n持仓中位市值≈{med_cap/1e8:.0f}亿（偏小盘→容量受限，需注意冲击成本）")

    # 图：净值 + 回撤
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[2, 1])
    for lab, r in [("ML long (net)", long_net), ("ML long-short (net)", ls_net), ("CSI500", bt["bench"])]:
        nav = (1 + r.fillna(0)).cumprod()
        a1.plot(nav.index, nav.values, label=lab, lw=1.5)
    a1.set_yscale("log"); a1.legend(); a1.grid(alpha=.3); a1.set_title("L4 net NAV vs CSI500 (cost=0.3%)")
    dd = (1 + long_net.fillna(0)).cumprod()
    dd = dd / dd.cummax() - 1
    a2.fill_between(dd.index, dd.values, 0, color="#C44E52", alpha=.5)
    a2.set_title("ML long drawdown"); a2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("results/09_backtest.png", bbox_inches="tight")
    bt.to_csv("results/09_backtest.csv")
    print("\n已保存 results/09_backtest.png")


if __name__ == "__main__":
    main()
