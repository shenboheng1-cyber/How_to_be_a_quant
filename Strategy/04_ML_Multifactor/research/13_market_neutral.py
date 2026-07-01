# -*- coding: utf-8 -*-
"""
研究脚本 13 —— 市场中性（股指期货对冲）
================================================================
用 L4 多头净值流，叠加 IC(中证500)/IM(中证1000) 期货对冲(按 beta)，
扣贴水成本情景，给出市场中性净业绩 + 杠杆版本。

中性收益 ≈ 多头净 − beta×指数 − beta×年化贴水/12

用法：python research/13_market_neutral.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import backtest

FREQ = "M"; C = 0.003


def main():
    bt = pd.read_csv("results/09_backtest.csv", index_col=0, parse_dates=True)
    long_net = bt["long_g"] - bt["to_l"] * C
    i500 = bt["bench"]                                              # 中证500 未来收益(已对齐)
    i1000 = backtest.load_benchmark("000852", FREQ).reindex(bt.index)  # 中证1000

    def beta(idx):
        d = pd.concat([long_net, idx], axis=1).dropna()
        return float(np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0])

    pd.set_option("display.unicode.east_asian_width", True)
    print("参照：")
    for nm, r in [("多头(扛市场)", long_net), ("中证500", i500), ("中证1000", i1000)]:
        m = backtest.metrics(r, FREQ)
        print(f"  {nm:<12} 年化{m['年化']:>7.1%} 夏普{m['夏普']:>5.2f} 回撤{m['最大回撤']:>6.0%}")

    rows = {}
    for name, idx in [("IC对冲(中证500)", i500), ("IM对冲(中证1000)", i1000)]:
        b = beta(idx)
        for basis in [0.0, 0.03, 0.06, 0.09]:
            neu = long_net - b * idx - b * basis / 12
            m = backtest.metrics(neu, FREQ)
            rows[f"{name} 贴水{basis:.0%}"] = {"beta": round(b, 2), "净年化": m["年化"],
                                              "净夏普": m["夏普"], "波动": m["波动"], "最大回撤": m["最大回撤"]}

    print("\n" + "=" * 74, "\n市场中性（按beta对冲，不同贴水情景）\n", "=" * 74, sep="")
    print(pd.DataFrame(rows).T.to_string())

    # 杠杆版：IM对冲、贴水6%、加2倍杠杆(扣4%融资)
    b = beta(i1000)
    neu = long_net - b * i1000 - b * 0.06 / 12
    lev = 2 * neu - 0.04 / 12
    m = backtest.metrics(lev, FREQ)
    print(f"\n杠杆示例：IM对冲+贴水6% 的中性组合 加2倍杠杆(扣4%融资) → "
          f"年化{m['年化']:.1%} 夏普{m['夏普']:.2f} 回撤{m['最大回撤']:.0%}")

    pd.DataFrame(rows).T.to_csv("results/13_market_neutral.csv", encoding="utf-8-sig")
    print("\n已保存 results/13_market_neutral.csv")


if __name__ == "__main__":
    main()
