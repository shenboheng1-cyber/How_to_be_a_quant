# -*- coding: utf-8 -*-
"""
研究脚本 14 —— 因子择时/regime：削减市场中性的 alpha 回撤
================================================================
对市场中性组合(IM对冲)叠加 波动目标 + 拥挤度降仓，看能否削掉 2024 风格踩踏回撤。

用法：python research/14_regime_timing.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import data, universe, preprocess, evaluate, backtest, regime
from quantlib.factors import classic

FREQ = "M"; C = 0.003


def main():
    # 市场中性收益流(IM对冲, 贴水6%)
    bt = pd.read_csv("results/09_backtest.csv", index_col=0, parse_dates=True)
    long_net = bt["long_g"] - bt["to_l"] * C
    im = backtest.load_benchmark("000852", FREQ).reindex(bt.index)
    d = pd.concat([long_net, im], axis=1).dropna()
    beta = float(np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0])
    neutral = (long_net - beta * im - beta * 0.06 / 12).dropna()

    # 拥挤度：6个经典因子的多空收益两两相关
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    frets = {}
    for k in ["reversal", "low_turnover", "low_vol", "max_ret", "size", "ep"]:
        f = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=True)
        qr = evaluate.quantile_returns(panel, f, 10)
        frets[k] = (qr.iloc[:, -1] - qr.iloc[:, 0])
    fr = pd.DataFrame(frets); fr.index = pd.to_datetime(fr.index)
    crowd = regime.crowding_index(fr, lookback=12).reindex(neutral.index)

    # 叠加
    s_vol = regime.vol_target(neutral, target_ann=0.12, lookback=6, cap=1.5)
    timed_vol = s_vol * neutral
    expo_cr = regime.derisk(neutral, crowd, hi_quantile=0.8, low_expo=0.5)
    timed_cr = expo_cr * neutral
    timed_both = s_vol * expo_cr * neutral

    def m(r):
        r = r.dropna(); n = (1 + r).cumprod()
        return {"年化": round((1 + r).prod() ** (12 / len(r)) - 1, 4),
                "夏普": round(r.mean() / r.std(ddof=1) * np.sqrt(12), 2),
                "最大回撤": round((n / n.cummax() - 1).min(), 3),
                "2024回撤": round(((lambda x: (x / x.cummax() - 1).min())((1 + r["2024-01":"2024-12"]).cumprod())), 3)}

    pd.set_option("display.unicode.east_asian_width", True)
    print("拥挤度峰值时点:", crowd.idxmax().date(), "值", round(crowd.max(), 2),
          "| 2024年初拥挤度:", round(crowd["2024-01":"2024-03"].mean(), 2),
          "| 全期均值:", round(crowd.mean(), 2))
    print("\n" + "=" * 60, "\n市场中性 + 择时叠加\n", "=" * 60, sep="")
    res = pd.DataFrame({"中性(基线)": m(neutral), "+波动目标": m(timed_vol),
                        "+拥挤度降仓": m(timed_cr), "+两者": m(timed_both)}).T
    print(res.to_string())
    res.to_csv("results/14_regime_timing.csv", encoding="utf-8-sig")
    print("\n已保存 results/14_regime_timing.csv")


if __name__ == "__main__":
    main()
