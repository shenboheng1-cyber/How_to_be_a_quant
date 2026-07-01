# -*- coding: utf-8 -*-
"""
研究脚本 11 —— A/B 对比：加基本面+事件因子前后的 IR 与回撤
================================================================
基线(231 价量/微观因子) vs 增强(+11基本面+4事件)，同一条防泄漏 walk-forward + L4 回测，
对比多头净夏普 / 对中证500 的 IR / 最大回撤——直接回答"新数据能不能救多头"。

用法：DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib python research/11_augmented_compare.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import data, universe, preprocess, ml, backtest, fundamentals, events

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
C = 0.003


def main():
    t0 = time.time()
    cache = pd.read_parquet("results/08_features.parquet")
    base_cols = [c for c in cache.columns if c not in ("stkcd", "trddt", "y")]
    print(f"基线特征 {len(base_cols)} 个")

    # 重建面板 → 基本面 + 事件因子
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = fundamentals.attach(panel)
    panel = events.attach_events(panel)
    new = {}
    for nm, (fn, _) in fundamentals.REGISTRY.items():
        new["fund_" + nm] = preprocess.preprocess_factor(panel, fn(panel), do_neutralize=True).values
    for nm, (fn, _) in events.REGISTRY.items():
        new["evt_" + nm] = preprocess.preprocess_factor(panel, fn(panel), do_neutralize=True).values
    ndf = pd.DataFrame({"stkcd": panel["stkcd"].values,
                        "trddt": panel["trddt"].astype("datetime64[ns]").values, **new})
    cache["trddt"] = cache["trddt"].astype("datetime64[ns]")
    cache = cache.merge(ndf, on=["stkcd", "trddt"], how="left")
    new_cols = list(new.keys())
    aug_cols = base_cols + new_cols
    print(f"增强特征 {len(aug_cols)} 个（+{len(new_cols)} 新）| 组装 {time.time()-t0:.0f}s")

    # 回测用面板(fwd_ret/市值)，对齐 cache 行序
    bp = data.load_research_panel(FREQ, START, END)
    base_full = cache[["stkcd", "trddt"]].merge(
        bp[["stkcd", "trddt", "fwd_ret", "total_mktcap"]].assign(
            trddt=lambda d: d["trddt"].astype("datetime64[ns]")),
        on=["stkcd", "trddt"], how="left")
    bench = backtest.load_benchmark("000905", FREQ, START, END)
    y, dates = cache["y"].values, cache["trddt"].values

    results = {}
    for label, cols in [("基线(231)", base_cols), ("+基本面&事件(246)", aug_cols)]:
        print(f"\n跑 {label} ...")
        pred = ml.walk_forward_predict(cache[cols].values, y, dates, ml.lgb_model(),
                                       init=36, embargo=1, step=3)
        oos = ~np.isnan(pred)
        po = base_full[oos].reset_index(drop=True)
        bt = backtest.backtest(po, pred[oos], bench, cost=0.0)
        long_net = bt["long_g"] - bt["to_l"] * C
        ls_net = bt["ls_g"] - bt["to_ls"] * C
        m = backtest.metrics(long_net, FREQ)
        results[label] = {"多头净年化": m["年化"], "多头净夏普": m["夏普"], "多头回撤": m["最大回撤"],
                          "对500_IR": backtest.info_ratio(long_net - bt["bench"], FREQ),
                          "多空净夏普": backtest.metrics(ls_net, FREQ)["夏普"],
                          "多空回撤": backtest.metrics(ls_net, FREQ)["最大回撤"]}

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 66, "\nA/B 对比：加新数据前后\n", "=" * 66, sep="")
    cmp = pd.DataFrame(results).T
    print(cmp.to_string())
    os.makedirs("results", exist_ok=True)
    cmp.to_csv("results/11_augmented_compare.csv", encoding="utf-8-sig")
    print("\n已保存 results/11_augmented_compare.csv")


if __name__ == "__main__":
    main()
