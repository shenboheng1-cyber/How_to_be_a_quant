# -*- coding: utf-8 -*-
"""
研究脚本 12 —— 质量筛选救多头（结构化使用基本面）
================================================================
同一个 ML 信号，构建多头时先【硬剔除】底部质量 + 有违规的垃圾股，再取 top decile。
对比不同剔除力度下的 多头净夏普/回撤/IR——验证"质量当筛选"能否真降回撤。

用法：DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib python research/12_quality_screen.py
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
    cache["trddt"] = cache["trddt"].astype("datetime64[ns]")
    base_cols = [c for c in cache.columns if c not in ("stkcd", "trddt", "y")]

    print("重建质量分 + 违规标记 ...")
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = fundamentals.attach(panel)
    panel = events.attach_events(panel)
    # 质量分 = ROE / 低应计 / 盈余含金量 / 低盈余操纵 的标准化均值（越大越干净）
    qparts = [
        preprocess.preprocess_factor(panel, fundamentals.roe(panel), do_neutralize=True),
        preprocess.preprocess_factor(panel, fundamentals.accruals(panel), do_neutralize=True),
        preprocess.preprocess_factor(panel, fundamentals.gross_cfo(panel), do_neutralize=True),
        preprocess.preprocess_factor(panel, events.earn_mgmt(panel), do_neutralize=True),
    ]
    qual = pd.concat(qparts, axis=1).mean(axis=1)
    qdf = pd.DataFrame({"stkcd": panel["stkcd"].values,
                        "trddt": panel["trddt"].astype("datetime64[ns]").values,
                        "qual": qual.values, "viol": panel["viol_count"].values})

    # ML 信号（基线 231）
    print("跑 ML 信号 ...")
    pred = ml.walk_forward_predict(cache[base_cols].values, cache["y"].values,
                                   cache["trddt"].values, ml.lgb_model(), init=36, embargo=1, step=3)
    oos = ~np.isnan(pred)

    # 回测面板（对齐 cache 行序）+ 质量分
    bp = data.load_research_panel(FREQ, START, END)
    pf = cache[["stkcd", "trddt"]].merge(
        bp[["stkcd", "trddt", "fwd_ret", "total_mktcap"]].assign(
            trddt=lambda d: d["trddt"].astype("datetime64[ns]")), on=["stkcd", "trddt"], how="left")
    pf = pf.merge(qdf, on=["stkcd", "trddt"], how="left")
    pf["sig"] = pred
    pf["qual"] = pf["qual"].fillna(pf["qual"].median())   # 缺质量→中性(不因缺失剔除)
    po = pf[oos].reset_index(drop=True)
    bench = backtest.load_benchmark("000905", FREQ, START, END)

    def run(screen):
        """screen=底部质量剔除比例；并硬剔有违规。返回多头净指标。"""
        d = po.copy()
        if screen > 0:
            thr = d.groupby("trddt")["qual"].transform(lambda s: s.quantile(screen))
            keep = (d["qual"] >= thr) & (d["viol"].fillna(0) == 0)
        else:
            keep = pd.Series(True, index=d.index)
        sub = d[keep].reset_index(drop=True)
        bt = backtest.backtest(sub, sub["sig"].values, bench, cost=0.0)
        ln = bt["long_g"] - bt["to_l"] * C
        m = backtest.metrics(ln, FREQ)
        return {"多头净年化": m["年化"], "多头净夏普": m["夏普"], "多头回撤": m["最大回撤"],
                "波动": m["波动"], "对500_IR": backtest.info_ratio(ln - bt["bench"], FREQ),
                "年化换手": round(bt["to_l"].mean() * 12, 1), "保留股均数": int(keep.sum() / d["trddt"].nunique())}

    rows = {}
    for s, name in [(0.0, "基线(无筛选)"), (0.2, "剔底20%质量"), (0.3, "剔底30%质量"), (0.4, "剔底40%质量")]:
        rows[name] = run(s)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n组装+回测 {time.time()-t0:.0f}s")
    print("=" * 78, "\n质量筛选救多头：硬剔垃圾股 + 违规 后的多头\n", "=" * 78, sep="")
    print(pd.DataFrame(rows).T.to_string())
    pd.DataFrame(rows).T.to_csv("results/12_quality_screen.csv", encoding="utf-8-sig")
    print("\n已保存 results/12_quality_screen.csv")


if __name__ == "__main__":
    main()
