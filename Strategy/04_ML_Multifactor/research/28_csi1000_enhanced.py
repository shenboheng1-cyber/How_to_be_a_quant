# -*- coding: utf-8 -*-
"""
研究脚本 28 —— 任务2：把小盘 alpha 做成「中证1000 指数增强」
================================================================
诊断:回撤主要是小盘 beta,既对冲不起(贴水13%)也分散不掉(多头都共享它)。
对策:不对冲、不在大盘指增——把小盘 beta 当基准接受,对标中证1000,只赛超额。
你的 alpha 本就是小盘选股,这是它的天然战场。看 超额/IR/超额回撤。
(无中证1000成分权重→用市值分档近似1000域;对比中证800指增 IR0.32)

用法：/opt/anaconda3/bin/python research/28_csi1000_enhanced.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import data, universe, backtest

FREQ, C, PPY = "M", 0.003, 12


def excess_stats(long_net, bench):
    ex = (long_net - bench.reindex(long_net.index)).dropna()
    te = ex.std(ddof=1) * np.sqrt(PPY); ir = ex.mean() / ex.std(ddof=1) * np.sqrt(PPY)
    nav = (1 + ex).cumprod(); mdd = (nav / nav.cummax() - 1).min()
    return {"超额年化": ex.mean() * PPY, "跟踪误差": te, "信息比IR": ir,
            "超额回撤": mdd, "月胜率": (ex > 0).mean()}


def long_net(panel, col):
    rows, prev = [], set()
    for dt, g in panel.dropna(subset=[col, "fwd_ret"]).groupby("trddt"):
        top = g.nlargest(max(1, len(g) // 10), col); cur = set(top["stkcd"])
        to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows.append({"dt": dt, "g": top["fwd_ret"].mean(), "to": to}); prev = cur
    L = pd.DataFrame(rows).set_index("dt"); return L["g"] - L["to"] * C


def main():
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    pred = pd.read_parquet("results/lgb_oos_pred.parquet"); pred["trddt"] = pred["trddt"].astype("datetime64[ns]")
    panel = panel.merge(pred, on=["stkcd", "trddt"], how="left")
    panel = panel[panel["lgb"].notna()].copy()

    # 市值分档近似中证1000域：每月按市值降序，取第 800–1800 名(剔除300+500大盘、剔最小微盘)
    panel["caprank"] = panel.groupby("trddt")["total_mktcap"].rank(ascending=False, method="first")
    mid = panel[(panel["caprank"] > 800) & (panel["caprank"] <= 1800)].copy()

    i1000 = backtest.load_benchmark("000852", FREQ)
    i800 = backtest.load_benchmark("000906", FREQ)

    rA = long_net(panel, "lgb")                    # 全市场小盘选股
    rB = long_net(mid, "lgb")                       # 限定中证1000域

    pd.set_option("display.unicode.east_asian_width", True)
    res = {
        "A 全市场多头 对标中证1000": excess_stats(rA, i1000),
        "B 中证1000域多头 对标中证1000": excess_stats(rB, i1000),
        "[参照] 全市场多头 对标中证800": excess_stats(rA, i800),
    }
    out = pd.DataFrame(res).T
    for c in ["超额年化", "跟踪误差", "超额回撤"]:
        out[c] = (out[c] * 100).round(1).astype(str) + "%"
    out["信息比IR"] = out["信息比IR"].round(2); out["月胜率"] = (out["月胜率"] * 100).round(0).astype(int).astype(str) + "%"
    print("\n" + "=" * 76, "\n任务2：小盘 alpha 做成中证1000指增（OOS，扣0.3%换手；对比中证800指增IR0.32）\n", "=" * 76, sep="")
    print(out.to_string())
    # 净值(指增口径:基准+超额) 供画图
    ex = (rB - i1000.reindex(rB.index)).dropna()
    print(f"\n中证1000域版: 超额夏普(=IR) {res['B 中证1000域多头 对标中证1000']['信息比IR']:.2f}，"
          f"超额回撤 {res['B 中证1000域多头 对标中证1000']['超额回撤']:.1%}")
    pd.DataFrame({"中证1000": (1 + i1000.reindex(rB.index)).cumprod(),
                  "指增(中证1000域)": (1 + rB).cumprod(),
                  "超额净值": (1 + ex).cumprod()}).to_csv("results/28_nav.csv", encoding="utf-8-sig")
    out.to_csv("results/28_csi1000.csv", encoding="utf-8-sig")
    print("已保存 results/28_*.csv")


if __name__ == "__main__":
    main()
