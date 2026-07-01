# -*- coding: utf-8 -*-
"""
研究脚本 37 —— 最终版纯多头：分行业中性 + CDaR 减震
================================================================
主腿 = 分行业中性多头(每行业内取LGB alpha前10%,组合行业分布≈全市场,行业暴露≈0)。
再和 低波防御腿 + 现金 三腿,用 research/35 的 min-CDaR(收益地板)LP 滚动配权减震。
对比: naive多头 / 分行业中性多头 / 分行业中性+CDaR。全部同期(CDaR活跃期)公平比。

用法：/opt/anaconda3/bin/python research/37_final_long.py
"""
import sys, os, time, importlib.util, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import data, universe, altdata
from quantlib.backtest import load_benchmark

# 复用 research/35 的 CDaR LP + 滚动配置(文件名非法标识符,用 importlib 载入)
spec = importlib.util.spec_from_file_location("cdar", os.path.join(os.path.dirname(__file__), "35_cdar_cvar_alloc.py"))
cdar = importlib.util.module_from_spec(spec); spec.loader.exec_module(cdar)
C, ALLOC_C, PPY = cdar.C, cdar.ALLOC_C, cdar.PPY


def ind_neutral_long(panel, col="lgb", cost=C):
    """分行业中性多头:每行业内取 col 前10% 等权。返回月度净收益(扣个股换手)。"""
    rows, prev = {}, set()
    g = panel.dropna(subset=[col, "fwd_ret", "industry"])
    for dt, x in g.groupby("trddt"):
        sel = x.groupby("industry", group_keys=False).apply(
            lambda d: d.nlargest(max(1, round(len(d) * 0.1)), col), include_groups=False)
        cur = set(sel["stkcd"]); to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows[dt] = sel["fwd_ret"].mean() - to * cost; prev = cur
    return pd.Series(rows).sort_index()


def mfull(r):
    r = pd.Series(r).dropna(); n = len(r); ann = (1 + r).prod() ** (PPY / n) - 1
    vol = r.std(ddof=1) * np.sqrt(PPY); nav = (1 + r).cumprod(); mdd = (nav / nav.cummax() - 1).min()
    return {"年化": ann, "波动": vol, "夏普": ann / vol, "最大回撤": mdd, "卡玛": ann / abs(mdd), "胜率": (r > 0).mean()}


def main():
    t0 = time.time()
    panel = data.load_research_panel("M", "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    pred = pd.read_parquet("results/lgb_oos_pred.parquet"); pred["trddt"] = pred["trddt"].astype("datetime64[ns]")
    panel = panel.merge(pred, on=["stkcd", "trddt"], how="left")
    print(f"面板就绪 {time.time()-t0:.0f}s，构建子腿 ...", flush=True)

    r_naive = cdar.long_decile(panel, "lgb", largest=True)             # naive 全局多头
    r_ind = ind_neutral_long(panel, "lgb")                             # 分行业中性主腿
    r_lowvol = cdar.long_decile(panel[panel["lgb"].notna()], "vol_60", largest=False)  # 低波腿
    idx = r_ind.index
    for s in (r_naive, r_lowvol):
        pass
    r_naive = r_naive.reindex(idx); r_lowvol = r_lowvol.reindex(idx)
    cash = pd.Series(0.0, index=idx)

    legs = pd.DataFrame({"行业中性主腿": r_ind, "低波腿": r_lowvol, "现金": cash}).dropna()
    print(f"子腿就绪，CDaR 滚动配权(扫 floor) ...", flush=True)

    results, navs = {}, {}
    for fl in [0.14, 0.16, 0.18]:
        port, w = cdar.rolling_alloc(legs, cdar.solve_cdar_exact, window=36, min_train=24,
                                     mode="min_risk", ret_floor=fl / PPY, alpha=0.05)
        results[f"分行业中性+CDaR(floor{int(fl*100)}%)"] = (port, w["行业中性主腿"].mean())
    # CDaR 活跃期,用来公平对比
    active = list(results.values())[0][0].index
    tab = {}
    tab["naive 全局多头"] = mfull(r_naive.reindex(active))
    tab["分行业中性多头(无减震)"] = mfull(r_ind.reindex(active))
    for k, (port, mw) in results.items():
        m = mfull(port); m["主腿平均权重"] = mw; tab[k] = m
    out = pd.DataFrame(tab).T
    for c in ["年化", "波动", "最大回撤"]: out[c] = (out[c] * 100).round(1).astype(str) + "%"
    out["夏普"] = out["夏普"].round(2); out["卡玛"] = out["卡玛"].round(2)
    out["胜率"] = (out["胜率"] * 100).round(0).astype(int).astype(str) + "%"
    if "主腿平均权重" in out: out["主腿平均权重"] = out["主腿平均权重"].apply(lambda v: f"{v*100:.0f}%" if pd.notna(v) else "—")
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\nCDaR 活跃期: {pd.Timestamp(active[0]).date()} ~ {pd.Timestamp(active[-1]).date()} (前24月预热)")
    print("=" * 84, "\n最终版纯多头 · 分行业中性 + CDaR 减震(同期公平对比,扣换手)\n", "=" * 84, sep="")
    print(out[["年化", "波动", "夏普", "最大回撤", "卡玛", "胜率", "主腿平均权重"]].to_string())

    best = "分行业中性+CDaR(floor14%)"
    port = results[best][0]
    pd.DataFrame({"最终版(行业中性+CDaR)": (1 + port).cumprod(),
                  "分行业中性多头": (1 + r_ind.reindex(active)).cumprod(),
                  "naive多头": (1 + r_naive.reindex(active)).cumprod()}).to_csv("results/37_nav.csv", encoding="utf-8-sig")
    print(f"\n净值存 results/37_nav.csv；完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
