# -*- coding: utf-8 -*-
"""
研究脚本 41 —— 两个最终策略定稿(全期诚实口径) + 图
================================================================
① 微盘多头 = LGB-231多头 top-decile + 分行业中性(每行业内取前10%)
② 中证1000指增 = research/32(真实成分+换手惩罚, 行业中性已内置)
全期2018+(含2018熊市)口径, 扣换手。产出指标CSV + 两张最终图。

用法：/opt/anaconda3/bin/python research/41_finalize.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False
from quantlib import data, universe, altdata, backtest
PPY, C = 12, 0.003


def ind_neutral_long(panel, col="lgb"):
    rows, prev = {}, set()
    for dt, x in panel.dropna(subset=[col, "fwd_ret", "industry"]).groupby("trddt"):
        sel = x.groupby("industry", group_keys=False).apply(
            lambda d: d.nlargest(max(1, round(len(d) * 0.1)), col), include_groups=False)
        cur = set(sel["stkcd"]); to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows[dt] = sel["fwd_ret"].mean() - to * C; prev = cur
    return pd.Series(rows).sort_index()


def M(r, bench=None):
    r = pd.Series(r).dropna(); n = len(r)
    ann = (1 + r).prod() ** (PPY / n) - 1; vol = r.std(ddof=1) * np.sqrt(PPY)
    nav = (1 + r).cumprod(); mdd = (nav / nav.cummax() - 1).min()
    out = {"年化收益": ann, "年化波动": vol, "夏普": ann / vol, "最大回撤": mdd, "卡玛": ann / abs(mdd), "胜率": (r > 0).mean()}
    if bench is not None:
        ex = (r - bench.reindex(r.index)).dropna(); nx = (1 + ex).cumprod()
        out["超额"] = (1 + r).prod() ** (PPY / n) - (1 + bench.reindex(r.index)).prod() ** (PPY / n)
        out["IR"] = ex.mean() / ex.std(ddof=1) * np.sqrt(PPY); out["超额回撤"] = (nx / nx.cummax() - 1).min()
    return out


def main():
    t0 = time.time()
    panel = data.load_research_panel("M", "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel); panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    pred = pd.read_parquet("results/lgb_oos_pred.parquet"); pred["trddt"] = pred["trddt"].astype("datetime64[ns]")
    panel = panel.merge(pred, on=["stkcd", "trddt"], how="left")

    naive = pd.read_csv("results/26_nav.csv", index_col=0, parse_dates=True)["V1多头"].pct_change(fill_method=None).dropna()
    indn = ind_neutral_long(panel, "lgb")
    i500 = backtest.load_benchmark("000905", "M")
    m_naive = M(naive, i500); m_ind = M(indn.reindex(naive.index), i500)

    nav32 = pd.read_csv("results/32_nav.csv", index_col=0, parse_dates=True)
    prod = nav32["指增产品"].pct_change(fill_method=None).dropna()
    i1000 = backtest.load_benchmark("000852", "M")
    m_idx = M(prod, i1000)

    pd.set_option("display.unicode.east_asian_width", True)
    def fmt(d):
        return {k: (f"{v*100:.1f}%" if k in ("年化收益", "年化波动", "最大回撤", "超额", "超额回撤") else
                    f"{v*100:.0f}%" if k == "胜率" else round(v, 2)) for k, v in d.items()}
    tab = pd.DataFrame({"① 微盘多头·分行业中性(对500)": fmt(m_ind),
                        "② 中证1000指增(对1000)": fmt(m_idx),
                        "  [参照]微盘多头 naive": fmt(m_naive)}).T
    print("=" * 90, "\n两个最终策略 · 全期2018-2025 诚实口径(扣换手)\n", "=" * 90, sep="")
    print(tab.to_string())
    tab.to_csv("results/41_final_metrics.csv", encoding="utf-8-sig")

    n1 = pd.DataFrame({"分行业中性微盘多头": (1 + indn.reindex(naive.index)).cumprod(),
                       "naive微盘多头": (1 + naive).cumprod(),
                       "中证500": (1 + i500.reindex(naive.index)).cumprod()}).dropna()
    n1 = n1 / n1.iloc[0]
    fig, (a, a2) = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    for c, col, lw in [("分行业中性微盘多头", '#1baf7a', 2.5), ("naive微盘多头", '#2a78d6', 1.8), ("中证500", '#b4b2a9', 1.4)]:
        a.plot(n1.index, n1[c], label=c, color=col, lw=lw); a.annotate(f"{n1[c].iloc[-1]:.2f}×", (n1.index[-1], n1[c].iloc[-1]), fontsize=10, color=col, weight='bold')
        d = n1[c] / n1[c].cummax() - 1; a2.fill_between(n1.index, d, 0, color=col, alpha=0.3)
    a.set_title(f"最终策略① 微盘多头·分行业中性(2018-2025,样本外,扣换手)\n年化{m_ind['年化收益']:.0%} 夏普{m_ind['夏普']:.2f} 回撤{m_ind['最大回撤']:.0%} 卡玛{m_ind['卡玛']:.2f} · 行业暴露≈0", fontsize=11)
    a.set_ylabel("净值"); a.legend(loc="upper left", fontsize=10, frameon=False); a.grid(True, alpha=0.3); a2.set_ylabel("回撤"); a2.grid(True, alpha=0.3)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
    plt.tight_layout(); plt.savefig("results/41_micro_final.png", dpi=130)

    n2 = nav32.dropna(); n2 = n2 / n2.iloc[0]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for c, col, lw, ls in [("指增产品", '#2a78d6', 2.3, '-'), ("超额净值", '#1baf7a', 2.6, '-'), ("中证1000", '#b4b2a9', 1.5, '--')]:
        ax.plot(n2.index, n2[c], label=c, color=col, lw=lw, ls=ls); ax.annotate(f"{n2[c].iloc[-1]:.2f}×", (n2.index[-1], n2[c].iloc[-1]), fontsize=10, color=col, weight='bold')
    ax.set_title(f"最终策略② 中证1000指增(真实成分,2018-2025,样本外,扣换手)\n超额{m_idx['超额']:.1%} IR{m_idx['IR']:.2f} 超额回撤{m_idx['超额回撤']:.1%} 月胜率{m_idx['胜率']:.0%} 换手3.4x", fontsize=11)
    ax.set_ylabel("净值"); ax.legend(loc="upper left", fontsize=11, frameon=False); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("results/41_csi1000_final.png", dpi=130)
    print(f"\n图: results/41_micro_final.png, results/41_csi1000_final.png；完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
