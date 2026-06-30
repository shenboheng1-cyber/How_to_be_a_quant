# -*- coding: utf-8 -*-
"""
quantlib.plotting —— 结果可视化
================================================================
用英文标签（便于放 GitHub 作品集，也避免中文字体坑）。

不强制 matplotlib 后端：在脚本里照常 savefig，在 notebook 里 `%matplotlib inline`
即可内联显示。每个函数都【返回 fig】，传 save_path 才落盘，不主动 close。
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3})


def plot_ic_series(ic: pd.Series, title: str = "IC", save_path: str | None = None):
    """单因子 IC 时序：柱=每期 IC，红线=累计 IC（看预测力是否持续累积）。"""
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(ic.index, ic.values, width=20, color="#4C72B0", alpha=0.7, label="IC each period")
    ax.axhline(0, color="k", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(ic.index, ic.cumsum().values, color="#C44E52", lw=1.8, label="Cumulative IC")
    ax2.grid(False)
    ax.set_title(f"{title}  (mean={ic.mean():.3f}, ICIR={ic.mean()/ic.std():.2f})")
    ax.set_ylabel("IC"); ax2.set_ylabel("Cumulative IC")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_quantile_bars(qsummary: pd.DataFrame, title: str = "Quantile annualized return",
                       save_path: str | None = None):
    """分层组合的年化收益柱状图：看是否【单调】（因子有效最直观的证据）。"""
    qrows = [i for i in qsummary.index if str(i).startswith("Q")]
    vals = qsummary.loc[qrows, "年化收益"].values
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = ["#C44E52" if v < 0 else "#55A868" for v in vals]
    ax.bar(range(len(qrows)), vals * 100, color=colors, alpha=0.85)
    ax.set_xticks(range(len(qrows))); ax.set_xticklabels(qrows)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Annualized return (%)")
    ax.set_title(title + "  (Q1=low factor → QN=high factor)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_factor_comparison(summary: pd.DataFrame, ls_navs: dict,
                           save_path: str | None = None):
    """左：各因子 RankIC / ICIR 柱状；右：top 因子多空净值曲线。

    summary : 含列 ['代码','RankIC','ICIR'] 的 DataFrame
    ls_navs : {factor_code: pd.Series(多空净值, index=trddt)}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    s = summary.sort_values("ICIR")
    codes = s["代码"].values
    y = range(len(codes))
    ax1.barh(list(y), s["ICIR"].values, color="#4C72B0", alpha=0.85, label="ICIR")
    ax1.barh(list(y), s["RankIC"].values * 5, color="#DD8452", alpha=0.6,
             label="RankIC ×5")
    ax1.set_yticks(list(y)); ax1.set_yticklabels(codes)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_title("Factor strength: ICIR & RankIC (×5)")
    ax1.legend(loc="lower right")

    for code, nav in ls_navs.items():
        ax2.plot(nav.index, nav.values, label=code, lw=1.4)
    ax2.set_title("Long-short cumulative NAV (top factors)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_yscale("log")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
