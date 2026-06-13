"""回测报告输出: 净值曲线 vs 中证全指、回撤图、逐年表 (复现 图12/表7)。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import yearly_table

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def make_report(nav: pd.Series, benchmark_nav: pd.Series,
                turnover: pd.Series, outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    strat = nav / nav.iloc[0]
    bench = benchmark_nav.reindex(nav.index).ffill()
    bench = bench / bench.iloc[0]
    excess = strat / bench

    fig, ax = plt.subplots(figsize=(12, 5))
    strat.plot(ax=ax, label="风格轮动策略")
    bench.plot(ax=ax, label="中证全指")
    excess.plot(ax=ax, label="超额收益")
    ax.legend(); ax.set_title("策略 VS 中证全指净值 (复现图12)")
    fig.savefig(outdir / "nav_vs_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    dd = strat / strat.cummax() - 1
    fig, ax = plt.subplots(figsize=(12, 3))
    dd.plot(ax=ax, color="darkred"); ax.set_title("策略回撤")
    fig.savefig(outdir / "drawdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    table = yearly_table(nav, turnover)
    table.to_csv(outdir / "yearly_performance.csv", encoding="utf-8-sig")
    print(table.round(4))
