"""生成报告用回测图：净值曲线、回撤曲线、滚动12M Sharpe、分年度收益、容量曲线、累计IC、相关热图。
全部后复权市价、V2(净5bps)。输出 -> figures/"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, backtest_monthly_strategy,
    make_monthly_weights_v2, FACTOR_WEIGHTS_V2)
import hfq_common as H

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti TC", "STHeiti", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
FIG = H.ROOT / "figures"
FIG.mkdir(exist_ok=True)
LAM = 0.4


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, amt, _ = H.load_hfq()
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    market = H.hs300()
    fac = compute_factor_panel(px)
    scored = score_factors_with_weights(fac, FACTOR_WEIGHTS_V2, score_column="risk_adjusted_score")

    w = make_monthly_weights_v2(scored, px, uni, top_n=20, max_per_theme=3, max_weight=0.12,
                                buffer_rank=35, volatility_target=0.18)
    eq, eff = backtest_monthly_strategy(px, w, transaction_cost_bps=0.0, rebalance_lambda=LAM)
    g = eq.copy(); g["date"] = pd.to_datetime(g["date"]); g = g.set_index("date").loc[H.START:H.END]
    ret = g["strategy_return"] - g["turnover"] * 5 / 1e4
    nav = (1 + ret).cumprod()

    wb = H.equal_weight_basket(px)
    eqb, _ = backtest_monthly_strategy(px, wb, transaction_cost_bps=0.0)
    rb = H.to_ret(eqb); navb = (1 + rb).cumprod()
    hs = market.loc[H.START:H.END]; navh = hs / hs.iloc[0]

    # 1. 净值曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nav.index, nav.values, label="ETF多因子 V2(净5bps)", lw=2, color="#c0392b")
    ax.plot(navb.index, navb.values, label="全市场等权ETF篮子", lw=1.3, color="#2980b9", alpha=0.8)
    ax.plot(navh.index, navh.values, label="沪深300", lw=1.3, color="#7f8c8d", alpha=0.8)
    ax.set_title("净值曲线（后复权市价，含成本，2018-01~2026-06）")
    ax.set_ylabel("净值（起点=1）"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "01_nav.png", dpi=130); plt.close(fig)

    # 2. 回撤曲线
    dd = nav / nav.cummax() - 1.0
    ddh = navh / navh.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(dd.index, dd.values * 100, 0, color="#c0392b", alpha=0.5, label="策略 V2")
    ax.plot(ddh.index, ddh.values * 100, color="#7f8c8d", lw=1, label="沪深300")
    ax.set_title("回撤曲线（策略最大回撤 -7.6% vs 沪深300 -45.6%）")
    ax.set_ylabel("回撤 %"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "02_drawdown.png", dpi=130); plt.close(fig)

    # 3. 滚动12M Sharpe
    roll = ret.rolling(H.TD)
    rs = (roll.mean() / roll.std(ddof=0) * np.sqrt(H.TD)).dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rs.index, rs.values, color="#c0392b", lw=1.3)
    ax.axhline(0, color="k", lw=0.8); ax.axhline(1, color="#27ae60", ls="--", lw=0.8, label="Sharpe=1")
    ax.fill_between(rs.index, rs.values, 0, where=(rs.values < 0), color="#e74c3c", alpha=0.3)
    ax.set_title(f"12个月滚动 Sharpe（中位 {rs.median():.2f}，为负约占 {(rs<0).mean():.0%}）")
    ax.set_ylabel("滚动 Sharpe"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "03_rolling_sharpe.png", dpi=130); plt.close(fig)

    # 4. 分年度收益
    yr = pd.DataFrame({"策略V2": ret, "等权篮子": rb.reindex(ret.index).fillna(0),
                       "沪深300": hs.pct_change().reindex(ret.index).fillna(0)})
    ann = yr.groupby(yr.index.year).apply(lambda x: (1 + x).prod() - 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ann.plot(kind="bar", ax=ax, color=["#c0392b", "#2980b9", "#7f8c8d"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("分年度收益"); ax.set_ylabel("年收益"); ax.set_xlabel("")
    ax.set_yticklabels([f"{y:.0%}" for y in ax.get_yticks()]); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "04_annual.png", dpi=130); plt.close(fig)

    # 5. 容量曲线
    aums = [0.1, 0.5, 1, 3, 5, 10]
    caps = [H.metrics(H.apply_impact(g["strategy_return"], g["turnover"], eff, px, amt, a * 1e8))["sharpe"]
            for a in aums]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(aums, caps, "o-", color="#c0392b", lw=2)
    for a, c in zip(aums, caps):
        ax.annotate(f"{c:.2f}", (a, c), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.axhline(1, color="#27ae60", ls="--", lw=0.8, label="Sharpe=1")
    ax.set_xscale("log"); ax.set_xticks(aums); ax.set_xticklabels([f"{a}亿" for a in aums])
    ax.set_title("容量曲线：净 Sharpe vs 资金规模（平方根冲击模型）")
    ax.set_xlabel("资金规模"); ax.set_ylabel("净 Sharpe"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "05_capacity.png", dpi=130); plt.close(fig)

    # 6. 累计 Rank IC（最终5因子）
    try:
        cic = pd.read_csv(H.ROOT / "outputs_factor_diag/cumulative_ic.csv", index_col=0, parse_dates=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in cic.columns:
            ax.plot(cic.index, cic[col], label=col, lw=1.4)
        ax.set_title("累计 Rank IC（最终5因子，后复权市价，月度）")
        ax.set_ylabel("累计 IC"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "06_cumulative_ic.png", dpi=130); plt.close(fig)
    except FileNotFoundError:
        print("跳过累计IC图（先跑 factor_diagnostics.py）")

    # 7. 因子相关热图
    try:
        corr = pd.read_csv(H.ROOT / "outputs_factor_diag/factor_corr_spearman.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(corr.index, fontsize=8)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, fraction=0.046); ax.set_title("因子相关矩阵（Spearman）")
        fig.tight_layout(); fig.savefig(FIG / "07_corr_heatmap.png", dpi=130); plt.close(fig)
    except FileNotFoundError:
        print("跳过相关热图")

    print(f"图已生成 -> {FIG}")
    for p in sorted(FIG.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
