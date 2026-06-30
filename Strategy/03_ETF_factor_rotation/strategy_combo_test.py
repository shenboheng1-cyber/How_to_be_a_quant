"""换思路:策略层分散。造一个与 V3(趋势)低相关的短期反转 ETF 子策略,合并看 Sharpe。
V3 趋势 + 反转 子策略 → 相关性 + 各权重合并 Sharpe。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_monthly_weights_v2,
    backtest_monthly_strategy, FACTOR_WEIGHTS_V2, _datewise_z, _month_end_dates)
import hfq_common as H

COST, LAM = 5.0, 0.4


def reversal_scored(px, lookback):
    """短期反转打分 = z(-近 lookback 日收益)。买跌得多的。"""
    rev = -(px / px.shift(lookback) - 1.0)
    long = rev.stack(future_stack=True).rename("rev").reset_index()
    long.columns = ["date", "fund_code", "rev"]
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    long["score"] = _datewise_z(long, "rev")
    return long.dropna(subset=["score"])


def run(scored, px, uni):
    w = make_monthly_weights_v2(scored, px, uni, buffer_rank=35, volatility_target=0.18, **H.PORT)
    eq, _ = backtest_monthly_strategy(px, w, transaction_cost_bps=COST, rebalance_lambda=LAM)
    return H.to_ret(eq)


def combo_sharpe(r1, r2, w1):
    r = (w1 * r1 + (1 - w1) * r2).dropna()
    return H.metrics(r)


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, _, _ = H.load_hfq(); px = px.loc[:H.END]
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    fac = compute_factor_panel(px)

    # V3 趋势
    sc_v3 = score_factors_with_weights(fac, FACTOR_WEIGHTS_V2, score_column="s")
    r_v3 = run(sc_v3, px, uni)

    print("=== 各反转窗口的子策略 + 与 V3 相关性 ===")
    print(f"{'子策略':16}{'年化':>8}{'Sharpe':>8}{'MDD':>8}{'与V3相关':>9}")
    print(f"{'V3趋势(基准)':16}{H.metrics(r_v3)['ann']*100:>7.1f}%{H.metrics(r_v3)['sharpe']:>8.2f}{H.metrics(r_v3)['mdd']*100:>7.0f}%{1.0:>9.2f}")
    revs = {}
    for lb in (5, 10, 20, 60):
        r = run(reversal_scored(px, lb), px, uni)
        revs[lb] = r
        m = H.metrics(r); rho = r_v3.corr(r.reindex(r_v3.index))
        print(f"{'反转'+str(lb)+'日':16}{m['ann']*100:>7.1f}%{m['sharpe']:>8.2f}{m['mdd']*100:>7.0f}%{rho:>9.2f}")

    # 选相关性最低的反转窗口,做合并扫描
    best_lb = min(revs, key=lambda lb: r_v3.corr(revs[lb].reindex(r_v3.index)))
    r_rev = revs[best_lb]
    rho = r_v3.corr(r_rev.reindex(r_v3.index))
    print(f"\n=== V3趋势 + 反转{best_lb}日 (相关{rho:.2f}) 合并 ===")
    print(f"{'权重(V3/反转)':16}{'年化':>8}{'Sharpe':>8}{'MDD':>8}")
    for w1 in (1.0, 0.8, 0.7, 0.6, 0.5, 0.4):
        m = combo_sharpe(r_v3, r_rev, w1)
        print(f"{f'{w1:.0%}/{1-w1:.0%}':16}{m['ann']*100:>7.1f}%{m['sharpe']:>8.2f}{m['mdd']*100:>7.0f}%")

    # 理论上限对照
    s = np.sqrt(2 / (1 + max(rho, -0.99)))
    print(f"\n理论(等权、同Sharpe、ρ={rho:.2f}): 合并Sharpe ≈ 单策略×{s:.2f}")


if __name__ == "__main__":
    main()
