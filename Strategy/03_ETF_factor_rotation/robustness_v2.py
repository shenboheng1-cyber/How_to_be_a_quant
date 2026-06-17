"""V2 一致口径的稳健性数字：分年度/去最强年/滚动Sharpe/bootstrap/成本sweep + 基准。
全部基于最终版 V2(去弱市、vol-target0.18、buffer35、lam0.4)，后复权市价。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, backtest_monthly_strategy, FACTOR_WEIGHTS_V2)
import hfq_common as H

OUT = H.ROOT / "outputs_robustness_v2"
OUT.mkdir(exist_ok=True)
np.random.seed(20260617)
LAM = 0.4


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, amt, _ = H.load_hfq()
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    market = H.hs300()
    fac = compute_factor_panel(px)
    scored = score_factors_with_weights(fac, FACTOR_WEIGHTS_V2, score_column="risk_adjusted_score")

    from etf_factor_strategy.engine import make_monthly_weights_v2
    w = make_monthly_weights_v2(scored, px, uni, buffer_rank=35, volatility_target=0.18, **H.PORT)
    eq, _ = backtest_monthly_strategy(px, w, transaction_cost_bps=0.0, rebalance_lambda=LAM)
    g = eq.copy(); g["date"] = pd.to_datetime(g["date"]); g = g.set_index("date").loc[H.START:H.END]
    ret, turn = g["strategy_return"], g["turnover"]
    net5 = ret - turn * 5 / 1e4

    # 基准
    wb = H.equal_weight_basket(px)
    eqb, _ = backtest_monthly_strategy(px, wb, transaction_cost_bps=0.0)
    basket = H.to_ret(eqb)
    hs = market.loc[H.START:H.END].pct_change().dropna()

    full = H.metrics(net5, "V2 净5bps")
    print(f"V2(净5bps) 全期: 年化{full['ann']:+.1%} Sharpe{full['sharpe']:.2f} MDD{full['mdd']:.1%} Calmar{full['calmar']:.2f}")

    # 分年度
    yr = pd.DataFrame({"strat": net5, "basket": basket.reindex(net5.index).fillna(0),
                       "hs300": hs.reindex(net5.index).fillna(0)})
    rows = []
    for y, gg in yr.groupby(yr.index.year):
        nav = (1 + gg["strat"]).cumprod()
        rows.append({"year": y, "strat": (1 + gg["strat"]).prod() - 1,
                     "basket": (1 + gg["basket"]).prod() - 1, "hs300": (1 + gg["hs300"]).prod() - 1,
                     "strat_mdd": float((nav / nav.cummax() - 1).min()),
                     "strat_sharpe": gg["strat"].mean() / gg["strat"].std(ddof=0) * np.sqrt(H.TD) if gg["strat"].std() else np.nan})
    at = pd.DataFrame(rows); at.to_csv(OUT / "annual_returns.csv", index=False, encoding="utf-8-sig")
    sh = at.copy()
    for c in ["strat", "basket", "hs300", "strat_mdd"]:
        sh[c] = (sh[c] * 100).map(lambda x: f"{x:+.1f}%")
    sh["strat_sharpe"] = sh["strat_sharpe"].map(lambda x: f"{x:.2f}")
    print("\n分年度:\n" + sh.to_string(index=False))
    best_y = int(at.loc[at["strat"].idxmax(), "year"])
    ex = H.metrics(net5[net5.index.year != best_y])
    print(f"去最强年({best_y}, 当年{at['strat'].max():+.1%}): 年化{ex['ann']:+.1%} Sharpe{ex['sharpe']:.2f}")

    # 滚动12M Sharpe
    roll = net5.rolling(H.TD)
    rs = (roll.mean() / roll.std(ddof=0) * np.sqrt(H.TD)).dropna()
    rs.to_csv(OUT / "rolling_sharpe_12m.csv", encoding="utf-8-sig")
    print(f"\n滚动12M Sharpe: 中位{rs.median():.2f} <0占比{(rs<0).mean():.1%} 最低{rs.min():.2f} 最高{rs.max():.2f}")

    # bootstrap
    r = net5.dropna().to_numpy(); L, B, n = 21, 3000, len(r)
    nb = int(np.ceil(n / L)); smax = len(r) - L; sh_b = np.empty(B)
    for b in range(B):
        st = np.random.randint(0, smax + 1, size=nb)
        s = np.concatenate([r[i:i + L] for i in st])[:n]; sd = s.std(ddof=0)
        sh_b[b] = s.mean() / sd * np.sqrt(H.TD) if sd else np.nan
    lo, med, hi = np.nanpercentile(sh_b, [2.5, 50, 97.5])
    pd.Series(sh_b, name="boot_sharpe").to_csv(OUT / "bootstrap_sharpe.csv", index=False, encoding="utf-8-sig")
    print(f"Bootstrap Sharpe: 点估{full['sharpe']:.2f} 中位{med:.2f} 95%CI[{lo:.2f},{hi:.2f}] P(>1){np.nanmean(sh_b>1):.0%}")

    # 成本 sweep
    sw = []
    for bps in [0, 5, 10, 15, 20, 30, 50]:
        m = H.metrics(ret - turn * bps / 1e4)
        sw.append({"cost_bps": bps, "ann": m["ann"], "sharpe": m["sharpe"], "mdd": m["mdd"]})
    swd = pd.DataFrame(sw); swd.to_csv(OUT / "cost_sweep.csv", index=False, encoding="utf-8-sig")
    sd = swd.copy(); sd["ann"] = (sd["ann"]*100).map(lambda x: f"{x:+.1f}%"); sd["mdd"]=(sd["mdd"]*100).map(lambda x: f"{x:.1f}%")
    sd["sharpe"] = sd["sharpe"].map(lambda x: f"{x:.2f}")
    print("\n成本sweep:\n" + sd.to_string(index=False))
    reb = turn[turn > 1e-9]
    print(f"\n换手(lam={LAM}): 单期均{reb.mean():.1%} 年化双边{turn.sum()/(len(turn)/H.TD):.1f}x")
    print(f"\n输出 -> {OUT}")


if __name__ == "__main__":
    main()
