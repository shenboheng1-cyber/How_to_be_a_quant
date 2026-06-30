"""拆分权重的诚实 out-of-sample 检验：
每个测试年只用该年之前的数据算 efficiency/accel 的 ICIR → 定权重 → 用于该年，拼接 OOS。
对比 ①训练集定权重(真OOS) ②固定combo(报告版) ③全样本拆分(样本内/作弊)。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_monthly_weights_v2,
    backtest_monthly_strategy, FACTOR_WEIGHTS_V2)
import hfq_common as H

COST, LAM = 5.0, 0.4
REST = {"momentum_12_1": 0.35, "fund_hit_rate_20": 0.20, "vol_60d": -0.15, "max_drawdown_60d": 0.10}
TEST_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]


def rank_ic(fac_me, ret_me):
    out = {}
    for d in fac_me.index:
        f, r = fac_me.loc[d], ret_me.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= 10:
            out[d] = f[m].rank().corr(r[m].rank())
    return pd.Series(out)


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, _, _ = H.load_hfq(); px = px.loc[:H.END]
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    fac = compute_factor_panel(px); fac["date"] = pd.to_datetime(fac["date"])

    me = [d for d in px.index.to_series().groupby(px.index.to_period("M")).max()
          if pd.Timestamp(H.START) <= d <= pd.Timestamp(H.END)]
    me_idx = pd.DatetimeIndex(me)
    fwd = px.reindex(me_idx).shift(-1) / px.reindex(me_idx) - 1.0
    fw = {f: fac.pivot_table(index="date", columns="fund_code", values=f).reindex(me_idx)
          for f in ["efficiency_20d", "fund_ret_accel_20_60"]}

    def full_ret(weights):
        sc = score_factors_with_weights(fac, weights, score_column="s")
        w = make_monthly_weights_v2(sc, px, uni, buffer_rank=35, volatility_target=0.18, **H.PORT)
        eq, _ = backtest_monthly_strategy(px, w, transaction_cost_bps=COST, rebalance_lambda=LAM)
        return H.to_ret(eq)

    # 固定版只算一次，按年切片
    r_combo = full_ret(dict(FACTOR_WEIGHTS_V2))
    r_fullsplit = full_ret({"efficiency_20d": 0.185, "fund_ret_accel_20_60": 0.265, **REST})

    # 训练集定权重(真OOS)：逐年用之前数据算 ICIR
    print("各测试年用「之前数据」算出的拆分权重：")
    trained_parts = []
    for Y in TEST_YEARS:
        tr = me_idx[me_idx < pd.Timestamp(f"{Y}-01-01")]
        ic_e = rank_ic(fw["efficiency_20d"].loc[tr], fwd.loc[tr])
        ic_a = rank_ic(fw["fund_ret_accel_20_60"].loc[tr], fwd.loc[tr])
        icir_e = ic_e.mean() / ic_e.std(ddof=1)
        icir_a = ic_a.mean() / ic_a.std(ddof=1)
        # ICIR 可能为负，做下限保护后按比例分 0.45
        e, a = max(icir_e, 0.01), max(icir_a, 0.01)
        wE, wA = 0.45 * e / (e + a), 0.45 * a / (e + a)
        print(f"  {Y}: ICIR(eff)={icir_e:+.2f} ICIR(accel)={icir_a:+.2f} -> eff={wE:.3f}, accel={wA:.3f}")
        rY = full_ret({"efficiency_20d": round(wE, 3), "fund_ret_accel_20_60": round(wA, 3), **REST})
        trained_parts.append(rY[rY.index.year == Y])
    r_trained = pd.concat(trained_parts).sort_index()

    # 同口径 OOS 段（2021-2026）对比
    oos_lo = pd.Timestamp("2021-01-01")
    def oos(r): return r[r.index >= oos_lo]
    rows = [
        H.metrics(oos(r_trained), "①训练集定权重(真OOS)"),
        H.metrics(oos(r_combo), "②固定combo(报告版)"),
        H.metrics(oos(r_fullsplit), "③全样本拆分(样本内/作弊)"),
    ]
    df = pd.DataFrame(rows)
    for c in ["total", "ann", "vol", "mdd"]:
        df[c] = (df[c] * 100).map(lambda x: f"{x:+.1f}%")
    df["sharpe"] = df["sharpe"].map(lambda x: f"{x:.2f}")
    df["calmar"] = df["calmar"].map(lambda x: f"{x:.2f}")
    print("\n" + "=" * 76)
    print(f"OOS 对比 (2021-2026, 后复权市价, 含{COST:.0f}bps, lam={LAM})")
    print("=" * 76)
    print(df[["label", "total", "ann", "vol", "sharpe", "mdd", "calmar"]].to_string(index=False))


if __name__ == "__main__":
    main()
