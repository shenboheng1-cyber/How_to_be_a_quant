"""因子审查：后复权市价口径下的 Rank IC / 分层 / 相关 / IC 衰减。

口径：后复权市价(close_hfq)，月度截面，下期收益=下个月末/本月末-1（衰减另用固定持有期）。
输出 -> outputs_factor_diag/
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from etf_factor_strategy.engine import compute_factor_panel
import hfq_common as H

OUT = H.ROOT / "outputs_factor_diag"
OUT.mkdir(exist_ok=True)

# 仅最终入选的 5 个打分因子（不含中间/合成原料）
FACTORS = ["combo_eff_accel", "momentum_12_1", "fund_hit_rate_20", "vol_60d", "max_drawdown_60d"]


def rank_ic_series(fac_me: pd.DataFrame, ret_me: pd.DataFrame) -> pd.Series:
    """逐月截面 Rank IC = corr(rank(因子), rank(下期收益))。"""
    ics = {}
    for d in fac_me.index:
        f, r = fac_me.loc[d], ret_me.loc[d]
        m = f.notna() & r.notna()
        if m.sum() < 10:
            continue
        ics[d] = f[m].rank().corr(r[m].rank())
    return pd.Series(ics).sort_index()


def main():
    px, _, _ = H.load_hfq()
    px = px.loc[:H.END]
    print(f"HFQ池 {px.shape[1]} 只；计算因子面板…")
    fac = compute_factor_panel(px)
    fac["date"] = pd.to_datetime(fac["date"])

    # 月末日期（每月最后交易日），限定回测区间
    me_dates = [d for d in px.index.to_series().groupby(px.index.to_period("M")).max()
                if pd.Timestamp(H.START) <= d <= pd.Timestamp(H.END)]
    me_idx = pd.DatetimeIndex(me_dates)

    # 月度下期收益（下个月末/本月末-1）
    rprice = px.reindex(me_idx)
    fwd_1m = rprice.shift(-1) / rprice - 1.0

    # 因子宽表（月末截面）
    fac_wide = {f: fac.pivot_table(index="date", columns="fund_code", values=f).reindex(me_idx)
                for f in FACTORS}

    # ---- 1. Rank IC 汇总 ----
    ic_rows, cum_ic = [], {}
    for f in FACTORS:
        ic = rank_ic_series(fac_wide[f], fwd_1m)
        cum_ic[f] = ic.cumsum()
        n = len(ic)
        mean, std = ic.mean(), ic.std(ddof=1)
        ic_rows.append({"factor": f, "ic_mean": mean, "ic_std": std,
                        "icir": mean / std if std else np.nan,
                        "ic_pos_pct": (ic > 0).mean(), "t_stat": mean / std * np.sqrt(n) if std else np.nan,
                        "n_months": n})
    ic_df = pd.DataFrame(ic_rows)
    pd.DataFrame(cum_ic).to_csv(OUT / "cumulative_ic.csv", encoding="utf-8-sig")
    ic_df.to_csv(OUT / "ic_summary.csv", index=False, encoding="utf-8-sig")
    show = ic_df.copy()
    for c in ["ic_mean", "ic_std", "icir", "t_stat"]:
        show[c] = show[c].map(lambda x: f"{x:+.3f}")
    show["ic_pos_pct"] = (show["ic_pos_pct"] * 100).map(lambda x: f"{x:.0f}%")
    print("\n=== 1. Rank IC 汇总（月度，市价口径）===")
    print(show.to_string(index=False))

    # ---- 2. 分层(5分位)下期收益 ----
    lay_rows = []
    for f in FACTORS:
        fw, fr = fac_wide[f], fwd_1m
        q_ret = {q: [] for q in range(1, 6)}
        for d in fw.index:
            s, r = fw.loc[d], fr.loc[d]
            m = s.notna() & r.notna()
            if m.sum() < 20:
                continue
            ranks = s[m].rank(pct=True)
            for q in range(1, 6):
                sel = ranks[(ranks > (q - 1) / 5) & (ranks <= q / 5)].index
                if len(sel):
                    q_ret[q].append(r[sel].mean())
        means = {q: np.mean(q_ret[q]) if q_ret[q] else np.nan for q in range(1, 6)}
        lay_rows.append({"factor": f, **{f"Q{q}": means[q] for q in range(1, 6)},
                         "long_short_Q5_Q1": means[5] - means[1]})
    lay_df = pd.DataFrame(lay_rows)
    lay_df.to_csv(OUT / "quintile_returns.csv", index=False, encoding="utf-8-sig")
    show2 = lay_df.copy()
    for c in [f"Q{q}" for q in range(1, 6)] + ["long_short_Q5_Q1"]:
        show2[c] = (show2[c] * 100).map(lambda x: f"{x:+.2f}%")
    print("\n=== 2. 分层(5分位)平均下期月收益 + 多空 ===")
    print(show2.to_string(index=False))

    # ---- 3. 因子相关矩阵（Spearman，月末截面池化）----
    sub = fac[fac["date"].isin(me_idx)][FACTORS]
    corr = sub.corr(method="spearman")
    corr.to_csv(OUT / "factor_corr_spearman.csv", encoding="utf-8-sig")
    print("\n=== 3. 因子相关矩阵（Spearman）===")
    print(corr.round(2).to_string())

    # ---- 4. IC 衰减（持有期 20/60/120 交易日）----
    dec_rows = []
    for f in FACTORS:
        row = {"factor": f}
        for h in [20, 60, 120]:
            fwd_h = (px.shift(-h) / px - 1.0).reindex(me_idx)
            ic = rank_ic_series(fac_wide[f], fwd_h)
            row[f"ic_h{h}"] = ic.mean()
        dec_rows.append(row)
    dec_df = pd.DataFrame(dec_rows)
    dec_df.to_csv(OUT / "ic_decay.csv", index=False, encoding="utf-8-sig")
    show4 = dec_df.copy()
    for c in ["ic_h20", "ic_h60", "ic_h120"]:
        show4[c] = show4[c].map(lambda x: f"{x:+.3f}")
    print("\n=== 4. IC 衰减（不同持有期 IC 均值）===")
    print(show4.to_string(index=False))

    print(f"\n输出 -> {OUT}")


if __name__ == "__main__":
    main()
