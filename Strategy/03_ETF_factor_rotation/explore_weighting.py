"""路线A-1：结构性加权改进。逆波动 vs 最小方差(收缩协方差) vs 等权。
无拟合参数(协方差用 PIT 滚动估计、收缩强度固定 0.5),不易过拟合。
报告 全期(2018-2026) + OOS段(2021-2026 / 2023-2026),看改善是否一致。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, backtest_monthly_strategy, FACTOR_WEIGHTS_V2,
    tradable_codes_at_date, _cap_and_redistribute, _estimate_portfolio_vol, _last_available_row,
    _infer_theme, _month_end_dates)
import hfq_common as H

COST, LAM, LOOKBACK, SHRINK = 5.0, 0.4, 120, 0.5


def inv_vol_w(picks, px, dt, vol60):
    vr = _last_available_row(vol60, dt)
    inv = 1.0 / vr.reindex(picks).replace([np.inf, -np.inf], np.nan)
    inv = inv.fillna(inv.median()).replace(0.0, np.nan)
    return pd.Series(1.0 / len(picks), index=picks) if inv.isna().all() else inv / inv.sum()


def minvar_w(picks, px, dt, vol60):
    R = px[list(picks)].pct_change(fill_method=None).loc[:dt].tail(LOOKBACK)
    cols = [c for c in picks if R[c].notna().sum() >= 40]
    if len(cols) < 2:
        return pd.Series(1.0 / len(picks), index=picks)
    cov = (R[cols].fillna(0.0).cov().values) * 252.0
    cov = SHRINK * np.diag(np.diag(cov)) + (1 - SHRINK) * cov     # 收缩向对角
    try:
        w = np.linalg.pinv(cov) @ np.ones(len(cols))
    except Exception:
        w = np.ones(len(cols))
    w = np.clip(w, 0.0, None)
    w = pd.Series((w / w.sum()) if w.sum() > 0 else (np.ones(len(cols)) / len(cols)), index=cols)
    w = w.reindex(picks).fillna(0.0)
    return w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(picks), index=picks)


def equal_w(picks, px, dt, vol60):
    return pd.Series(1.0 / len(picks), index=picks)


def build(scored, px, uni, weight_fn, top_n=20, max_per_theme=3, max_weight=0.12,
          buffer_rank=35, vol_target=0.18, cash="511880"):
    meta = uni.copy()
    if "theme" not in meta.columns:
        meta["theme"] = meta.apply(_infer_theme, axis=1)
    enr = scored.merge(meta, on="fund_code", how="left")
    vol60 = px.pct_change(fill_method=None).rolling(60, min_periods=40).std() * np.sqrt(252.0)
    rows, prev = [], set()
    med = _month_end_dates(enr["date"])
    me = enr[enr["date"].isin(med)].dropna(subset=["score"])
    for date, g in me.groupby("date", sort=True):
        dt = pd.Timestamp(date)
        tr = tradable_codes_at_date(px, dt)
        g = g[g["fund_code"].isin(tr)]
        if g.empty:
            continue
        rk = g.sort_values(["score", "fund_code"], ascending=[False, True]).reset_index(drop=True)
        order = list(rk["fund_code"]); rank = {c: i + 1 for i, c in enumerate(order)}
        theme = dict(zip(rk["fund_code"], rk["theme"].astype(str)))
        sel, tc = [], {}
        def add(c):
            t = theme[c]
            if tc.get(t, 0) < max_per_theme:
                sel.append(c); tc[t] = tc.get(t, 0) + 1
        for c in sorted([c for c in order if c in prev and rank[c] <= buffer_rank], key=lambda x: rank[x]):
            if len(sel) >= top_n: break
            add(c)
        for c in order:
            if len(sel) >= top_n: break
            if c not in sel: add(c)
        if not sel: continue
        picks = pd.Index(sel)
        raw = weight_fn(picks, px, dt, vol60)
        capped = _cap_and_redistribute(raw, max_weight)
        exp = 1.0
        pv = _estimate_portfolio_vol(px, capped, dt)
        if pv and not pd.isna(pv): exp = min(1.0, vol_target / pv)
        risky = capped * exp
        for c, wt in risky.items():
            if wt > 0: rows.append({"date": date, "fund_code": c, "weight": float(wt)})
        cw = 1.0 - float(risky.sum())
        if cw > 1e-10: rows.append({"date": date, "fund_code": cash, "weight": cw})
        prev = set(picks)
    return pd.DataFrame(rows, columns=["date", "fund_code", "weight"])


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, _, _ = H.load_hfq(); px = px.loc[:H.END]
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    fac = compute_factor_panel(px)
    scored = score_factors_with_weights(fac, FACTOR_WEIGHTS_V2, score_column="s")

    schemes = {"逆波动(V2现状)": inv_vol_w, "最小方差(收缩)": minvar_w, "等权": equal_w}
    segs = {"全期2018-2026": ("2018-01-02", H.END), "OOS 2021-2026": ("2021-01-01", H.END),
            "OOS 2023-2026": ("2023-01-01", H.END)}
    print(f"{'方案':>14} | " + " | ".join(f"{s:>16}" for s in segs))
    print("-" * 80)
    for name, fn in schemes.items():
        w = build(scored, px, uni, fn)
        eq, _ = backtest_monthly_strategy(px, w, transaction_cost_bps=COST, rebalance_lambda=LAM)
        r = H.to_ret(eq)
        cells = []
        for s, (lo, hi) in segs.items():
            rr = r.loc[lo:hi]; m = H.metrics(rr)
            cells.append(f"{m['ann']*100:>5.1f}%/S{m['sharpe']:.2f}")
        print(f"{name:>14} | " + " | ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
