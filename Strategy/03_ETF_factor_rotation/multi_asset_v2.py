"""分层跨资产策略:类别内用 V3 因子选最强(选股alpha) + 类别间趋势过滤&风险平价(分散) + 杠杆。
目标:收益和 Sharpe 同时高于 V3 / 原 GTAA。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_monthly_weights_v2,
    backtest_monthly_strategy, FACTOR_WEIGHTS_V2, _minvar_weights,
    tradable_codes_at_date, _month_end_dates)
from multi_asset_test import classify, RISK_CLASSES
import hfq_common as H

COST, LAM, CASH = 5.0, 0.4, "511880"


def hier_weights(px, uni, scored, topk=5, a_cap=0.30, vol_target=0.20):
    ret = px.pct_change(fill_method=None)
    members = {c: [m for m in uni[uni["cls"] == c]["fund_code"] if m in ret.columns] for c in RISK_CLASSES}
    cls_ret = pd.DataFrame({c: ret[members[c]].mean(axis=1) for c in RISK_CLASSES})
    cls_nav = (1 + cls_ret.fillna(0)).cumprod()
    sc_by_date = {d: g.set_index("fund_code")["score"] for d, g in scored.dropna(subset=["score"]).groupby("date")}
    rows = []
    for d in sorted(_month_end_dates(pd.Series(px.index.strftime("%Y-%m-%d")))):
        dt = pd.Timestamp(d)
        sc = sc_by_date.get(d)
        if sc is None:
            continue
        tr = tradable_codes_at_date(px, dt)
        elig, vols, picks_w = [], {}, {}
        for c in RISK_CLASSES:
            h = cls_nav[c].loc[:dt]
            if len(h) < 252:
                continue
            if h.iloc[-1] / h.iloc[-252] - 1.0 <= 0:           # 时序趋势过滤
                continue
            cand = [m for m in members[c] if m in tr and m in sc.index]
            if not cand:
                continue
            top = sc.reindex(cand).dropna().sort_values(ascending=False).head(topk).index
            if len(top) == 0:
                continue
            v = cls_ret[c].loc[:dt].tail(60).std() * np.sqrt(252)
            if not v or v <= 0:
                continue
            elig.append(c); vols[c] = v
            picks_w[c] = _minvar_weights(px, pd.Index(list(top)), dt)   # 类别内 min-var
        if not elig:
            rows.append({"date": d, "fund_code": CASH, "weight": 1.0}); continue
        inv = pd.Series({c: 1 / vols[c] for c in elig}); wc = (inv / inv.sum())
        if "A股" in wc.index:                                   # 压 A股 主导 → 更分散
            wc["A股"] = min(wc["A股"], a_cap); wc = wc / wc.sum()
        cov = cls_ret[elig].loc[:dt].tail(60).cov() * 252
        pv = float(np.sqrt(wc.values @ cov.values @ wc.values))
        exp = min(1.0, vol_target / pv) if pv > 0 else 1.0
        wc = wc * exp
        for c in elig:
            for code, w_in in picks_w[c].items():
                if w_in > 0:
                    rows.append({"date": d, "fund_code": code, "weight": float(wc[c] * w_in)})
        cash = 1.0 - float(wc.sum())
        if cash > 1e-9:
            rows.append({"date": d, "fund_code": CASH, "weight": cash})
    return pd.DataFrame(rows, columns=["date", "fund_code", "weight"])


def line(tag, r):
    m = H.metrics(r)
    print(f"{tag:26}年化{m['ann']*100:+6.1f}% 波动{m['vol']*100:5.1f}% Sharpe{m['sharpe']:.2f} MDD{m['mdd']*100:6.1f}% Calmar{m['calmar']:.2f}")


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, _, _ = H.load_hfq(); px = px.loc[:H.END]
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    uni["cls"] = uni["fund_name"].map(classify)
    fac = compute_factor_panel(px)
    scored = score_factors_with_weights(fac, FACTOR_WEIGHTS_V2, score_column="s")

    wv3 = make_monthly_weights_v2(scored, px, uni, buffer_rank=35, volatility_target=0.18, **H.PORT)
    r_v3 = H.to_ret(backtest_monthly_strategy(px, wv3, transaction_cost_bps=COST, rebalance_lambda=LAM)[0])

    print("=" * 80)
    line("V3 基准", r_v3)
    best = None
    for topk in (3, 5, 8):
        for ac in (0.25, 0.30, 0.40):
            w = hier_weights(px, uni, scored, topk=topk, a_cap=ac)
            r = H.to_ret(backtest_monthly_strategy(px, w, transaction_cost_bps=COST, rebalance_lambda=LAM)[0])
            line(f"分层 topk={topk} A股cap={ac:.0%}", r)
            if best is None or H.metrics(r)["sharpe"] > best[1]:
                best = (r, H.metrics(r)["sharpe"], topk, ac)
    rb, sb, tk, ac = best
    print(f"\n最优分层: topk={tk} A股cap={ac:.0%} (Sharpe {sb:.2f})")
    print("=== 加杠杆到 ~15% 年化(融资2%/年) ===")
    for L in (1.0, 1.5, 2.0):
        line(f"  分层 ×{L}", L * rb - (L - 1) * 0.02 / 252)
    print("\n=== V3 + 最优分层 合并 ===")
    for w1 in (0.5, 0.4, 0.3):
        line(f"  {w1:.0%}V3/{1-w1:.0%}分层", w1 * r_v3 + (1 - w1) * rb.reindex(r_v3.index).fillna(0))
    print(f"\n分层 vs V3 相关 {r_v3.corr(rb.reindex(r_v3.index)):.2f}")


if __name__ == "__main__":
    main()
