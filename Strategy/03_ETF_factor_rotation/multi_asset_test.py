"""跨资产 GTAA:把 ETF 按资产类别分桶(A股/港股/海外股/黄金/债券/商品),
每月对各类别做时序趋势过滤(12月动量>0才持有)+ 类别间风险平价(逆波动)+ vol-target,
类别内等权其成份。对比 V3、测相关性、合并。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_monthly_weights_v2,
    backtest_monthly_strategy, FACTOR_WEIGHTS_V2, tradable_codes_at_date, _month_end_dates)
import hfq_common as H

COST, LAM, CASH = 5.0, 0.4, "511880"
RISK_CLASSES = ["A股", "港股", "海外股", "黄金", "债券", "商品"]


def classify(n):
    n = str(n)
    if any(k in n for k in ["债", "国债", "信用", "政金", "可转债"]): return "债券"
    if any(k in n for k in ["货币", "日利", "现金"]): return "货币"
    if ("黄金" in n and "股" not in n) or "金ETF" in n: return "黄金"
    if any(k in n for k in ["原油", "石油", "有色", "商品", "白银", "豆粕", "能源化工", "矿业"]): return "商品"
    if any(k in n for k in ["纳斯达克", "纳指", "标普", "美国", "道琼斯", "德国", "法国", "日经", "东南亚", "亚太"]): return "海外股"
    if any(k in n for k in ["恒生", "港股", "H股", "香港", "中概"]): return "港股"
    return "A股"


def gtaa_weights(px, uni):
    ret = px.pct_change(fill_method=None)
    members = {c: uni[uni["cls"] == c]["fund_code"].tolist() for c in RISK_CLASSES}
    cls_ret = pd.DataFrame({c: ret[[m for m in members[c] if m in ret.columns]].mean(axis=1) for c in RISK_CLASSES})
    cls_nav = (1 + cls_ret.fillna(0)).cumprod()
    rows = []
    for d in sorted(_month_end_dates(pd.Series(px.index.strftime("%Y-%m-%d")))):
        dt = pd.Timestamp(d)
        elig, vols = [], {}
        for c in RISK_CLASSES:
            h = cls_nav[c].loc[:dt]
            if len(h) < 252:
                continue
            trend = h.iloc[-1] / h.iloc[-252] - 1.0
            v = cls_ret[c].loc[:dt].tail(60).std() * np.sqrt(252)
            if trend > 0 and v and v > 0:
                elig.append(c); vols[c] = v
        if not elig:
            rows.append({"date": d, "fund_code": CASH, "weight": 1.0}); continue
        inv = pd.Series({c: 1 / vols[c] for c in elig}); wc = (inv / inv.sum()).clip(upper=0.35)
        wc = wc / wc.sum()
        cov = cls_ret[elig].loc[:dt].tail(60).cov() * 252
        pv = float(np.sqrt(wc.values @ cov.values @ wc.values))
        exp = min(1.0, 0.18 / pv) if pv > 0 else 1.0
        wc = wc * exp
        tr = tradable_codes_at_date(px, dt)
        for c in elig:
            mem = [m for m in members[c] if m in tr]
            if not mem:
                continue
            for m in mem:
                rows.append({"date": d, "fund_code": m, "weight": float(wc[c] / len(mem))})
        cash = 1.0 - float(wc.sum())
        if cash > 1e-9:
            rows.append({"date": d, "fund_code": CASH, "weight": cash})
    return pd.DataFrame(rows, columns=["date", "fund_code", "weight"])


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, _, _ = H.load_hfq(); px = px.loc[:H.END]
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    uni["cls"] = uni["fund_name"].map(classify)

    # V3
    fac = compute_factor_panel(px); sc = score_factors_with_weights(fac, FACTOR_WEIGHTS_V2, score_column="s")
    wv3 = make_monthly_weights_v2(sc, px, uni, buffer_rank=35, volatility_target=0.18, **H.PORT)
    eq3, _ = backtest_monthly_strategy(px, wv3, transaction_cost_bps=COST, rebalance_lambda=LAM)
    r_v3 = H.to_ret(eq3)

    # GTAA
    wg = gtaa_weights(px, uni)
    eqg, _ = backtest_monthly_strategy(px, wg, transaction_cost_bps=COST, rebalance_lambda=LAM)
    r_g = H.to_ret(eqg)

    def line(tag, r):
        m = H.metrics(r)
        print(f"{tag:24}年化{m['ann']*100:+6.1f}% 波动{m['vol']*100:5.1f}% Sharpe{m['sharpe']:.2f} MDD{m['mdd']*100:6.1f}% Calmar{m['calmar']:.2f}")

    print("=" * 76)
    line("V3(横截面ETF轮动)", r_v3)
    line("跨资产GTAA(趋势+风险平价)", r_g)
    rho = r_v3.corr(r_g.reindex(r_v3.index))
    print(f"\n两者相关 {rho:.2f}")
    print("\n=== V3 + GTAA 合并 ===")
    for w1 in (0.7, 0.6, 0.5, 0.4, 0.3):
        line(f"  {w1:.0%}V3/{1-w1:.0%}GTAA", w1 * r_v3 + (1 - w1) * r_g.reindex(r_v3.index).fillna(0))
    s = np.sqrt(2 / (1 + max(rho, -0.99)))
    print(f"\n理论(等权同Sharpe ρ={rho:.2f}): ×{s:.2f}")


if __name__ == "__main__":
    main()
