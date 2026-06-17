"""后复权市价口径的共享回测工具：加载面板、灵活权重(支持消融)、指标、冲击成本。"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

from etf_factor_strategy.data import load_etf_universe, load_nav_prices, load_hfq_market, DEFAULT_DATA_DIR
from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, backtest_monthly_strategy,
    make_monthly_weights_v2, tradable_codes_at_date, _select_with_theme_cap, _cap_and_redistribute,
    _estimate_portfolio_vol, _is_weak_market, _last_available_row, _infer_theme, _month_end_dates,
)

ROOT = Path(__file__).resolve().parent
IFIND_DB = DEFAULT_DATA_DIR / "etf_market_ifind.db"
HISTORY_START, START, END = "2016-01-01", "2018-01-02", "2026-06-05"
TD = 252

FW = {"combo_eff_accel": 0.45, "momentum_12_1": 0.35, "fund_hit_rate_20": 0.20,
      "vol_60d": -0.15, "max_drawdown_60d": 0.10}
FW3 = {"combo_eff_accel": 0.45, "momentum_12_1": 0.35, "fund_hit_rate_20": 0.20}
PORT = dict(top_n=20, max_per_theme=3, max_weight=0.12)
DEFENSIVE_THEMES = {"固收", "货币现金", "黄金"}


def load_hfq():
    # 委托给固化后的 data.load_hfq_market（单一数据源）
    return load_hfq_market(data_dir=DEFAULT_DATA_DIR, start=HISTORY_START, end=END, min_obs=280)


def hs300():
    con = sqlite3.connect(f"file:{DEFAULT_DATA_DIR/'idx_store.db'}?mode=ro", uri=True)
    d = pd.read_sql_query("SELECT date,close FROM index_daily WHERE symbol='CSI300' ORDER BY date", con)
    con.close()
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["close"].astype(float)


def make_weights_flexible(scored, prices, universe, *, top_n=20, max_per_theme=3, max_weight=0.12,
                          weighting="inv_vol", volatility_target=None, market_nav=None,
                          weak_market_exposure=1.0, cash_code="511880", exclude_themes=None,
                          recent_nav_days=5, max_missing_60=0.10, max_missing_252=0.20):
    """make_robust_monthly_weights 的可配置版本，用于消融实验。
    weighting='inv_vol'|'equal'; volatility_target=None 关闭波动目标; market_nav=None 关闭弱市择时。"""
    meta = universe.copy()
    if "theme" not in meta.columns:
        meta["theme"] = meta.apply(_infer_theme, axis=1)
    enriched = scored.merge(meta, on="fund_code", how="left")
    returns = prices.pct_change(fill_method=None)
    vol60 = returns.rolling(60, min_periods=40).std() * np.sqrt(TD)
    rows = []
    med = _month_end_dates(enriched["date"])
    month_end = enriched[enriched["date"].isin(med)].dropna(subset=["score"])
    for date, group in month_end.groupby("date", sort=True):
        if exclude_themes:
            group = group[~group["theme"].isin(exclude_themes)]
        dt = pd.Timestamp(date)
        tradable = tradable_codes_at_date(prices, dt, recent_nav_days, max_missing_60, max_missing_252)
        group = group[group["fund_code"].isin(tradable)]
        picks = _select_with_theme_cap(group, top_n=top_n, max_per_theme=max_per_theme)
        if picks.empty:
            continue
        if weighting == "equal":
            raw = pd.Series(1.0 / len(picks), index=picks["fund_code"])
        else:
            vr = _last_available_row(vol60, dt)
            inv = 1.0 / vr.reindex(picks["fund_code"]).replace([np.inf, -np.inf], np.nan)
            inv = inv.fillna(inv.median()).replace(0.0, np.nan)
            raw = pd.Series(1.0 / len(picks), index=picks["fund_code"]) if inv.isna().all() else inv / inv.sum()
        capped = _cap_and_redistribute(raw, max_weight=max_weight)
        exposure = 1.0
        if volatility_target is not None:
            pv = _estimate_portfolio_vol(prices, capped, dt)
            if pv and not pd.isna(pv):
                exposure = min(exposure, volatility_target / pv)
        if market_nav is not None and _is_weak_market(market_nav, dt):
            exposure = min(exposure, weak_market_exposure)
        risky = capped * exposure
        for code, wt in risky.items():
            if wt > 0:
                rows.append({"date": date, "fund_code": code, "weight": float(wt)})
        cash = 1.0 - float(risky.sum())
        if cash > 1e-10:
            rows.append({"date": date, "fund_code": cash_code, "weight": cash})
    return pd.DataFrame(rows, columns=["date", "fund_code", "weight"])


def equal_weight_basket(prices, *, recent_nav_days=5, max_missing_60=0.10, max_missing_252=0.20):
    """全市场可交易 ETF 月度等权篮子（beta 基准）。"""
    rows = []
    dates = pd.Series(prices.index.strftime("%Y-%m-%d"))
    for d in sorted(_month_end_dates(dates)):
        dt = pd.Timestamp(d)
        tr = sorted(tradable_codes_at_date(prices, dt, recent_nav_days, max_missing_60, max_missing_252))
        if not tr:
            continue
        w = 1.0 / len(tr)
        for c in tr:
            rows.append({"date": d, "fund_code": c, "weight": w})
    return pd.DataFrame(rows, columns=["date", "fund_code", "weight"])


def to_ret(eq):
    eq = eq.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    return pd.Series(eq["strategy_return"].to_numpy(), index=eq["date"]).loc[START:END]


def metrics(ret, label=""):
    ret = ret.dropna()
    nav = (1 + ret).cumprod()
    ann = nav.iloc[-1] ** (TD / len(ret)) - 1
    vol = ret.std(ddof=0) * np.sqrt(TD)
    mdd = float((nav / nav.cummax() - 1).min())
    return {"label": label, "total": nav.iloc[-1] - 1, "ann": ann, "vol": vol,
            "sharpe": ann / vol if vol else np.nan, "mdd": mdd,
            "calmar": ann / abs(mdd) if mdd else np.nan}


def fmt(rows, cols=("total", "ann", "vol", "mdd")):
    df = pd.DataFrame(rows)
    for c in cols:
        df[c] = (df[c] * 100).map(lambda x: f"{x:+.1f}%")
    df["sharpe"] = df["sharpe"].map(lambda x: f"{x:.2f}")
    df["calmar"] = df["calmar"].map(lambda x: f"{x:.2f}")
    return df


def make_weights_v2(scored, prices, universe, **kw):
    # 委托给固化后的 engine.make_monthly_weights_v2（单一逻辑源）
    return make_monthly_weights_v2(scored, prices, universe, **kw)


def backtest_smoothed(prices, weights, lam=1.0, cost_bps=0.0):
    # 委托给固化后的 engine.backtest_monthly_strategy（rebalance_lambda=lam）
    return backtest_monthly_strategy(prices, weights, transaction_cost_bps=cost_bps, rebalance_lambda=lam)


def resample_weights(w, every_k):
    """只保留每 every_k 个调仓日的权重（回测会 ffill → 持有更久），用于降调仓频率。"""
    ds = sorted(w["date"].unique())
    keep = set(ds[::every_k])
    return w[w["date"].isin(keep)].copy()


def apply_impact(g_ret, turn, eff, px, amt, aum, base_bps=5.0, coef=0.5):
    """平方根冲击模型净收益：gross - 固定佣金/价差 - 冲击。"""
    rets = px.pct_change(fill_method=None)
    sigma = rets.rolling(60, min_periods=40).std().reindex(eff.index).reindex(columns=eff.columns)
    adv = (amt.reindex(columns=px.columns).rolling(20, min_periods=10).mean()
           .reindex(eff.index).reindex(columns=eff.columns))
    dW = eff.diff().abs()
    part = (aum * dW) / adv.replace(0, np.nan)
    impact_frac = coef * sigma * np.sqrt(part.clip(lower=0))
    impact_t = (dW * impact_frac).sum(axis=1).reindex(g_ret.index).fillna(0.0)
    return g_ret - turn * base_bps / 1e4 - impact_t
