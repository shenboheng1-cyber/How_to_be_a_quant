"""净值口径 vs 后复权市价口径 回测对比。

三条曲线(同一套加固参数):
  A. NAV          : 信号+成交都用累计净值(复现 headline)
  B. HFQ          : 信号+成交都用 iFinD 后复权市价(close_hfq), 511880 同口径
  C. HFQ+filters  : 在 B 基础上加 PIT 流动性(近20日均额≥3000万) + 折溢价(|premiumRatio|≤5%)过滤
对照: 沪深300。
"""
from __future__ import annotations
import sys, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from etf_factor_strategy.data import load_etf_universe, load_nav_prices, DEFAULT_DATA_DIR
from etf_factor_strategy.engine import (
    compute_factor_panel, score_factors_with_weights, make_robust_monthly_weights,
    backtest_monthly_strategy,
)

IFIND_DB = DEFAULT_DATA_DIR / "etf_market_ifind.db"
HISTORY_START, START, END = "2017-01-01", "2020-02-04", "2026-06-05"
COST_BPS = 5.0
ADV_MIN = 3e7          # 近20日均额下限(元)
PREM_MAX = 5.0         # |贴水率|上限(%)

FW = {"combo_eff_accel": 0.45, "momentum_12_1": 0.35, "fund_hit_rate_20": 0.20,
      "vol_60d": -0.15, "max_drawdown_60d": 0.10}
PORT = {"top_n": 20, "max_per_theme": 3, "max_weight": 0.12,
        "volatility_target": 0.18, "weak_market_exposure": 0.60}
TD = 252


def hs300() -> pd.Series:
    con = sqlite3.connect(f"file:{DEFAULT_DATA_DIR/'idx_store.db'}?mode=ro", uri=True)
    d = pd.read_sql_query("SELECT date,close FROM index_daily WHERE symbol='CSI300' ORDER BY date", con)
    con.close()
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["close"].astype(float)


def metrics(ret: pd.Series, label: str) -> dict:
    ret = ret.dropna()
    nav = (1 + ret).cumprod()
    days = len(ret)
    ann = nav.iloc[-1] ** (TD / days) - 1
    vol = ret.std(ddof=0) * np.sqrt(TD)
    mdd = (nav / nav.cummax() - 1).min()
    return {"策略": label, "累计": nav.iloc[-1] - 1, "年化": ann, "波动": vol,
            "Sharpe": ann / vol if vol else np.nan, "MDD": float(mdd),
            "Calmar": ann / abs(mdd) if mdd else np.nan}


def load_ifind() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(f"file:{IFIND_DB}?mode=ro", uri=True)
    q = pd.read_sql_query(
        "SELECT fund_code, date, close_hfq, amount, premiumRatio FROM etf_quote",
        con, parse_dates=["date"])
    con.close()
    q["fund_code"] = q["fund_code"].astype(str).str.zfill(6)
    q = q[(q["date"] >= HISTORY_START) & (q["date"] <= END)]
    px = q.pivot_table(index="date", columns="fund_code", values="close_hfq").sort_index()
    amt = q.pivot_table(index="date", columns="fund_code", values="amount").sort_index()
    prem = q.pivot_table(index="date", columns="fund_code", values="premiumRatio").sort_index()
    return px, amt, prem


def run(prices: pd.DataFrame, universe: pd.DataFrame, market: pd.Series,
        score_mask: pd.DataFrame | None = None) -> pd.Series:
    uni2 = universe[universe["fund_code"].isin(prices.columns)].copy()
    fac = compute_factor_panel(prices)
    scored = score_factors_with_weights(fac, FW, score_column="risk_adjusted_score")
    if score_mask is not None:
        scored = scored.merge(score_mask, on=["date", "fund_code"], how="left")
        bad = (scored["tradable_ok"] != True)  # noqa: E712  (NaN -> 也判不可交易)
        scored.loc[bad, "risk_adjusted_score"] = np.nan
        scored.loc[bad, "score"] = np.nan
    w = make_robust_monthly_weights(scored, prices, uni2, market_nav=market,
                                    cash_code="511880", **PORT)
    eq, _ = backtest_monthly_strategy(prices, w, transaction_cost_bps=COST_BPS)
    eq["date"] = pd.to_datetime(eq["date"])
    return pd.Series(eq["strategy_return"].to_numpy(), index=eq["date"]).loc[START:END]


def build_mask(amt: pd.DataFrame, prem: pd.DataFrame) -> pd.DataFrame:
    adv20 = amt.rolling(20, min_periods=10).mean()
    ok = (adv20 >= ADV_MIN) & (prem.abs() <= PREM_MAX)
    # 现金腿永远可交易
    if "511880" in ok.columns:
        ok["511880"] = True
    long = ok.stack(future_stack=True).rename("tradable_ok").reset_index()
    long.columns = ["date", "fund_code", "tradable_ok"]
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    return long


def main() -> None:
    uni = load_etf_universe(data_dir=DEFAULT_DATA_DIR)
    market = hs300()

    # A. NAV
    nav_px = load_nav_prices(uni["fund_code"].tolist(), data_dir=DEFAULT_DATA_DIR,
                             start=HISTORY_START, end=END).dropna(axis=1, thresh=280)
    nav_px.columns = nav_px.columns.astype(str).str.zfill(6)
    r_nav = run(nav_px, uni, market)

    # B/C. HFQ
    hfq_px, amt, prem = load_ifind()
    hfq_px = hfq_px.dropna(axis=1, thresh=280)
    print(f"NAV池 {nav_px.shape[1]} 只 | HFQ池 {hfq_px.shape[1]} 只 | 交集 "
          f"{len(set(nav_px.columns) & set(hfq_px.columns))}")
    r_hfq = run(hfq_px, uni, market)
    mask = build_mask(amt.reindex(columns=hfq_px.columns), prem.reindex(columns=hfq_px.columns))
    r_hfqf = run(hfq_px, uni, market, score_mask=mask)

    # 基准
    hs = market.loc[START:END]
    r_hs = hs.pct_change().dropna()

    rows = [metrics(r_nav, "A. NAV(累计净值)"),
            metrics(r_hfq, "B. 后复权市价"),
            metrics(r_hfqf, f"C. 市价+流动性/折溢价过滤"),
            metrics(r_hs, "沪深300")]
    df = pd.DataFrame(rows)
    pct = ["累计", "年化", "波动", "MDD"]
    for c in pct:
        df[c] = (df[c] * 100).map(lambda x: f"{x:+.1f}%")
    df["Sharpe"] = df["Sharpe"].map(lambda x: f"{x:.2f}")
    df["Calmar"] = df["Calmar"].map(lambda x: f"{x:.2f}")
    print("\n" + "=" * 78)
    print(f"回测区间 {START} ~ {END} | 成本 {COST_BPS}bps | 过滤: 近20日均额≥{ADV_MIN/1e4:.0f}万, |贴水|≤{PREM_MAX}%")
    print("=" * 78)
    print(df.to_string(index=False))

    out = ROOT / "outputs_price_vs_nav"
    out.mkdir(exist_ok=True)
    navdf = pd.DataFrame({"NAV": (1 + r_nav).cumprod(), "HFQ": (1 + r_hfq).cumprod(),
                          "HFQ_filtered": (1 + r_hfqf).cumprod(),
                          "HS300": (1 + r_hs).cumprod()})
    navdf.to_csv(out / "nav_curves.csv", encoding="utf-8-sig")
    df.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    print(f"\n输出 -> {out}")


if __name__ == "__main__":
    main()
