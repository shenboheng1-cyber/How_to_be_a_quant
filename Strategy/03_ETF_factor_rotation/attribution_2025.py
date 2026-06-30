"""2025 激增归因: 三因子在 2025 持仓 + 贡献拆解(按ETF/主题/资产类)。"""
import sys, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from etf_factor_strategy.data import load_etf_universe, load_nav_prices, DEFAULT_DATA_DIR
from etf_factor_strategy.engine import compute_factor_panel, score_factors_with_weights, make_robust_monthly_weights, _infer_theme
from factor_research import factor_panels
from incremental_factor_backtest import add_factor_cols
SCORE = {"combo_eff_accel": 0.45, "momentum_12_1": 0.35, "fund_hit_rate_20": 0.20, "downside_vol_60d": -0.15, "max_drawdown_60d": 0.10}
TIGHT = {"top_n": 20, "max_per_theme": 3, "max_weight": 0.12, "volatility_target": 0.18, "weak_market_exposure": 0.60}


def hs():
    con = sqlite3.connect(f"file:{DEFAULT_DATA_DIR}/idx_store.db?mode=ro", uri=True)
    d = pd.read_sql_query("SELECT date,close FROM index_daily WHERE symbol='CSI300' ORDER BY date", con)
    con.close(); d["date"] = pd.to_datetime(d["date"]); return d.set_index("date")["close"].astype(float)


def cls(n):
    n = str(n)
    if any(k in n for k in ["货币", "日利", "现金"]): return "现金"
    if any(k in n for k in ["国债", "政金", "债", "信用", "转债", "国开"]): return "债券"
    if "黄金" in n or "金ETF" in n: return "黄金"
    if any(k in n for k in ["原油", "油气", "商品", "有色", "豆粕", "能源化工", "白银"]): return "商品"
    if any(k in n for k in ["标普", "纳指", "纳斯达克", "恒生", "港股", "日经", "德国", "法国", "海外", "中概", "亚太", "美国"]): return "海外"
    return "A股权益"


def main():
    uni = load_etf_universe(data_dir=DEFAULT_DATA_DIR)
    px = load_nav_prices(uni["fund_code"].tolist(), data_dir=DEFAULT_DATA_DIR, start="2017-01-01", end="2026-06-05").dropna(axis=1, thresh=280)
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    nm = uni.set_index("fund_code")["fund_name"].to_dict()
    panels, _ = factor_panels(px)
    fac = compute_factor_panel(px); fac = add_factor_cols(fac, panels, ["low_corr_120", "resid_mom_120", "dd_resilience_252"])
    sc = score_factors_with_weights(fac, SCORE, score_column="s")
    w = make_robust_monthly_weights(sc, px, uni, market_nav=hs(), cash_code="511880", **TIGHT)
    w["date"] = pd.to_datetime(w["date"])
    mret = px.resample("ME").last().pct_change(fill_method=None)
    me = sorted(w["date"].unique())

    # 贡献 = w(t) * 下一月收益. 聚焦 2025-01 ~ 2026-06
    rows = []
    for t in me:
        nxt = mret.index[mret.index > t]
        if len(nxt) == 0: continue
        fwd = mret.loc[nxt[0]]
        g = w[w["date"] == t]
        for _, r in g.iterrows():
            code = r["fund_code"]
            rows.append({"month": nxt[0], "code": code, "name": nm.get(code, ""),
                         "cls": cls(nm.get(code, "")), "w": r["weight"], "contrib": r["weight"] * fwd.get(code, 0.0)})
    A = pd.DataFrame(rows)
    A25 = A[(A["month"] >= "2025-01-01") & (A["month"] <= "2026-06-30")]

    print("=== 2025-2026 资产类别: 平均权重 vs 收益贡献 ===")
    comp = A25.groupby("cls").agg(平均权重=("w", lambda x: x.sum() / A25["month"].nunique()),
                                  累计贡献=("contrib", "sum")).sort_values("累计贡献", ascending=False)
    print((comp * 100).round(1).to_string())
    print(f"\n2025-2026 组合累计收益(各月贡献和): {A25.groupby('month')['contrib'].sum().add(1).prod() - 1:.1%}")

    print("\n=== 贡献最大的 top15 ETF (2025-2026) ===")
    top = A25.groupby(["code", "name", "cls"])["contrib"].sum().sort_values(ascending=False).head(15)
    for (code, name, c), v in top.items():
        print(f"  {code} {str(name)[:20]:22} {c:6} 累计贡献{v:+.1%}")

    print("\n=== 激增期 2025-07 ~ 2025-10 月度贡献分解 ===")
    surge = A[(A["month"] >= "2025-07-01") & (A["month"] <= "2025-10-31")]
    for mth, g in surge.groupby("month"):
        tot = g["contrib"].sum()
        bycls = g.groupby("cls")["contrib"].sum().sort_values(ascending=False)
        topetf = g.sort_values("contrib", ascending=False).head(3)
        print(f"  {pd.Timestamp(mth).strftime('%Y-%m')} 当月+{tot:.1%} | 类别: " +
              ", ".join(f"{k}{v:+.1%}" for k, v in bycls.items() if abs(v) > 0.003))
        print("      主力: " + "; ".join(f"{str(r['name'])[:14]}{r['contrib']:+.1%}" for _, r in topetf.iterrows()))


if __name__ == "__main__":
    main()
