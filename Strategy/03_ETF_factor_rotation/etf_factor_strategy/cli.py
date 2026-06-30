from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_DATA_DIR, load_etf_universe, load_hfq_market
from .engine import (
    FACTOR_WEIGHTS_V2,
    backtest_monthly_strategy,
    compute_factor_panel,
    make_monthly_weights_v2,
    score_factors_with_weights,
    summarize_performance,
)


def _load_hs300(data_dir: Path) -> pd.Series:
    con = sqlite3.connect(f"file:{data_dir / 'idx_store.db'}?mode=ro", uri=True)
    try:
        d = pd.read_sql_query(
            "SELECT date, close FROM index_daily WHERE symbol='CSI300' ORDER BY date", con)
    finally:
        con.close()
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["close"].astype(float)


def main() -> None:
    """最终版(V2) ETF 多因子月度轮动 —— 后复权市价口径，一键复现。

    口径：后复权市价(close_hfq)；信号 T 日月末收盘，T+1 执行；含成本；部分再平衡 lam。
    """
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    universe = load_etf_universe(data_dir=data_dir)
    prices, amount, _ = load_hfq_market(
        data_dir=data_dir, start=args.history_start, end=args.end, min_obs=args.min_observations)
    prices = prices.loc[: args.end]
    universe = universe[universe["fund_code"].isin(prices.columns)].copy()

    factors = compute_factor_panel(prices)
    scored = score_factors_with_weights(factors, FACTOR_WEIGHTS_V2, score_column="risk_adjusted_score")
    weights = make_monthly_weights_v2(
        scored, prices, universe,
        top_n=args.top_n, max_per_theme=args.max_per_theme, max_weight=args.max_weight,
        buffer_rank=args.buffer_rank, weighting=args.weighting,
        volatility_target=args.vol_target, cash_code=args.cash_code,
    )
    equity, effective = backtest_monthly_strategy(
        prices, weights, transaction_cost_bps=args.cost_bps, rebalance_lambda=args.lam)

    # 截到回测区间
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq[(eq["date"] >= args.start) & (eq["date"] <= args.end)].copy()
    eq["nav"] = (1.0 + eq["strategy_return"]).cumprod()
    summary = summarize_performance(eq.assign(date=eq["date"].dt.strftime("%Y-%m-%d")))

    params = {
        "basis": "close_hfq(后复权市价)", "version": "v3" if args.weighting == "minvar" else "v2",
        "factor_weights": FACTOR_WEIGHTS_V2, "weighting": args.weighting,
        "top_n": args.top_n, "max_per_theme": args.max_per_theme, "max_weight": args.max_weight,
        "buffer_rank": args.buffer_rank, "volatility_target": args.vol_target,
        "rebalance_lambda": args.lam, "cash_code": args.cash_code,
        "transaction_cost_bps": args.cost_bps, "start": args.start, "end": args.end,
    }

    enriched_weights = weights.merge(universe, on="fund_code", how="left")
    enriched_weights.to_csv(output_dir / "rebalance_weights.csv", index=False, encoding="utf-8-sig")
    eq.assign(date=eq["date"].dt.strftime("%Y-%m-%d")).to_csv(
        output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(
        json.dumps({"summary": asdict(summary), "params": params}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    print(f"ETF universe (HFQ池): {prices.shape[1]} funds | 调仓次数 {weights['date'].nunique()}")
    print(f"Outputs: {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETF 多因子月度轮动 最终版(V2) — 后复权市价口径")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-dir", default="outputs_v3_final")
    p.add_argument("--start", default="2018-01-02", help="回测起点")
    p.add_argument("--history-start", default="2016-01-01", help="多加载历史供回看因子")
    p.add_argument("--end", default="2026-06-05")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--max-per-theme", type=int, default=3)
    p.add_argument("--max-weight", type=float, default=0.12)
    p.add_argument("--buffer-rank", type=int, default=35, help="hysteresis 名次滞后带")
    p.add_argument("--weighting", default="minvar", choices=["minvar", "inv_vol", "equal"],
                   help="加权方案：minvar(V3最终/最小方差) / inv_vol(V2低回撤) / equal")
    p.add_argument("--vol-target", type=float, default=0.18)
    p.add_argument("--lam", type=float, default=0.4, help="部分再平衡系数(每月朝目标移动比例)")
    p.add_argument("--cash-code", default="511880")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--min-observations", type=int, default=280)
    return p.parse_args()


if __name__ == "__main__":
    main()
