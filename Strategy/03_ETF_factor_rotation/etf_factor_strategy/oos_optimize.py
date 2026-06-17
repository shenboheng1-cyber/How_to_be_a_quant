from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DEFAULT_DATA_DIR, load_etf_universe, load_nav_prices
from .engine import (
    backtest_monthly_strategy,
    compute_factor_panel,
    make_robust_monthly_weights,
    score_factors,
    score_factors_with_weights,
)


BASE_ALPHA_WEIGHTS = {
    "combo_eff_accel": 0.45,
    "momentum_12_1": 0.35,
    "fund_hit_rate_20": 0.20,
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    universe = load_etf_universe(data_dir=data_dir)
    prices = load_nav_prices(
        universe["fund_code"].tolist(),
        data_dir=data_dir,
        start=args.history_start,
        end=args.end,
    ).dropna(axis=1, thresh=args.min_observations)
    universe = universe[universe["fund_code"].isin(prices.columns)].copy()
    market_nav = load_hs300(data_dir)

    factors = compute_factor_panel(prices)
    factors = factors[factors["date"].isin(month_end_dates(factors["date"]))].copy()
    rows = []
    nav_curves = {}
    for vol_penalty, drawdown_penalty in itertools.product(args.vol_penalties, args.drawdown_penalties):
        weights = {
            **BASE_ALPHA_WEIGHTS,
            "vol_60d": -float(vol_penalty),
            "max_drawdown_60d": float(drawdown_penalty),
        }
        scored = score_factors_with_weights(factors, weights, score_column="risk_adjusted_score")
        rebalance = make_robust_monthly_weights(
            scored,
            prices,
            universe,
            top_n=args.top_n,
            max_per_theme=args.max_per_theme,
            max_weight=args.max_weight,
            volatility_target=args.volatility_target,
            cash_code=args.cash_code,
            market_nav=market_nav,
            weak_market_exposure=args.weak_market_exposure,
        )
        equity, _ = backtest_monthly_strategy(prices, rebalance)
        label = f"vol_{vol_penalty:.2f}_dd_{drawdown_penalty:.2f}"
        nav_curves[label] = equity
        train_metrics = period_metrics(equity, args.train_start, args.train_end)
        test_metrics = period_metrics(equity, args.test_start, args.test_end)
        rows.append(
            {
                "label": label,
                "vol_penalty": vol_penalty,
                "drawdown_penalty": drawdown_penalty,
                **prefix_metrics("train", train_metrics),
                **prefix_metrics("test", test_metrics),
            }
        )

    result = pd.DataFrame(rows)
    result["selection_score"] = (
        result["train_calmar"].rank(ascending=False, pct=True)
        + result["train_sharpe_rf0"].rank(ascending=False, pct=True)
        + result["train_annual_return"].rank(ascending=False, pct=True)
        - result["train_max_drawdown"].abs().rank(ascending=True, pct=True)
    )
    result = result.sort_values(
        ["selection_score", "train_calmar", "train_sharpe_rf0", "train_annual_return"],
        ascending=[False, False, False, False],
    )
    result.to_csv(output_dir / "oos_parameter_grid.csv", index=False)

    best = result.iloc[0].to_dict()
    best_equity = nav_curves[str(best["label"])]
    best_equity.to_csv(output_dir / "best_equity_curve_full.csv", index=False)
    (output_dir / "best_params.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Grid results: {output_dir / 'oos_parameter_grid.csv'}")


def load_hs300(data_dir: Path) -> pd.Series:
    db_path = data_dir / "idx_store.db"
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        data = pd.read_sql_query(
            "SELECT date, close FROM index_daily WHERE symbol='CSI300' ORDER BY date",
            con,
        )
    finally:
        con.close()
    data["date"] = pd.to_datetime(data["date"])
    return data.set_index("date")["close"].astype(float)


def period_metrics(equity: pd.DataFrame, start: str, end: str) -> dict[str, float | str | int]:
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    if frame.empty:
        raise ValueError(f"empty evaluation period: {start} to {end}")
    nav = frame["nav"].astype(float)
    nav = nav / nav.iloc[0]
    ret = nav.pct_change().fillna(0.0)
    days = len(nav)
    total_return = nav.iloc[-1] - 1.0
    annual_return = nav.iloc[-1] ** (252.0 / days) - 1.0
    annual_volatility = ret.std(ddof=0) * np.sqrt(252.0)
    max_drawdown = (nav / nav.cummax() - 1.0).min()
    return {
        "start": frame["date"].iloc[0].strftime("%Y-%m-%d"),
        "end": frame["date"].iloc[-1].strftime("%Y-%m-%d"),
        "trading_days": int(days),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_rf0": float(annual_return / annual_volatility) if annual_volatility else np.nan,
        "max_drawdown": float(max_drawdown),
        "calmar": float(annual_return / abs(max_drawdown)) if max_drawdown else np.nan,
    }


def prefix_metrics(prefix: str, metrics: dict[str, float | str | int]) -> dict[str, float | str | int]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def month_end_dates(dates: pd.Series) -> set[str]:
    date_index = pd.to_datetime(dates)
    month_key = date_index.dt.to_period("M")
    ends = pd.Series(dates.to_numpy(), index=month_key).groupby(level=0).max()
    return set(ends.astype(str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Out-of-sample risk-adjusted ETF factor search")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default="outputs_oos_risk_adjusted")
    parser.add_argument("--history-start", default="2017-01-01")
    parser.add_argument("--end", default="2026-06-05")
    parser.add_argument("--train-start", default="2020-02-04")
    parser.add_argument("--train-end", default="2023-12-29")
    parser.add_argument("--test-start", default="2024-01-02")
    parser.add_argument("--test-end", default="2026-06-05")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-per-theme", type=int, default=3)
    parser.add_argument("--max-weight", type=float, default=0.12)
    parser.add_argument("--volatility-target", type=float, default=0.18)
    parser.add_argument("--weak-market-exposure", type=float, default=0.60)
    parser.add_argument("--cash-code", default="511880")
    parser.add_argument("--min-observations", type=int, default=280)
    parser.add_argument("--vol-penalties", nargs="+", type=float, default=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25])
    parser.add_argument(
        "--drawdown-penalties",
        nargs="+",
        type=float,
        default=[0.00, 0.05, 0.10, 0.15, 0.20],
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
