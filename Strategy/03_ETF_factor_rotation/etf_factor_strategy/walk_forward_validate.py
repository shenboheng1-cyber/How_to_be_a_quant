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
    score_factors_with_weights,
)


FACTOR_PRESETS = {
    "base": {"combo_eff_accel": 0.45, "momentum_12_1": 0.35, "fund_hit_rate_20": 0.20},
    "mom_heavy": {"combo_eff_accel": 0.35, "momentum_12_1": 0.45, "fund_hit_rate_20": 0.20},
    "hit_quality": {"combo_eff_accel": 0.40, "momentum_12_1": 0.30, "fund_hit_rate_20": 0.30},
    "eff_heavy": {"combo_eff_accel": 0.55, "momentum_12_1": 0.30, "fund_hit_rate_20": 0.15},
}

RISK_PRESETS = {
    "risk_light": {"vol_60d": -0.15, "max_drawdown_60d": 0.10},
    "risk_balanced": {"vol_60d": -0.20, "max_drawdown_60d": 0.20},
    "risk_defensive": {"vol_60d": -0.25, "max_drawdown_60d": 0.20},
}

PORTFOLIO_PRESETS = {
    "balanced": {
        "top_n": 20,
        "max_per_theme": 3,
        "max_weight": 0.12,
        "volatility_target": 0.18,
        "weak_market_exposure": 0.60,
    },
    "defensive": {
        "top_n": 25,
        "max_per_theme": 3,
        "max_weight": 0.10,
        "volatility_target": 0.15,
        "weak_market_exposure": 0.50,
    },
}

FOLDS = [
    ("2020_2022_to_2023", "2020-02-04", "2022-12-30", "2023-01-03", "2023-12-29"),
    ("2020_2023_to_2024", "2020-02-04", "2023-12-29", "2024-01-02", "2024-12-31"),
    ("2020_2024_to_2025", "2020-02-04", "2024-12-31", "2025-01-02", "2025-12-31"),
    ("2020_2025_to_2026", "2020-02-04", "2025-12-31", "2026-01-02", "2026-06-05"),
]


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
        ffill=False,
    ).dropna(axis=1, thresh=args.min_observations)
    universe = universe[universe["fund_code"].isin(prices.columns)].copy()
    market_nav = load_hs300(data_dir)

    factors = compute_factor_panel(prices)
    factors = factors[factors["date"].isin(month_end_dates(factors["date"]))].copy()
    candidates = build_candidates()

    all_rows = []
    chosen_rows = []
    fold_curves = []
    for fold_name, train_start, train_end, test_start, test_end in FOLDS:
        fold_rows = []
        for candidate in candidates:
            equity = run_candidate(candidate, factors, prices, universe, market_nav, args.cost_bps)
            train_metrics = period_metrics(equity, train_start, train_end)
            test_metrics = period_metrics(equity, test_start, test_end)
            row = {
                "fold": fold_name,
                **candidate["flat"],
                **prefix_metrics("train", train_metrics),
                **prefix_metrics("test", test_metrics),
            }
            row["selection_score"] = selection_score(row)
            fold_rows.append(row)
            all_rows.append(row)
        fold_result = pd.DataFrame(fold_rows).sort_values(
            ["selection_score", "train_calmar", "train_sharpe_rf0", "train_annual_return"],
            ascending=[False, False, False, False],
        )
        best = fold_result.iloc[0].to_dict()
        chosen_rows.append(best)
        best_candidate = next(c for c in candidates if c["flat"]["candidate_id"] == best["candidate_id"])
        best_equity = run_candidate(best_candidate, factors, prices, universe, market_nav, args.cost_bps)
        test_curve = slice_and_normalize(best_equity, test_start, test_end)
        test_curve["fold"] = fold_name
        fold_curves.append(test_curve)

    all_result = pd.DataFrame(all_rows)
    chosen = pd.DataFrame(chosen_rows)
    stitched = stitch_fold_curves(fold_curves)

    all_result.to_csv(output_dir / "walk_forward_all_candidates.csv", index=False)
    chosen.to_csv(output_dir / "walk_forward_chosen_by_fold.csv", index=False)
    stitched.to_csv(output_dir / "walk_forward_stitched_test_curve.csv", index=False)
    summary = {
        "cost_bps": args.cost_bps,
        "tradability_filter": {
            "recent_nav_days": 5,
            "max_missing_60": 0.10,
            "max_missing_252": 0.20,
            "note": "Universe is filtered point-in-time within the current ETF roster; historical delisted ETFs cannot be recovered from current roster alone.",
        },
        "stitched_test": period_metrics_from_nav(stitched["date"], stitched["stitched_nav"]),
        "chosen_by_fold": chosen.to_dict(orient="records"),
    }
    (output_dir / "walk_forward_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["stitched_test"], ensure_ascii=False, indent=2))
    print(output_dir / "walk_forward_chosen_by_fold.csv")


def build_candidates() -> list[dict]:
    candidates = []
    for factor_name, risk_name, portfolio_name in itertools.product(
        FACTOR_PRESETS, RISK_PRESETS, PORTFOLIO_PRESETS
    ):
        factor_weights = {**FACTOR_PRESETS[factor_name], **RISK_PRESETS[risk_name]}
        portfolio = PORTFOLIO_PRESETS[portfolio_name]
        candidate_id = f"{factor_name}__{risk_name}__{portfolio_name}"
        candidates.append(
            {
                "factor_weights": factor_weights,
                "portfolio": portfolio,
                "flat": {
                    "candidate_id": candidate_id,
                    "factor_preset": factor_name,
                    "risk_preset": risk_name,
                    "portfolio_preset": portfolio_name,
                    **{f"w_{k}": v for k, v in factor_weights.items()},
                    **portfolio,
                },
            }
        )
    return candidates


def run_candidate(
    candidate: dict,
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    market_nav: pd.Series,
    cost_bps: float,
) -> pd.DataFrame:
    scored = score_factors_with_weights(
        factors,
        candidate["factor_weights"],
        score_column="risk_adjusted_score",
    )
    weights = make_robust_monthly_weights(
        scored,
        prices,
        universe,
        cash_code="511880",
        market_nav=market_nav,
        **candidate["portfolio"],
    )
    equity, _ = backtest_monthly_strategy(prices, weights, transaction_cost_bps=cost_bps)
    return equity


def load_hs300(data_dir: Path) -> pd.Series:
    con = sqlite3.connect(f"file:{data_dir / 'idx_store.db'}?mode=ro", uri=True)
    try:
        data = pd.read_sql_query(
            "SELECT date, close FROM index_daily WHERE symbol='CSI300' ORDER BY date",
            con,
        )
    finally:
        con.close()
    data["date"] = pd.to_datetime(data["date"])
    return data.set_index("date")["close"].astype(float)


def period_metrics(equity: pd.DataFrame, start: str, end: str) -> dict:
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    return period_metrics_from_nav(frame["date"], frame["nav"] / frame["nav"].iloc[0])


def period_metrics_from_nav(dates: pd.Series, nav: pd.Series) -> dict:
    nav = pd.Series(nav).astype(float).reset_index(drop=True)
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    ret = nav.pct_change().fillna(0.0)
    days = len(nav)
    total_return = nav.iloc[-1] - 1.0
    annual_return = nav.iloc[-1] ** (252.0 / days) - 1.0
    annual_volatility = ret.std(ddof=0) * np.sqrt(252.0)
    max_drawdown = (nav / nav.cummax() - 1.0).min()
    return {
        "start": dates.iloc[0].strftime("%Y-%m-%d"),
        "end": dates.iloc[-1].strftime("%Y-%m-%d"),
        "trading_days": int(days),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_rf0": float(annual_return / annual_volatility) if annual_volatility else np.nan,
        "max_drawdown": float(max_drawdown),
        "calmar": float(annual_return / abs(max_drawdown)) if max_drawdown else np.nan,
    }


def selection_score(row: dict) -> float:
    return (
        float(row["train_calmar"])
        + 0.5 * float(row["train_sharpe_rf0"])
        + 0.5 * float(row["train_annual_return"])
        - 0.5 * abs(float(row["train_max_drawdown"]))
    )


def prefix_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def month_end_dates(dates: pd.Series) -> set[str]:
    date_index = pd.to_datetime(dates)
    month_key = date_index.dt.to_period("M")
    ends = pd.Series(dates.to_numpy(), index=month_key).groupby(level=0).max()
    return set(ends.astype(str))


def slice_and_normalize(equity: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    frame["nav"] = frame["nav"] / frame["nav"].iloc[0]
    return frame[["date", "nav"]]


def stitch_fold_curves(curves: list[pd.DataFrame]) -> pd.DataFrame:
    nav_base = 1.0
    rows = []
    for curve in curves:
        curve = curve.sort_values("date").copy()
        curve["stitched_nav"] = nav_base * curve["nav"]
        nav_base = float(curve["stitched_nav"].iloc[-1])
        rows.append(curve[["date", "fold", "stitched_nav"]])
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward ETF factor validation")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default="outputs_walk_forward_hardened")
    parser.add_argument("--history-start", default="2017-01-01")
    parser.add_argument("--end", default="2026-06-05")
    parser.add_argument("--min-observations", type=int, default=280)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
