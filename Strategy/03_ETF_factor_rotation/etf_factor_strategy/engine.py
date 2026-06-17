from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FACTOR_WEIGHTS = {
    "combo_eff_accel": 0.45,
    "momentum_12_1": 0.35,
    "fund_hit_rate_20": 0.20,
}

# 最终版(V2)合成打分：三因子 + 风险因子(总波动惩罚 / 60日回撤)
FACTOR_WEIGHTS_V2 = {
    "combo_eff_accel": 0.45,
    "momentum_12_1": 0.35,
    "fund_hit_rate_20": 0.20,
    "vol_60d": -0.15,
    "max_drawdown_60d": 0.10,
}


@dataclass(frozen=True)
class PerformanceSummary:
    start: str
    end: str
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    total_return: float


def zscore_cross_section(row: pd.Series) -> pd.Series:
    values = pd.to_numeric(row, errors="coerce")
    mean = values.mean(skipna=True)
    std = values.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return values * np.nan
    return (values - mean) / std


def compute_factor_panel(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index().astype(float)
    daily_ret = prices.pct_change(fill_method=None)
    downside_ret = daily_ret.clip(upper=0.0)
    ret20 = prices / prices.shift(20) - 1.0
    ret60 = prices / prices.shift(60) - 1.0
    abs_path20 = daily_ret.abs().rolling(20, min_periods=20).sum()

    raw = {
        "efficiency_20d": ret20 / abs_path20.replace(0.0, np.nan),
        "fund_ret_accel_20_60": ret20 - ret60 / 3.0,
        "momentum_12_1": prices.shift(21) / prices.shift(252) - 1.0,
        "fund_hit_rate_20": (daily_ret > 0).rolling(20, min_periods=20).mean(),
        "vol_60d": daily_ret.rolling(60, min_periods=40).std() * np.sqrt(252.0),
        "downside_vol_60d": downside_ret.rolling(60, min_periods=40).std() * np.sqrt(252.0),
        "max_drawdown_60d": prices / prices.rolling(60, min_periods=40).max() - 1.0,
    }
    panel = pd.concat(raw, axis=1)
    panel.columns = panel.columns.set_names(["factor", "fund_code"])
    long = panel.stack(level="fund_code", future_stack=True).reset_index()
    long = long.rename(columns={long.columns[0]: "date"})
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")

    long["combo_eff_accel"] = _datewise_z(long, "efficiency_20d") + _datewise_z(
        long, "fund_ret_accel_20_60"
    )
    return long


def score_factors(factors: pd.DataFrame) -> pd.DataFrame:
    return score_factors_with_weights(factors, FACTOR_WEIGHTS, score_column="score")


def score_factors_with_weights(
    factors: pd.DataFrame,
    factor_weights: dict[str, float],
    score_column: str = "score",
) -> pd.DataFrame:
    scored = factors.copy()
    score = pd.Series(0.0, index=scored.index)
    valid_component_count = pd.Series(0, index=scored.index)
    for factor, weight in factor_weights.items():
        z = _datewise_z(scored, factor)
        scored[f"z_{factor}"] = z
        score = score.add(z.fillna(0.0) * weight, fill_value=0.0)
        valid_component_count += z.notna().astype(int)
    scored[score_column] = score.where(valid_component_count == len(factor_weights))
    if score_column != "score":
        scored["score"] = scored[score_column]
    return scored


def make_monthly_weights(scored: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    month_end_dates = _month_end_dates(scored["date"])
    month_end = scored[scored["date"].isin(month_end_dates)].dropna(subset=["score"])
    for date, group in month_end.groupby("date", sort=True):
        picks = group.sort_values(["score", "fund_code"], ascending=[False, True]).head(top_n)
        if picks.empty:
            continue
        weight = 1.0 / len(picks)
        for code in picks["fund_code"]:
            rows.append({"date": date, "fund_code": code, "weight": weight})
    return pd.DataFrame(rows, columns=["date", "fund_code", "weight"])


def make_robust_monthly_weights(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    top_n: int = 20,
    max_per_theme: int = 3,
    max_weight: float = 0.12,
    volatility_target: float = 0.18,
    cash_code: str = "511880",
    market_nav: pd.Series | None = None,
    weak_market_exposure: float = 0.60,
    recent_nav_days: int = 5,
    max_missing_60: float = 0.10,
    max_missing_252: float = 0.20,
    weight_volatility_kind: str = "total",
) -> pd.DataFrame:
    meta = universe.copy()
    if "theme" not in meta.columns:
        meta["theme"] = meta.apply(_infer_theme, axis=1)
    enriched = scored.merge(meta, on="fund_code", how="left")
    returns = prices.pct_change(fill_method=None)
    if weight_volatility_kind == "downside":
        vol60 = returns.clip(upper=0.0).rolling(60, min_periods=40).std() * np.sqrt(252.0)
    elif weight_volatility_kind == "total":
        vol60 = returns.rolling(60, min_periods=40).std() * np.sqrt(252.0)
    else:
        raise ValueError("weight_volatility_kind must be 'total' or 'downside'")
    rows = []
    month_end_dates = _month_end_dates(enriched["date"])
    month_end = enriched[enriched["date"].isin(month_end_dates)].dropna(subset=["score"])
    for date, group in month_end.groupby("date", sort=True):
        tradable = tradable_codes_at_date(
            prices,
            pd.Timestamp(date),
            recent_days=recent_nav_days,
            max_missing_60=max_missing_60,
            max_missing_252=max_missing_252,
        )
        group = group[group["fund_code"].isin(tradable)]
        picks = _select_with_theme_cap(group, top_n=top_n, max_per_theme=max_per_theme)
        if picks.empty:
            continue
        dt = pd.Timestamp(date)
        vol_row = _last_available_row(vol60, dt)
        inv_vol = 1.0 / vol_row.reindex(picks["fund_code"]).replace([np.inf, -np.inf], np.nan)
        inv_vol = inv_vol.fillna(inv_vol.median()).replace(0.0, np.nan)
        if inv_vol.isna().all():
            raw_weights = pd.Series(1.0 / len(picks), index=picks["fund_code"])
        else:
            raw_weights = inv_vol / inv_vol.sum()
        capped = _cap_and_redistribute(raw_weights, max_weight=max_weight)

        portfolio_vol = _estimate_portfolio_vol(prices, capped, dt)
        exposure = 1.0
        if portfolio_vol and not pd.isna(portfolio_vol):
            exposure = min(exposure, volatility_target / portfolio_vol)
        if market_nav is not None and _is_weak_market(market_nav, dt):
            exposure = min(exposure, weak_market_exposure)

        risky_weights = capped * exposure
        for code, weight in risky_weights.items():
            if weight > 0:
                rows.append({"date": date, "fund_code": code, "weight": float(weight)})
        cash_weight = 1.0 - float(risky_weights.sum())
        if cash_weight > 1e-10:
            rows.append({"date": date, "fund_code": cash_code, "weight": cash_weight})
    return pd.DataFrame(rows, columns=["date", "fund_code", "weight"])


def make_monthly_weights_v2(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    top_n: int = 20,
    max_per_theme: int = 3,
    max_weight: float = 0.12,
    buffer_rank: int = 35,
    weighting: str = "inv_vol",
    volatility_target: float | None = 0.18,
    cash_code: str = "511880",
    recent_nav_days: int = 5,
    max_missing_60: float = 0.10,
    max_missing_252: float = 0.20,
) -> pd.DataFrame:
    """最终版(V2)月度目标权重：相对 robust 版去掉弱市择时、保留 vol-target，并加入名次
    滞后带(hysteresis)。持仓只要仍排在 ``buffer_rank`` 内即续持，空位才从排名最高的新名补入，
    以降低换手。配合 ``backtest_monthly_strategy(..., rebalance_lambda<1)`` 做部分再平衡。"""
    if weighting not in ("inv_vol", "equal"):
        raise ValueError("weighting must be 'inv_vol' or 'equal'")
    meta = universe.copy()
    if "theme" not in meta.columns:
        meta["theme"] = meta.apply(_infer_theme, axis=1)
    enriched = scored.merge(meta, on="fund_code", how="left")
    returns = prices.pct_change(fill_method=None)
    vol60 = returns.rolling(60, min_periods=40).std() * np.sqrt(252.0)
    rows: list[dict] = []
    prev: set[str] = set()
    month_end_dates = _month_end_dates(enriched["date"])
    month_end = enriched[enriched["date"].isin(month_end_dates)].dropna(subset=["score"])
    for date, group in month_end.groupby("date", sort=True):
        dt = pd.Timestamp(date)
        tradable = tradable_codes_at_date(prices, dt, recent_nav_days, max_missing_60, max_missing_252)
        group = group[group["fund_code"].isin(tradable)]
        if group.empty:
            continue
        ranked = group.sort_values(["score", "fund_code"], ascending=[False, True]).reset_index(drop=True)
        order = list(ranked["fund_code"])
        rank = {code: i + 1 for i, code in enumerate(order)}
        theme = dict(zip(ranked["fund_code"], ranked["theme"].astype(str)))
        selected: list[str] = []
        theme_counts: dict[str, int] = {}

        def _try_add(code: str) -> None:
            th = theme[code]
            if theme_counts.get(th, 0) >= max_per_theme:
                return
            selected.append(code)
            theme_counts[th] = theme_counts.get(th, 0) + 1

        for code in sorted([c for c in order if c in prev and rank[c] <= buffer_rank], key=lambda x: rank[x]):
            if len(selected) >= top_n:
                break
            _try_add(code)
        for code in order:
            if len(selected) >= top_n:
                break
            if code not in selected:
                _try_add(code)
        if not selected:
            continue
        picks = pd.Index(selected)
        if weighting == "equal":
            raw_weights = pd.Series(1.0 / len(picks), index=picks)
        else:
            vol_row = _last_available_row(vol60, dt)
            inv_vol = 1.0 / vol_row.reindex(picks).replace([np.inf, -np.inf], np.nan)
            inv_vol = inv_vol.fillna(inv_vol.median()).replace(0.0, np.nan)
            raw_weights = (pd.Series(1.0 / len(picks), index=picks)
                           if inv_vol.isna().all() else inv_vol / inv_vol.sum())
        capped = _cap_and_redistribute(raw_weights, max_weight=max_weight)
        exposure = 1.0
        if volatility_target is not None:
            portfolio_vol = _estimate_portfolio_vol(prices, capped, dt)
            if portfolio_vol and not pd.isna(portfolio_vol):
                exposure = min(exposure, volatility_target / portfolio_vol)
        risky_weights = capped * exposure
        for code, weight in risky_weights.items():
            if weight > 0:
                rows.append({"date": date, "fund_code": code, "weight": float(weight)})
        cash_weight = 1.0 - float(risky_weights.sum())
        if cash_weight > 1e-10:
            rows.append({"date": date, "fund_code": cash_code, "weight": cash_weight})
        prev = set(picks)
    return pd.DataFrame(rows, columns=["date", "fund_code", "weight"])


def backtest_monthly_strategy(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
    rebalance_lambda: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """月度回测。``rebalance_lambda`` ∈ (0,1] 为部分再平衡系数：每个调仓日只朝目标权重
    移动 lambda 比例(lambda=1 即完全再平衡，等同标准回测)，用于降低单次交易额/换手。
    信号 T 日(月末收盘)生成，``searchsorted(side='right')`` + ``shift(1)`` 确保 T+1 后才计收益。"""
    returns = prices.sort_index().pct_change(fill_method=None).fillna(0.0)
    weight_matrix = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    rebalance_weights = weights.copy()
    rebalance_weights["date"] = pd.to_datetime(rebalance_weights["date"])
    rebalance_weights = (
        rebalance_weights.groupby(["date", "fund_code"], as_index=False, sort=True)["weight"].sum()
    )

    targets: dict[pd.Timestamp, pd.Series] = {}
    for date, group in rebalance_weights.groupby("date", sort=True):
        idx = returns.index.searchsorted(date, side="right")
        if idx >= len(returns.index):
            continue
        effective_date = returns.index[idx]
        target = pd.Series(0.0, index=returns.columns)
        target[group["fund_code"]] = group["weight"].to_numpy()
        targets[effective_date] = target

    executed_prev = pd.Series(0.0, index=returns.columns)
    for effective_date in sorted(targets):
        executed = executed_prev + rebalance_lambda * (targets[effective_date] - executed_prev)
        weight_matrix.loc[effective_date] = executed.to_numpy()
        executed_prev = executed

    effective_weights = weight_matrix.ffill().fillna(0.0)
    holding_weights = effective_weights.shift(1).fillna(0.0)
    strategy_ret = (holding_weights * returns).sum(axis=1)

    turnover = effective_weights.diff().abs().sum(axis=1).fillna(effective_weights.abs().sum(axis=1))
    cost = turnover * (transaction_cost_bps / 10000.0)
    strategy_ret = strategy_ret - cost
    nav = (1.0 + strategy_ret).cumprod()
    equity = pd.DataFrame(
        {
            "date": returns.index.strftime("%Y-%m-%d"),
            "strategy_return": strategy_ret.to_numpy(),
            "nav": nav.to_numpy(),
            "turnover": turnover.to_numpy(),
            "cost": cost.to_numpy(),
        }
    )
    return equity, effective_weights


def tradable_codes_at_date(
    prices: pd.DataFrame,
    date: pd.Timestamp | str,
    recent_days: int = 5,
    max_missing_60: float = 0.10,
    max_missing_252: float = 0.20,
) -> set[str]:
    dt = pd.Timestamp(date)
    history = prices.sort_index().loc[:dt]
    if history.empty:
        return set()
    recent = history.tail(recent_days)
    window60 = history.tail(60)
    window252 = history.tail(252)

    has_recent_nav = recent.notna().any()
    enough_history = window252.notna().sum() >= int(np.ceil(252 * (1.0 - max_missing_252)))
    miss60_ok = window60.isna().mean() <= max_missing_60
    miss252_ok = window252.isna().mean() <= max_missing_252
    tradable = has_recent_nav & enough_history & miss60_ok & miss252_ok
    return set(tradable[tradable].index)


def summarize_performance(equity: pd.DataFrame) -> PerformanceSummary:
    if equity.empty:
        raise ValueError("equity curve is empty")
    ret = equity["strategy_return"].astype(float)
    nav = equity["nav"].astype(float)
    days = max(len(equity), 1)
    total_return = nav.iloc[-1] - 1.0
    annual_return = nav.iloc[-1] ** (252.0 / days) - 1.0
    annual_vol = ret.std(ddof=0) * np.sqrt(252.0)
    sharpe = annual_return / annual_vol if annual_vol else np.nan
    drawdown = nav / nav.cummax() - 1.0
    return PerformanceSummary(
        start=str(equity["date"].iloc[0]),
        end=str(equity["date"].iloc[-1]),
        annual_return=float(annual_return),
        annual_volatility=float(annual_vol),
        sharpe=float(sharpe),
        max_drawdown=float(drawdown.min()),
        total_return=float(total_return),
    )


def _datewise_z(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date", sort=False)[column].transform(zscore_cross_section)


def _month_end_dates(dates: pd.Series) -> set[str]:
    date_index = pd.to_datetime(dates)
    month_key = date_index.dt.to_period("M")
    ends = pd.Series(dates.to_numpy(), index=month_key).groupby(level=0).max()
    return set(ends.astype(str))


def _infer_theme(row: pd.Series) -> str:
    name = str(row.get("fund_name") or "")
    fund_type = str(row.get("fund_type") or "")
    rules = [
        ("半导体", "半导体"),
        ("芯片", "半导体"),
        ("通信", "通信"),
        ("5G", "通信"),
        ("稀有金属", "金属资源"),
        ("有色", "金属资源"),
        ("黄金", "黄金"),
        ("金ETF", "黄金"),
        ("酒", "消费"),
        ("食品", "消费"),
        ("消费", "消费"),
        ("医药", "医药"),
        ("医疗", "医药"),
        ("新能源", "新能源"),
        ("电池", "新能源"),
        ("光伏", "新能源"),
        ("证券", "金融"),
        ("银行", "金融"),
        ("金融", "金融"),
        ("债", "固收"),
        ("日利", "货币现金"),
        ("货币", "货币现金"),
    ]
    for key, theme in rules:
        if key in name:
            return theme
    if "固收" in fund_type or "债券" in fund_type:
        return "固收"
    if "货币" in fund_type:
        return "货币现金"
    if "海外" in fund_type or "QDII" in fund_type.upper():
        return "海外"
    return fund_type or "其他"


def _select_with_theme_cap(group: pd.DataFrame, top_n: int, max_per_theme: int) -> pd.DataFrame:
    selected = []
    theme_counts: dict[str, int] = {}
    ranked = group.sort_values(["score", "fund_code"], ascending=[False, True])
    for _, row in ranked.iterrows():
        theme = str(row["theme"])
        if theme_counts.get(theme, 0) >= max_per_theme:
            continue
        selected.append(row)
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
        if len(selected) >= top_n:
            break
    if not selected:
        return group.iloc[0:0]
    return pd.DataFrame(selected)


def _cap_and_redistribute(weights: pd.Series, max_weight: float) -> pd.Series:
    weights = weights.astype(float).copy()
    if weights.empty:
        return weights
    weights = weights / weights.sum()
    capped = pd.Series(0.0, index=weights.index)
    remaining = weights.copy()
    remaining_total = 1.0
    while not remaining.empty:
        scaled = remaining / remaining.sum() * remaining_total
        over = scaled > max_weight
        if not over.any():
            capped.loc[scaled.index] = scaled
            break
        capped.loc[scaled[over].index] = max_weight
        remaining_total -= max_weight * int(over.sum())
        remaining = remaining[~over]
    return capped


def _last_available_row(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
    eligible = frame.loc[frame.index <= date]
    if eligible.empty:
        return pd.Series(dtype=float)
    return eligible.iloc[-1]


def _estimate_portfolio_vol(prices: pd.DataFrame, weights: pd.Series, date: pd.Timestamp) -> float:
    returns = prices[list(weights.index)].pct_change(fill_method=None)
    window = returns.loc[returns.index <= date].tail(60)
    if len(window) < 20:
        return float("nan")
    portfolio_ret = (window.fillna(0.0) * weights).sum(axis=1)
    return float(portfolio_ret.std(ddof=0) * np.sqrt(252.0))


def _is_weak_market(market_nav: pd.Series, date: pd.Timestamp) -> bool:
    series = market_nav.loc[market_nav.index <= date].dropna()
    if len(series) < 200:
        return False
    latest = series.iloc[-1]
    ma200 = series.rolling(200).mean().iloc[-1]
    ret60 = latest / series.iloc[-61] - 1.0 if len(series) > 60 else 0.0
    return bool(latest < ma200 or ret60 < 0.0)
