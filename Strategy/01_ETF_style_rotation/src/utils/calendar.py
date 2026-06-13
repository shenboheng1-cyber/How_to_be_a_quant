"""交易日历工具。日历本身由 Choice 拉取后缓存在 data/raw/trade_calendar.parquet。"""
import pandas as pd

from .io import load_parquet


def load_trading_days() -> pd.DatetimeIndex:
    cal = load_parquet("raw", "trade_calendar")
    return pd.DatetimeIndex(sorted(pd.to_datetime(cal["date"]).unique()))


def weekly_last_trading_days(trading_days: pd.DatetimeIndex,
                             start=None, end=None) -> pd.DatetimeIndex:
    """每个自然周(周一~周日)的最后一个交易日 = 周度调仓日。"""
    s = pd.Series(trading_days, index=trading_days)
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    grouped = s.groupby(s.index.to_period("W")).max()
    return pd.DatetimeIndex(grouped.values)


def next_trading_day(trading_days: pd.DatetimeIndex, date) -> pd.Timestamp:
    """T+1 执行日: 严格大于 date 的第一个交易日。"""
    date = pd.Timestamp(date)
    later = trading_days[trading_days > date]
    if len(later) == 0:
        raise ValueError(f"{date} 之后无交易日数据")
    return later[0]


def month_end_trading_days(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(trading_days, index=trading_days)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).max().values)
