"""读取 notebook 取数落地的 parquet。所有 src 内部代码只依赖这里, 不直接碰 Choice API。

约定的数据文件 (由 notebooks/01_choice_data_fetch.ipynb 产生, 详见 docs/DATA_FIELDS.md):
  raw/trade_calendar          : [date]
  raw/stock_universe          : [code, name, list_date, delist_date]
  raw/stock_daily             : [date, code, close_adj, float_mv, total_mv, turnover, amount]
  raw/stock_industry          : [date, code, industry]           (月度快照即可)
  raw/stock_financials        : [report_date, code, <财务字段...>]
  raw/index_daily             : [date, code, close]
  raw/index_constituents      : [date, index_code, code, weight] (月末)
  raw/etf_info                : [code, name, list_date, tracking_index]
  raw/etf_daily               : [date, code, nav, close, amount]
  raw/macro_<indicator_key>   : [date, value]                    (每个宏观指标一个文件)
"""
import pandas as pd

from ..utils.io import load_parquet


def stock_daily() -> pd.DataFrame:
    df = load_parquet("raw", "stock_daily")
    df["date"] = pd.to_datetime(df["date"])
    return df


def index_daily() -> pd.DataFrame:
    df = load_parquet("raw", "index_daily")
    df["date"] = pd.to_datetime(df["date"])
    return df


def index_constituents() -> pd.DataFrame:
    df = load_parquet("raw", "index_constituents")
    df["date"] = pd.to_datetime(df["date"])
    return df


def etf_info() -> pd.DataFrame:
    return load_parquet("raw", "etf_info")


def etf_daily() -> pd.DataFrame:
    df = load_parquet("raw", "etf_daily")
    df["date"] = pd.to_datetime(df["date"])
    return df


def macro_indicator(key: str) -> pd.Series:
    df = load_parquet("raw", f"macro_{key}")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].sort_index()
