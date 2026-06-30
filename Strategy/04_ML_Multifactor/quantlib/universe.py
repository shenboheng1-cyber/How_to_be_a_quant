# -*- coding: utf-8 -*-
"""
quantlib.universe —— 股票池构建
================================================================
给研究面板的每一行打上 in_universe 标记：该股在该调仓日是否"可纳入组合"。

剔除四类（A股回测的经典陷阱，少处理一个，结果就虚高）：
  1. ST/*ST     —— 退市风险、流动性差、规则特殊；is_st=True
  2. 次新股      —— 上市不满 N 个交易日，价格未稳定、易被炒作；用上市日算
  3. 停牌        —— 当日无法成交。本框架中"停牌=快照里没有这一行"，已自动剔除
  4. 涨跌停      —— 调仓日封死涨停买不进 / 跌停卖不出；limit_status ∈ {1,-1}

设计：不直接删行，而是加 in_universe 布尔列（透明、可诊断每类剔除多少）。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from . import data


def _trading_calendar() -> np.ndarray:
    """全市场交易日数组（升序），用于精确计算"上市以来的交易日数"。"""
    con = data.connect()
    cal = con.sql(
        f"SELECT DISTINCT trddt FROM '{data.DAILY_PARQUET}' ORDER BY trddt"
    ).df()["trddt"].values
    con.close()
    return cal.astype("datetime64[ns]")


def _listing_first_day() -> pd.DataFrame:
    """每只股票的首个交易日（来自 dim_stock），用于判定次新。"""
    dim = os.path.join(data.MART_DIR, "dim_stock.parquet")
    con = data.connect()
    df = con.sql(f"SELECT stkcd, first_trddt FROM '{dim}'").df()
    con.close()
    df["first_trddt"] = pd.to_datetime(df["first_trddt"])
    return df


def add_universe(panel: pd.DataFrame, min_list_days: int = 120,
                 exclude_st: bool = True, exclude_limit: bool = True,
                 verbose: bool = True) -> pd.DataFrame:
    """给 panel 增加 in_universe 列（及各剔除原因列，便于诊断）。

    min_list_days: 次新阈值，单位=交易日（默认120≈半年）。用真实交易日历精确计数。
    """
    out = panel.copy()

    # --- 次新：上市以来的交易日数 = 交易日历中 [first_trddt, trddt] 的交易日个数 ---
    cal = _trading_calendar()
    first = _listing_first_day()
    out = out.merge(first, on="stkcd", how="left")
    # searchsorted：每个日期在交易日历中的序号；两序号之差=区间内交易日数
    idx_now = np.searchsorted(cal, out["trddt"].values.astype("datetime64[ns]"), side="right")
    idx_ipo = np.searchsorted(cal, out["first_trddt"].values.astype("datetime64[ns]"), side="left")
    out["list_days"] = idx_now - idx_ipo
    is_new = out["list_days"] < min_list_days

    # --- ST ---
    is_st = out["is_st"].fillna(False).astype(bool) if exclude_st else pd.Series(False, index=out.index)

    # --- 涨跌停：limit_status ∈ {1,-1} 视为当日不可成交；0 或 NULL 可交易 ---
    if exclude_limit:
        at_limit = out["limit_status"].isin([1, -1])
    else:
        at_limit = pd.Series(False, index=out.index)

    out["in_universe"] = ~(is_new | is_st | at_limit)
    # 记录剔除原因（诊断用）
    out["drop_new"], out["drop_st"], out["drop_limit"] = is_new, is_st, at_limit

    if verbose:
        n = len(out)
        print(f"[universe] 总行数 {n:,}")
        print(f"  次新剔除  {is_new.sum():>9,} ({is_new.mean():.1%})")
        print(f"  ST 剔除   {is_st.sum():>9,} ({is_st.mean():.1%})")
        print(f"  涨跌停剔除 {at_limit.sum():>9,} ({at_limit.mean():.1%})")
        print(f"  保留      {out['in_universe'].sum():>9,} ({out['in_universe'].mean():.1%})")
    return out


def filter_universe(panel: pd.DataFrame, **kw) -> pd.DataFrame:
    """便捷封装：加标记并只返回 in_universe=True 的行。"""
    p = add_universe(panel, **kw)
    return p[p["in_universe"]].reset_index(drop=True)
