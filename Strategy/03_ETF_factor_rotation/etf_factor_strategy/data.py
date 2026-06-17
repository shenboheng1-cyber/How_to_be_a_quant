from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("/Users/shenboheng/Documents/ClaudeCode/dataset/基金深度分析")


def load_etf_universe(data_dir: Path = DEFAULT_DATA_DIR, include_feeder: bool = False) -> pd.DataFrame:
    path = data_dir / "bulk_universe.json"
    raw = json.loads(path.read_text(encoding="utf-8"))["data"]
    rows = []
    for code, item in raw.items():
        name = str(item.get("name", ""))
        fund_type = str(item.get("type", ""))
        upper_name = name.upper()
        is_etf = "ETF" in upper_name or "交易型开放式" in name
        is_feeder = "联接" in name or "ETF连接" in upper_name
        if is_etf and (include_feeder or not is_feeder):
            rows.append({"fund_code": code, "fund_name": name, "fund_type": fund_type})
    return pd.DataFrame(rows).sort_values("fund_code").reset_index(drop=True)


def load_nav_prices(
    fund_codes: list[str],
    data_dir: Path = DEFAULT_DATA_DIR,
    start: str | None = None,
    end: str | None = None,
    ffill: bool = False,
) -> pd.DataFrame:
    if not fund_codes:
        raise ValueError("fund_codes is empty")
    db_path = data_dir / "nav_store.db"
    placeholders = ",".join("?" for _ in fund_codes)
    params: list[str] = list(fund_codes)
    where = [f"fund_code IN ({placeholders})", "cum_nav IS NOT NULL"]
    if start:
        where.append("date >= ?")
        params.append(start)
    if end:
        where.append("date <= ?")
        params.append(end)
    sql = f"""
        SELECT fund_code, date, cum_nav
        FROM fund_nav
        WHERE {' AND '.join(where)}
        ORDER BY date, fund_code
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        nav = pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()
    if nav.empty:
        raise ValueError("no NAV rows found for selected universe and date range")
    nav["date"] = pd.to_datetime(nav["date"])
    prices = nav.pivot(index="date", columns="fund_code", values="cum_nav").sort_index()
    return prices.ffill() if ffill else prices


def load_hfq_market(
    data_dir: Path = DEFAULT_DATA_DIR,
    start: str | None = None,
    end: str | None = None,
    min_obs: int = 280,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """后复权市价口径数据(iFinD)：返回 (后复权收盘价, 成交额, 贴水率) 三张 date×fund_code 面板。

    数据来自 ``etf_market_ifind.db`` 的 ``etf_quote`` 表(由 ifind_etf_history.ipynb 抓取)。
    后复权收盘价 ``close_hfq`` 用于回测/因子；成交额 ``amount`` 用于流动性/冲击；贴水率
    ``premiumRatio`` 用于折溢价过滤。价格面板按 ``min_obs`` 过滤历史过短的标的。
    """
    db_path = data_dir / "etf_market_ifind.db"
    where = ["close_hfq IS NOT NULL"]
    params: list[str] = []
    if start:
        where.append("date >= ?")
        params.append(start)
    if end:
        where.append("date <= ?")
        params.append(end)
    sql = f"""
        SELECT fund_code, date, close_hfq, amount, premiumRatio
        FROM etf_quote
        WHERE {' AND '.join(where)}
        ORDER BY date, fund_code
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()
    if df.empty:
        raise ValueError("no HFQ rows found for selected date range")
    df["fund_code"] = df["fund_code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    px = df.pivot_table(index="date", columns="fund_code", values="close_hfq").sort_index()
    amt = df.pivot_table(index="date", columns="fund_code", values="amount").sort_index()
    prem = df.pivot_table(index="date", columns="fund_code", values="premiumRatio").sort_index()
    px = px.dropna(axis=1, thresh=min_obs)
    return px, amt.reindex(columns=px.columns), prem.reindex(columns=px.columns)
