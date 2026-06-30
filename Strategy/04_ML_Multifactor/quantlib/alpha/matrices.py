# -*- coding: utf-8 -*-
"""
quantlib.alpha.matrices —— 宽矩阵数据层（因子工厂的原料）
================================================================
把日频大表转成一组【日期 × 股票】宽矩阵，供算子与 alpha 表达式使用。
价格统一用复权口径（adj = adj_close/close 调整 open/high/low/vwap），避免除权跳变。

为留足滚动窗口左侧历史，日频起点自动比 start 前推 ~400 天。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import data


class Matrices:
    """一组对齐的宽矩阵（index=交易日, columns=股票）。"""
    __slots__ = ["close", "open", "high", "low", "vwap", "volume",
                 "amount", "returns", "cap", "adv20", "dates", "stocks"]

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def load_matrices(start: str = "2015-01-01", end: str = "2025-12-31") -> Matrices:
    """构建因子工厂所需的全部宽矩阵（复权口径）。"""
    con = data.connect()
    df = con.sql(f"""
        SELECT stkcd, trddt, open, high, low, close, adj_close,
               volume, amount, ret, total_mktcap
        FROM '{data.DAILY_PARQUET}'
        WHERE trddt >= DATE '{start}' - INTERVAL 400 DAY
          AND trddt <= DATE '{end}'
    """).df()
    con.close()

    df["trddt"] = pd.to_datetime(df["trddt"])
    adj = df["adj_close"] / df["close"]            # 复权比例
    df["o"] = df["open"] * adj
    df["h"] = df["high"] * adj
    df["l"] = df["low"] * adj
    df["vw"] = (df["amount"] / df["volume"].replace(0, np.nan)) * adj

    def W(col):                                    # long -> 宽矩阵
        return df.pivot(index="trddt", columns="stkcd", values=col).sort_index()

    close = W("adj_close")
    vol = W("volume")
    amount = W("amount")
    M = Matrices(
        close=close, open=W("o"), high=W("h"), low=W("l"), vwap=W("vw"),
        volume=vol, amount=amount, returns=W("ret"), cap=W("total_mktcap"),
        adv20=amount.rolling(20, min_periods=10).mean(),   # 20日平均成交额（流动性）
    )
    M.dates = close.index
    M.stocks = close.columns
    return M
