"""指数 -> ETF 映射与权重计算 (报告 四.(二) ETF筛选4步)。"""
from __future__ import annotations

import pandas as pd


def select_etf_for_index(index_code: str,
                         rebalance_date: pd.Timestamp,
                         etf_info: pd.DataFrame,
                         etf_amount: pd.DataFrame,
                         trading_days: pd.DatetimeIndex,
                         min_listed_days: int = 30,
                         liquidity_window: int = 30) -> str | None:
    """返回该指数应持有的 ETF 代码; 无合格 ETF 返回 None。

    etf_info:   [code, list_date, tracking_index]
    etf_amount: index=date, columns=etf_code, 值=日成交额
    """
    cand = etf_info[etf_info["tracking_index"] == index_code]
    if cand.empty:
        return None
    days_upto = trading_days[trading_days <= rebalance_date]
    ok_codes = []
    for _, row in cand.iterrows():
        listed = days_upto[days_upto >= pd.Timestamp(row["list_date"])]
        if len(listed) >= min_listed_days:
            ok_codes.append(row["code"])
    ok_codes = [c for c in ok_codes if c in etf_amount.columns]
    if not ok_codes:
        return None
    recent = etf_amount.loc[etf_amount.index <= rebalance_date, ok_codes].tail(liquidity_window)
    avg_amt = recent.mean()
    if avg_amt.isna().all():
        return None
    return avg_amt.idxmax()


def build_target_weights(selected_indices: list,
                         norm_score: pd.Series,
                         index_to_etf: dict) -> pd.Series:
    """ETF 权重 ∝ 其追踪指数的 Norm_score, 归一化到 100%。无ETF的指数剔除。"""
    pairs = {index_to_etf[i]: norm_score[i]
             for i in selected_indices if index_to_etf.get(i) is not None}
    w = pd.Series(pairs, dtype=float)
    if w.empty:
        return w
    if w.sum() <= 0:                 # norm_score 可能全为0(被选中者含最低分)
        return pd.Series(1.0 / len(w), index=w.index)
    return w / w.sum()
