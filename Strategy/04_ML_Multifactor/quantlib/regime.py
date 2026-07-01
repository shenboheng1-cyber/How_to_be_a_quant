# -*- coding: utf-8 -*-
"""
quantlib.regime —— 因子择时 / regime（降回撤）
================================================================
两类 regime 信号 + 仓位叠加，全用滞后值(只看过去)，无前视：
  vol_target   波动目标：组合近期实现波动越高→仓位越低
  crowding     拥挤度：因子收益两两相关上升=拥挤=踩踏风险(2024小微盘那种)
  derisk       组合：拥挤/高波 regime 下降仓
目标：在不大损收益的前提下，削掉系统性 alpha 回撤。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def vol_target(returns: pd.Series, target_ann: float = 0.12,
               lookback: int = 6, cap: float = 1.5, ppy: int = 12) -> pd.Series:
    """波动目标仓位(滞后)：scale = 目标年化波动 / 近期实现年化波动，封顶 cap。"""
    realized = returns.rolling(lookback, min_periods=3).std().shift(1) * np.sqrt(ppy)
    scale = (target_ann / realized).clip(upper=cap).fillna(1.0)
    return scale


def crowding_index(factor_rets: pd.DataFrame, lookback: int = 12) -> pd.Series:
    """拥挤度 = 过去 lookback 期因子收益两两相关的均值(上升=因子趋同=拥挤)。"""
    cols = factor_rets.columns
    vals = []
    for i in range(len(factor_rets)):
        if i < lookback:
            vals.append(np.nan); continue
        c = factor_rets.iloc[i - lookback:i].corr().values
        vals.append(np.nanmean(c[np.triu_indices(len(cols), 1)]))
    return pd.Series(vals, index=factor_rets.index)


def derisk(returns: pd.Series, signal: pd.Series, hi_quantile: float = 0.8,
           low_expo: float = 0.5) -> pd.Series:
    """regime 叠加：signal(如拥挤度)进入历史高位时降仓到 low_expo。仓位用滞后信号。"""
    thr = signal.expanding(min_periods=12).quantile(hi_quantile)
    expo = pd.Series(1.0, index=returns.index)
    expo[signal.shift(1) > thr.shift(1)] = low_expo
    return expo
