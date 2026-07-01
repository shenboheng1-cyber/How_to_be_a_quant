# -*- coding: utf-8 -*-
"""
quantlib.factor_timing —— 因子择时（动态因子权重）
================================================================
文献主流的【内生】因子择时：按因子自身近期表现动态赋权，而非对最终组合降仓。
- 因子动量(time-series)：近 W 月平均 IC 作权重（顾明等：A股因子动量由情绪驱动，证据强）。
- IC_IR 加权：近 W 月 IC 均值/标准差（更稳）。
- 对照 baseline：等权 / 全样本静态IC（peeks，仅参照）/ 扩张窗IC。

铁律(知乎复现教训)：择时必须【样本外、扣成本后】跑赢静态 baseline 才算数。
全程 PIT：权重只用已实现的历史 IC（rolling().shift(1)）。
参考：Arnott "Factor Timing: Keep It Simple"（动量+估值最稳）、Haddad-Kozak(2020,SDF)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import preprocess, evaluate


def factor_panel(panel, factor_funcs: dict, neutralize: bool = True):
    """返回 (fvals: {名:预处理后因子值Series}, ic_df: 月度IC 表 index=调仓日,col=因子)。"""
    fvals = {nm: preprocess.preprocess_factor(panel, fn(panel), do_neutralize=neutralize)
             for nm, fn in factor_funcs.items()}
    ic = {nm: evaluate.compute_ic(panel, fv) for nm, fv in fvals.items()}
    return fvals, pd.DataFrame(ic)


def weights(ic_df: pd.DataFrame, scheme: str, window: int = 12) -> pd.DataFrame:
    """各择时方案的因子权重(行=调仓日)。除 static 外都 shift(1) 保证 PIT。"""
    if scheme == "equal":
        return pd.DataFrame(1.0, index=ic_df.index, columns=ic_df.columns)
    if scheme == "static":                                    # 全样本IC(偷看未来,仅作上限参照)
        return pd.DataFrame(np.tile(ic_df.mean().values, (len(ic_df), 1)),
                            index=ic_df.index, columns=ic_df.columns)
    if scheme == "expanding":                                 # 扩张窗历史IC
        return ic_df.expanding(min_periods=12).mean().shift(1)
    if scheme == "mom":                                       # 因子动量：近W月平均IC
        return ic_df.rolling(window, min_periods=6).mean().shift(1)
    if scheme == "icir":                                      # 近W月 IC_IR
        m = ic_df.rolling(window, min_periods=6).mean()
        s = ic_df.rolling(window, min_periods=6).std()
        return (m / s).shift(1)
    raise ValueError(scheme)


def composite(panel, fvals: dict, w_df: pd.DataFrame) -> pd.Series:
    """按每月因子权重合成截面信号：sum_f w_{f,t}·factor_f(stock,t)。"""
    comp = np.zeros(len(panel))
    dt = panel["trddt"]
    for nm, fv in fvals.items():
        wr = dt.map(w_df[nm]).values
        comp += np.nan_to_num(wr) * np.nan_to_num(fv.values)
    return pd.Series(comp, index=panel.index, name="timed_composite")
