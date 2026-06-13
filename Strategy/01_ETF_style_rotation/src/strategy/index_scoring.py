"""信号 -> 指数打分 (报告 四.(二))。

Style_strength_k : Composite(0..3) 线性映射到 (-1,-0.33,0.33,1)
E_ik             : 指数风格暴露 = Σ_n w_n * style_exposure_nk
style_exposure_n : 成分股近2年(24个)月度收益对10个风格因子月度收益的线性回归系数
Style_score_i    : Σ_k E_ik * Style_strength_k
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def map_strength(composite_row: pd.Series, strength_map: dict) -> pd.Series:
    """Composite(0,1,2,3) -> (-1,-0.33,0.33,1)。"""
    return composite_row.map(lambda v: strength_map.get(int(v)) if pd.notna(v) else np.nan)


def stock_style_exposures(monthly_stock_returns: pd.DataFrame,
                          monthly_factor_returns: pd.DataFrame,
                          asof: pd.Timestamp,
                          window: int = 24,
                          min_obs: int = 12) -> pd.DataFrame:
    """截至 asof 的近 window 个月, 对每只个股做多元线性回归求暴露。

    monthly_stock_returns: index=月末, columns=股票
    monthly_factor_returns: index=月末, columns=10因子
    返回: index=股票, columns=10因子 (观测不足 min_obs 的股票为 NaN)
    """
    F = monthly_factor_returns.loc[:asof].tail(window)
    R = monthly_stock_returns.loc[F.index]
    X = np.column_stack([np.ones(len(F)), F.values])     # 带截距
    cols = list(monthly_factor_returns.columns)
    betas = {}
    for code in R.columns:
        y = R[code].values.astype(float)
        mask = np.isfinite(y) & np.isfinite(F.values).all(axis=1)
        if mask.sum() < min_obs:
            betas[code] = [np.nan] * len(cols)
            continue
        b, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        betas[code] = list(b[1:])                         # 去掉截距
    return pd.DataFrame(betas, index=cols).T


def index_exposure(constituents: pd.DataFrame,
                   stock_exposures: pd.DataFrame) -> pd.Series:
    """E_ik = Σ w_n * exposure_nk。constituents: [code, weight](单个指数, 权重和归一)。"""
    df = constituents.set_index("code").join(stock_exposures, how="inner").dropna()
    if df.empty:
        return pd.Series(dtype=float)
    w = df["weight"] / df["weight"].sum()
    return df[stock_exposures.columns].mul(w, axis=0).sum()


def style_score(index_exposures: pd.DataFrame, strength: pd.Series) -> pd.Series:
    """index_exposures: index=指数, columns=10因子; strength: 10因子强度。"""
    common = index_exposures.columns.intersection(strength.index)
    return index_exposures[common].mul(strength[common], axis=1).sum(axis=1)
