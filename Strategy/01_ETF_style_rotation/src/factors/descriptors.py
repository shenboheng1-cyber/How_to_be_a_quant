"""23个描述变量计算。

⚠️ 本模块依赖 Choice 字段确认 (docs/DATA_FIELDS.md), 当前为带签名的骨架。
每个函数输入统一为已缓存的 panel 数据, 输出 index=股票 的截面 Series (asof 日)。
Phase1 优先实现行情类描述变量 (LNCAP/MIDCAP/波动率族/流动性族/动量族),
财务类 (DTOA/VSAL/.../DTOP) 在财务字段确认后补齐。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def lncap(float_mv: pd.Series) -> pd.Series:
    """LNCAP = ln(流通市值)。"""
    return np.log(float_mv.where(float_mv > 0))


def midcap(lncap_s: pd.Series) -> pd.Series:
    """MIDCAP: LNCAP^3 对 LNCAP 回归取残差(等权), 再去极值标准化由上层完成。"""
    x = lncap_s.dropna()
    if len(x) < 10:
        return lncap_s * np.nan
    X = np.column_stack([np.ones(len(x)), x.values])
    y = x.values ** 3
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    return pd.Series(resid, index=x.index).reindex(lncap_s.index)


def _ewm_weights(n: int, halflife: int) -> np.ndarray:
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n - 1, -1, -1)
    return w / w.sum()


def hbeta_halpha_hsigma(stock_ret: pd.DataFrame, mkt_ret: pd.Series,
                        window: int = 252, halflife: int = 63) -> pd.DataFrame:
    """加权回归 r_i = alpha + beta * r_m, 返回 [HBETA, HALPHA, HSIGMA] 截面。
    TODO: 大规模向量化实现; 当前为参考实现。"""
    R = stock_ret.tail(window)
    m = mkt_ret.reindex(R.index)
    w = _ewm_weights(len(R), halflife)
    out = {}
    X = np.column_stack([np.ones(len(R)), m.values])
    for code in R.columns:
        y = R[code].values
        mask = np.isfinite(y) & np.isfinite(m.values)
        if mask.sum() < window // 2:
            out[code] = (np.nan, np.nan, np.nan)
            continue
        Wm = w[mask] / w[mask].sum()
        Xw = X[mask] * np.sqrt(Wm)[:, None]
        yw = y[mask] * np.sqrt(Wm)
        b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = y[mask] - X[mask] @ b
        out[code] = (b[1], b[0], float(np.sqrt(np.average(resid ** 2, weights=Wm))))
    return pd.DataFrame(out, index=["HBETA", "HALPHA", "HSIGMA"]).T


def dastd(stock_ret: pd.DataFrame, window: int = 252, halflife: int = 42) -> pd.Series:
    R = stock_ret.tail(window)
    w = _ewm_weights(len(R), halflife)
    mu = R.mul(w, axis=0).sum()
    var = ((R - mu) ** 2).mul(w, axis=0).sum()
    return np.sqrt(var)


def cmra(stock_ret: pd.DataFrame, months: int = 12) -> pd.Series:
    """累积对数收益的 max-min 范围 (月度累积)。"""
    log_r = np.log1p(stock_ret)
    monthly = log_r.groupby(log_r.index.to_period("M")).sum().tail(months)
    cum = monthly.cumsum()
    return cum.max() - cum.min()


def stom_stoq_stoa(turnover: pd.DataFrame) -> pd.DataFrame:
    """STOM=ln(Σ21日换手), STOQ/STOA 为近3/12个月 STOM 的均值口径。"""
    t21 = turnover.tail(21).sum()
    t63 = turnover.tail(63).sum() / 3.0
    t252 = turnover.tail(252).sum() / 12.0
    return pd.DataFrame({"STOM": np.log(t21.where(t21 > 0)),
                         "STOQ": np.log(t63.where(t63 > 0)),
                         "STOA": np.log(t252.where(t252 > 0))})


def atvr(turnover: pd.DataFrame, window: int = 252, halflife: int = 63) -> pd.Series:
    T = turnover.tail(window)
    w = _ewm_weights(len(T), halflife)
    return T.mul(w, axis=0).sum() * 252


def strev(stock_ret: pd.DataFrame, window: int = 21, halflife: int = 10) -> pd.Series:
    R = np.log1p(stock_ret.tail(window))
    w = _ewm_weights(len(R), halflife)
    return R.mul(w, axis=0).sum()


# ---- 财务类描述变量: 待 Choice 字段确认后实现 ----
def financial_descriptors(financials: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """DTOA/VSAL/VERN/ATO/ROA/GPM/BTOP/ETOP/EGRO/SGRO/DTOP。
    TODO(Phase1.5): 依赖 docs/DATA_FIELDS.md 中财务字段确认。
    注意财报发布滞后: 使用 asof 当日已公告的最新报告期数据。"""
    raise NotImplementedError("待 Choice 财务字段确认后实现")
