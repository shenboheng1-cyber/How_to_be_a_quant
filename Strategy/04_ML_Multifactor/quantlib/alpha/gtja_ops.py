# -*- coding: utf-8 -*-
"""
quantlib.alpha.gtja_ops —— 国泰君安 191 因子的算子库 + 公式解释器
================================================================
实现国泰君安《短周期价量特征》报告附录"表15 函数定义"里的全部算子，
并提供一个小解释器：把 191 个公式按原样存为字符串，翻译 ^ 与三元表达式后 eval。
这样公式与报告几乎一字不差，转写错误最小。

所有算子作用在【日期×股票】宽矩阵上；ts_* / 滚动算子只用过去 → 无前视。
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from . import operators as op


# ---------- 基础算子（名称对齐报告表15）----------
def RANK(x):              return x.rank(axis=1, pct=True)
def DELAY(x, n):          return x.shift(int(n))
def DELTA(x, n):          return x - x.shift(int(n))
def SUM(x, n):            return x.rolling(int(n), min_periods=1).sum()
def MEAN(x, n):           return x.rolling(int(n), min_periods=1).mean()
def STD(x, n):            return x.rolling(int(n), min_periods=2).std()
def TSMIN(x, n):          return x.rolling(int(n), min_periods=1).min()
def TSMAX(x, n):          return x.rolling(int(n), min_periods=1).max()
def TSRANK(x, n):         return x.rolling(int(n), min_periods=2).rank(pct=True)
def PROD(x, n):           return x.rolling(int(n), min_periods=1).apply(np.prod, raw=True)
def ABS(x):               return x.abs() if hasattr(x, "abs") else np.abs(x)
def LOG(x):               return np.log(x)
def SIGN(x):              return np.sign(x)
def CORR(x, y, n):        return op.ts_corr(x, y, int(n))
def SQRT(x):              return np.sqrt(x)


def _as_df(a, like):
    """把标量/数组对齐成与 like 同形状的 DataFrame。"""
    if isinstance(a, pd.DataFrame):
        return a
    return pd.DataFrame(np.broadcast_to(a, like.shape), index=like.index, columns=like.columns)


def _is_window(b):
    """第二参数像窗口（正整数 ≥2）→ 滚动语义；0 或序列 → 逐元素。"""
    return isinstance(b, (int, float)) and float(b) == int(b) and b >= 2


def MAX(a, b):
    """重载：MAX(X,n) 滚动最大(TSMAX)；MAX(A,B)/MAX(0,X) 逐元素取大。"""
    if isinstance(a, pd.DataFrame) and _is_window(b):
        return TSMAX(a, int(b))
    like = a if isinstance(a, pd.DataFrame) else b
    A, B = _as_df(a, like), _as_df(b, like)
    return A.where(A >= B, B)


def MIN(a, b):
    """重载：MIN(X,n) 滚动最小(TSMIN)；MIN(A,B)/MIN(0,X) 逐元素取小。"""
    if isinstance(a, pd.DataFrame) and _is_window(b):
        return TSMIN(a, int(b))
    like = a if isinstance(a, pd.DataFrame) else b
    A, B = _as_df(a, like), _as_df(b, like)
    return A.where(A <= B, B)


def COVIANCE(x, y, n):
    n = int(n)
    mx, my = x.rolling(n, min_periods=2).mean(), y.rolling(n, min_periods=2).mean()
    return (x * y).rolling(n, min_periods=2).mean() - mx * my


def SMA(x, n, m=1):
    """递归均线 Y_t=(x_t·m + Y_{t-1}·(n-m))/n，等价 ewm(alpha=m/n)。"""
    return x.ewm(alpha=float(m) / float(n), adjust=False).mean()


def WMA(x, n):
    """权重 0.9^i（i=距当前的间隔）的加权移动平均。"""
    n = int(n)
    w = 0.9 ** np.arange(n)            # i=0 最近
    w = w[::-1] / w.sum()              # 对齐 rolling 窗口顺序（末尾=最近）
    return x.rolling(n, min_periods=1).apply(
        lambda a: np.dot(a, w[-len(a):]) / w[-len(a):].sum(), raw=True)


def DECAYLINEAR(x, d):    return op.decay_linear(x, int(d))
def COUNT(cond, n):       return cond.astype(float).rolling(int(n), min_periods=1).sum()
def SUMIF(x, n, cond):    return (x * cond.astype(float)).rolling(int(n), min_periods=1).sum()
def FILTER(x, cond):      return x.where(cond)
def HIGHDAY(x, n):        return op.ts_argmax(x, int(n))   # 最大值距今天数
def LOWDAY(x, n):         return op.ts_argmin(x, int(n))   # 最小值距今天数


def SUMAC(x, n):
    """前 n 项累加（窗口内累计和的当前值 = 窗口和）。"""
    return x.rolling(int(n), min_periods=1).sum()


def _rolling_ols(y, xv, n, want="beta"):
    """对每只股票滚动回归 y~xv（窗口 n），返回 beta 或当期残差。xv 可为 SEQUENCE。"""
    n = int(n)
    if isinstance(xv, _Sequence):
        xb = np.arange(1, n + 1, dtype=float)
        xmean = xb.mean(); xc = xb - xmean; sxx = (xc ** 2).sum()
        def f(a):
            a = np.asarray(a, float)
            if np.isnan(a).any(): return np.nan
            beta = (xc * (a - a.mean())).sum() / sxx
            if want == "beta": return beta
            return a[-1] - (a.mean() + beta * (xb[-1] - xmean))
        return y.rolling(n, min_periods=n).apply(f, raw=True)
    # y 与 xv 都是矩阵：逐元素滚动回归，较慢，仅 Alpha 少量使用
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    yv = y.values; xx = xv.values
    for j in range(y.shape[1]):
        col_y, col_x = yv[:, j], xx[:, j]
        for i in range(n - 1, y.shape[0]):
            ay, ax = col_y[i - n + 1:i + 1], col_x[i - n + 1:i + 1]
            if np.isnan(ay).any() or np.isnan(ax).any(): continue
            axc = ax - ax.mean(); sxx = (axc ** 2).sum()
            if sxx == 0: continue
            beta = (axc * (ay - ay.mean())).sum() / sxx
            out.iat[i, j] = beta if want == "beta" else ay[-1] - (ay.mean() + beta * (ax[-1] - ax.mean()))
    return out


def REGBETA(a, b, n):   return _rolling_ols(a, b, n, "beta")
def REGRESI(a, b, n):   return _rolling_ols(a, b, n, "resi")


class _Sequence:
    """SEQUENCE(n) 占位：在 REGBETA/REGRESI 里识别为 1..n 等差列。"""
    def __init__(self, n): self.n = int(n)
def SEQUENCE(n):        return _Sequence(n)


def IF(c, a, b):
    """三元 A?B:C 的逐元素实现。"""
    like = c if isinstance(c, pd.DataFrame) else (a if isinstance(a, pd.DataFrame) else b)
    res = np.where(np.asarray(c, dtype=bool) if not isinstance(c, pd.DataFrame) else c.values,
                   _as_df(a, like).values, _as_df(b, like).values)
    return pd.DataFrame(res, index=like.index, columns=like.columns)


# ---------- 公式翻译：^ → **，三元 A?B:C → IF(A,B,C) ----------
def _toplevel_ternary(s: str) -> str:
    """转换当前层（depth 0）的三元 A?B:C → IF(A,B,C)。a/b 递归处理嵌套三元。"""
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        elif ch == "?" and depth == 0:
            d = 0; q = 0
            for j in range(i + 1, len(s)):
                c = s[j]
                if c == "(": d += 1
                elif c == ")": d -= 1
                elif d == 0 and c == "?": q += 1
                elif d == 0 and c == ":":
                    if q == 0:
                        return (f"IF({_toplevel_ternary(s[:i])},"
                                f"{_toplevel_ternary(s[i+1:j])},"
                                f"{_toplevel_ternary(s[j+1:])})")
                    q -= 1
    return s


def _split_commas(s: str):
    """按 depth 0 的逗号切分（函数参数边界）。"""
    parts, depth, last = [], 0, 0
    for i, ch in enumerate(s):
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        elif ch == "," and depth == 0:
            parts.append(s[last:i]); last = i + 1
    parts.append(s[last:])
    return parts


def _ternary_to_if(s: str) -> str:
    """先递归进入每个括号组转换其内部三元，再按顶层逗号分段转换本层三元。
    这样嵌在函数参数里的三元（如 SUM((cond?a:b),6)）也能正确处理，
    且三元 else 分支不会越过逗号吞掉后一个参数。"""
    out, i, n = "", 0, len(s)
    while i < n:
        if s[i] == "(":
            depth, j = 1, i + 1
            while j < n and depth > 0:
                if s[j] == "(": depth += 1
                elif s[j] == ")": depth -= 1
                j += 1
            out += "(" + _ternary_to_if(s[i + 1:j - 1]) + ")"
            i = j
        else:
            out += s[i]; i += 1
    return ",".join(_toplevel_ternary(seg) for seg in _split_commas(out))


def translate(formula: str) -> str:
    f = formula.replace("^", "**").replace("||", "|")
    f = re.sub(r"(?<![<>=!])=(?!=)", "==", f)   # 条件里的单 = 转 ==（不动 <= >= ==）
    f = _ternary_to_if(f)
    return f


def namespace(M, bench=None) -> dict:
    """构造 eval 用命名空间：基础字段 + 衍生量 + 所有算子。"""
    HIGH, LOW, CLOSE, OPEN = M.high, M.low, M.close, M.open
    delay_open = OPEN.shift(1)
    ns = {
        "OPEN": OPEN, "HIGH": HIGH, "LOW": LOW, "CLOSE": CLOSE,
        "VWAP": M.vwap, "VOLUME": M.volume, "AMOUNT": M.amount,
        "RET": M.returns, "CAP": M.cap,
        "DTM": IF(OPEN <= delay_open, 0.0, MAX(HIGH - OPEN, OPEN - delay_open)),
        "DBM": IF(OPEN >= delay_open, 0.0, MAX(OPEN - LOW, OPEN - delay_open)),
        "TR":  MAX(MAX(HIGH - LOW, ABS(HIGH - CLOSE.shift(1))), ABS(LOW - CLOSE.shift(1))),
        "HD":  HIGH - HIGH.shift(1),
        "LD":  LOW.shift(1) - LOW,
    }
    if bench is not None:
        ns["BANCHMARKINDEXCLOSE"] = bench["close"]
        ns["BANCHMARKINDEXOPEN"] = bench["open"]
    # 注入所有算子
    for name in ("RANK DELAY DELTA SUM MEAN STD TSMIN TSMAX TSRANK PROD ABS LOG SIGN "
                 "CORR MAX MIN COVIANCE SMA WMA DECAYLINEAR COUNT SUMIF FILTER HIGHDAY "
                 "LOWDAY SUMAC REGBETA REGRESI SEQUENCE IF SQRT").split():
        ns[name] = globals()[name]
    ns["MA"] = MEAN      # 报告 Alpha78 用 MA = MEAN
    ns["SMEAN"] = SMA    # 报告 Alpha22 的 SMEAN（OCR）= SMA
    return ns
