# -*- coding: utf-8 -*-
"""
quantlib.evaluate —— 因子有效性检验
================================================================
两套互补的"判官"，回答"这个因子到底有没有预测力"：

A) IC / RankIC 分析（信息系数）
   每个调仓日算"因子值"与"未来收益"的横截面相关，得到一条 IC 时间序列。
   - IC      = Pearson 相关
   - RankIC  = Spearman 秩相关（抗异常值，业界更常用）
   汇总指标：
   - IC 均值      预测方向与强度
   - ICIR=均值/标准差   稳定性（最重要，类似因子的"信息比率"）
   - t 值         统计显著性（≈ ICIR·√期数）
   - IC 胜率      IC>0 的比例
   经验阈值（月频 A股）：|RankIC 均值|>0.03 且 ICIR>0.3 就算不错的因子。

B) 分层回测（quantile portfolios）
   每期把股票按因子值排序分 N 组，看各组未来收益是否【单调】，
   并构造"多空组合"(最高组−最低组)，看年化收益、夏普、回撤。
   单调性 = 因子有效最直观的证据。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

_PPY = {"M": 12, "W": 52, "Q": 4, "D": 252}   # 每年期数，用于年化


# ============ A) IC 分析 ============
def compute_ic(panel: pd.DataFrame, factor: pd.Series,
               fwd_col: str = "fwd_ret", method: str = "spearman") -> pd.Series:
    """逐调仓日计算 IC，返回以 trddt 为索引的 IC 序列。method='spearman'即RankIC。"""
    df = pd.DataFrame({"dt": panel["trddt"].values,
                       "f": np.asarray(factor),
                       "r": panel[fwd_col].values}).dropna()

    def _corr(x):
        a, b = x["f"], x["r"]
        if method == "spearman":      # 秩相关=对排名做普通(Pearson)相关，自己排名以免依赖 scipy
            a, b = a.rank(), b.rank()
        return a.corr(b)

    # 在 apply 前先用 [["f","r"]] 选好列，分组列 dt 不进子表：
    # 既无 "operated on grouping columns" 警告，也兼容各版本 pandas（不依赖 include_groups）。
    ic = df.groupby("dt")[["f", "r"]].apply(_corr)
    ic.name = "IC"
    return ic


def ic_summary(ic: pd.Series, freq: str = "M") -> dict:
    """把 IC 序列汇总成一组标量指标。"""
    ic = ic.dropna()
    n = len(ic)
    mean, std = ic.mean(), ic.std(ddof=1)
    icir = mean / std if std else np.nan
    return {
        "IC均值": round(mean, 4),
        "IC标准差": round(std, 4),
        "ICIR": round(icir, 3),
        "t值": round(icir * np.sqrt(n), 2) if std else np.nan,
        "IC>0占比": round((ic > 0).mean(), 3),
        "|IC|>0.02占比": round((ic.abs() > 0.02).mean(), 3),
        "期数": n,
    }


# ============ B) 分层回测 ============
def quantile_returns(panel: pd.DataFrame, factor: pd.Series, n_groups: int = 10,
                     fwd_col: str = "fwd_ret") -> pd.DataFrame:
    """每期按因子值分 N 组，返回各组等权未来收益。

    返回 DataFrame：index=trddt，columns=Q1..QN（Q1=因子最小组，QN=最大组）。
    """
    df = pd.DataFrame({"dt": panel["trddt"].values,
                       "f": np.asarray(factor),
                       "r": panel[fwd_col].values}).dropna()

    # 每个横截面内按因子秩分组（用秩而非qcut，避免重复值导致分箱失败）
    def _bucket(s):
        r = s.rank(method="first")
        return np.ceil(r / len(s) * n_groups).clip(1, n_groups).astype(int)
    df["q"] = df.groupby("dt")["f"].transform(_bucket)

    grp = df.groupby(["dt", "q"])["r"].mean().unstack("q")
    grp.columns = [f"Q{int(c)}" for c in grp.columns]
    return grp


def quantile_summary(qret: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """各组年化收益 + 多空组合(QN−Q1)的年化/夏普/最大回撤。"""
    ppy = _PPY[freq]
    rows = {}
    for col in qret.columns:
        r = qret[col].dropna()
        rows[col] = {"年化收益": (1 + r).prod() ** (ppy / len(r)) - 1,
                     "年化波动": r.std(ddof=1) * np.sqrt(ppy)}
    summ = pd.DataFrame(rows).T

    # 多空组合 = 最高组 − 最低组
    hi, lo = qret.columns[-1], qret.columns[0]
    ls = (qret[hi] - qret[lo]).dropna()
    nav = (1 + ls).cumprod()
    mdd = (nav / nav.cummax() - 1).min()
    ann = (1 + ls).prod() ** (ppy / len(ls)) - 1
    sharpe = ls.mean() / ls.std(ddof=1) * np.sqrt(ppy) if ls.std(ddof=1) else np.nan
    summ.loc["多空(QN-Q1)"] = {"年化收益": ann, "年化波动": ls.std(ddof=1) * np.sqrt(ppy)}
    summ["夏普"] = np.nan
    summ.loc["多空(QN-Q1)", "夏普"] = round(sharpe, 2)
    summ.loc["多空(QN-Q1)", "最大回撤"] = round(mdd, 3)
    summ["年化收益"] = summ["年化收益"].round(4)
    summ["年化波动"] = summ["年化波动"].round(4)
    return summ


def long_short_nav(qret: pd.DataFrame) -> pd.Series:
    """多空组合净值曲线（画图用）。"""
    hi, lo = qret.columns[-1], qret.columns[0]
    ls = (qret[hi] - qret[lo]).dropna()
    return (1 + ls).cumprod()


def evaluate_factor(panel: pd.DataFrame, factor: pd.Series, n_groups: int = 10,
                    freq: str = "M") -> dict:
    """一站式：返回 IC 汇总、分层汇总、IC序列、分层收益。"""
    ic = compute_ic(panel, factor)
    qret = quantile_returns(panel, factor, n_groups)
    return {
        "ic_summary": ic_summary(ic, freq),
        "quantile_summary": quantile_summary(qret, freq),
        "ic_series": ic,
        "quantile_returns": qret,
    }


def fama_macbeth(panel: pd.DataFrame, factors: dict,
                 fwd_col: str = "fwd_ret") -> pd.DataFrame:
    """Fama-MacBeth 回归：逐期把未来收益对多个因子做横截面回归，
    汇总各因子系数的时序均值与 t 值。

    回答"控制住其它因子后，某因子是否仍有独立的预测力"——多因子有效性的金标准。
    factors: {名称: Series}（建议各因子已 zscore，系数可比）。
    返回：index=因子名(+const)，列=[coef均值, t值, 显著占比]。
    """
    names = list(factors.keys())
    cols = {n: np.asarray(s) for n, s in factors.items()}
    cols["_y"] = panel[fwd_col].values
    df = pd.DataFrame(cols)
    df["trddt"] = panel["trddt"].values

    coefs = []
    for _, g in df.groupby("trddt"):
        sub = g[names + ["_y"]].dropna()
        if len(sub) < 20:
            continue
        X = np.column_stack([np.ones(len(sub)), sub[names].values])
        beta, *_ = np.linalg.lstsq(X, sub["_y"].values, rcond=None)
        coefs.append(beta)
    coefs = np.array(coefs)                      # shape: [期数, 1+因子数]
    labels = ["const"] + names
    mean = coefs.mean(axis=0)
    se = coefs.std(axis=0, ddof=1) / np.sqrt(len(coefs))
    t = mean / se
    return pd.DataFrame({
        "coef均值": mean.round(5),
        "t值": t.round(2),                       # |t|>2 即该因子有显著独立预测力
        "系数为正占比": (coefs > 0).mean(axis=0).round(3),
        "期数": len(coefs),
    }, index=labels)
