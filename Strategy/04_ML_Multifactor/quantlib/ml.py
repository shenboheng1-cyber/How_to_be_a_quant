# -*- coding: utf-8 -*-
"""
quantlib.ml —— 机器学习因子合成（L3）
================================================================
把一堆因子(特征)用 ML 非线性合成为一个 alpha 信号。核心是【防泄漏】：
purged & embargoed walk-forward 交叉验证——训练永远在测试之前，中间留缓冲带。

提供：
  make_label        横截面去均值的未来收益(相对收益标签)
  walk_forward_predict   滚动训练→样本外预测(可换模型)
  lgb_model / ridge_model  LightGBM 与 岭回归(numpy手写) 两个模型
  ic_weight_signal  IC加权 baseline
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_label(panel: pd.DataFrame, fwd_col: str = "fwd_ret") -> np.ndarray:
    """标签 = 横截面去均值的未来收益（剥离市场，预测相对强弱）。"""
    df = pd.DataFrame({"dt": panel["trddt"].values, "y": panel[fwd_col].values})
    return df.groupby("dt")["y"].transform(lambda s: s - s.mean()).values


# ---------- 模型 ----------
def ridge_model(reg: float = 10.0):
    """岭回归(闭式解, numpy)。NaN→0(特征已zscore，0=均值)。返回 fit→predict 闭包。"""
    def fit(X, y):
        Xf = np.nan_to_num(X, nan=0.0)
        n_feat = Xf.shape[1]
        A = Xf.T @ Xf + reg * np.eye(n_feat)
        b = Xf.T @ y
        beta = np.linalg.solve(A, b)
        return lambda Z: np.nan_to_num(Z, nan=0.0) @ beta
    return fit


def lgb_model(**params):
    """LightGBM 回归。原生支持 NaN。验证尾段早停防过拟合。"""
    import lightgbm as lgb
    p = dict(objective="regression", n_estimators=300, learning_rate=0.03,
             num_leaves=31, min_child_samples=200, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
             n_jobs=-1, verbose=-1)
    p.update(params)

    def fit(X, y):
        n = len(y); cut = int(n * 0.85)
        m = lgb.LGBMRegressor(**p)
        m.fit(X[:cut], y[:cut], eval_set=[(X[cut:], y[cut:])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        return lambda Z: m.predict(Z)
    fit._last_model = None
    return fit


# ---------- 防泄漏 walk-forward ----------
def walk_forward_predict(X, y, dates, model_fit, init: int = 36,
                         embargo: int = 1, step: int = 3, min_train: int = 5000,
                         log=print):
    """purged & embargoed 扩张窗 walk-forward，返回样本外预测(与行对齐，训练期为NaN)。

    init   : 起始训练期数(月)
    embargo: 训练集末尾与测试集之间留的缓冲期数
    step   : 每次重训后预测多少期(每3期重训一次，省算力)
    """
    uniq = sorted(pd.unique(dates))
    pred = np.full(len(y), np.nan)
    i = init
    while i < len(uniq):
        test_ps = uniq[i:i + step]
        train_ps = uniq[: max(0, i - embargo)]               # purge: 末尾embargo期不进训练
        tr = np.isin(dates, train_ps) & ~np.isnan(y)
        te = np.isin(dates, test_ps)
        if tr.sum() >= min_train and te.sum() > 0:
            predict = model_fit(X[tr], y[tr])
            pred[te] = predict(X[te])
        i += step
    return pred


# ---------- baseline ----------
def equal_weight_signal(X) -> np.ndarray:
    """等权合成：标准化特征的行均值(完全无拟合，最公平 baseline)。"""
    return np.nanmean(X, axis=1)


def ic_weight_signal(X, y, dates, init: int = 36) -> np.ndarray:
    """IC加权 baseline：用截至当期的历史平均IC做权重(扩张窗，无未来)。"""
    uniq = sorted(pd.unique(dates))
    pred = np.full(X.shape[0], np.nan)
    # 逐期特征IC
    per_ic = {}
    for p in uniq:
        m = (dates == p) & ~np.isnan(y)
        if m.sum() < 20: continue
        Xm, ym = X[m], y[m]
        ic = np.array([_safe_corr(Xm[:, j], ym) for j in range(Xm.shape[1])])
        per_ic[p] = ic
    cum, cnt = np.zeros(X.shape[1]), 0
    for i, p in enumerate(uniq):
        if i >= init and cnt > 0:
            w = np.nan_to_num(cum / cnt)
            te = dates == p
            pred[te] = np.nan_to_num(X[te]) @ w
        if p in per_ic:
            cum = cum + np.nan_to_num(per_ic[p]); cnt += 1
    return pred


def _safe_corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 20: return np.nan
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0: return np.nan
    return np.corrcoef(a, b)[0, 1]
