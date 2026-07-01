# -*- coding: utf-8 -*-
"""
quantlib.optimizer —— 组合优化器（L5，cvxpy QP）
================================================================
max_w  alpha·w − λ·风险 − γ·换手
s.t.   Σw=1, 0≤w≤cap(多头), 行业中性(各行业权重=候选universe占比),
       风格中性(组合风格暴露=universe均值)。
风险用因子结构 (Xᵀw)ᵀF(Xᵀw)+Σd_i w_i²，避免 N×N。
需 cvxpy(base 环境有)。求解失败回退等权。
"""
from __future__ import annotations
import numpy as np


def optimize(alpha, X_ind, X_style, F, d, w_prev=None,
             cap=0.03, lam=8.0, gamma=0.0, neutral=True):
    """返回最优权重 w (n,)。X_ind:(n,n_ind) 行业哑变量, X_style:(n,n_sty),
    F:(n_ind+n_sty 方阵) 因子协方差, d:(n,) 特质方差。"""
    import cvxpy as cp
    n = len(alpha)
    X = np.hstack([X_ind, X_style])
    w = cp.Variable(n)
    fexp = X.T @ w
    risk = cp.quad_form(fexp, cp.psd_wrap(F)) + cp.sum(cp.multiply(np.maximum(d, 1e-8), cp.square(w)))
    obj = alpha @ w - lam * risk
    if gamma > 0 and w_prev is not None:
        obj = obj - gamma * cp.norm1(w - w_prev)
    cons = [cp.sum(w) == 1, w >= 0, w <= cap]
    if neutral:
        cons.append(X_ind.T @ w == X_ind.mean(axis=0))      # 行业中性(=候选等权占比,恒可行)
        cons.append(X_style.T @ w == X_style.mean(axis=0))  # 风格中性
    try:
        cp.Problem(cp.Maximize(obj), cons).solve(solver=cp.OSQP, max_iter=20000, verbose=False)
        if w.value is None or np.isnan(w.value).any():
            raise ValueError("no solution")
        return np.clip(w.value, 0, cap)
    except Exception:
        return np.full(n, 1.0 / n)                          # 回退等权


def optimize_enhanced(alpha, b, X_ind, X_style, F, d,
                      active_cap=0.02, te=0.05, style_band=0.10,
                      gamma=0.0, w_prev=None):
    """指数增强:基准相对优化。max alpha·w − gamma·‖w−w_prev‖₁(换手惩罚)
    s.t. Σw=1, 0≤w≤b+active_cap(可行,w=b即解), 行业中性 X_ind·(w-b)=0,
         风格暴露受控 |X_style·(w-b)|≤band, 跟踪误差 (w-b)ᵀΣ(w-b)≤te²。
    b=基准权重(和为1)。gamma>0 时罚跨期权重变动(降换手),w_prev须与本期股票池对齐。"""
    import cvxpy as cp
    n = len(alpha)
    X = np.hstack([X_ind, X_style])
    w = cp.Variable(n)
    a = w - b                                                # 主动权重
    afe = X.T @ a
    te2 = cp.quad_form(afe, cp.psd_wrap(F)) + cp.sum(cp.multiply(np.maximum(d, 1e-8), cp.square(a)))
    cons = [cp.sum(w) == 1, w >= 0, w <= b + active_cap,
            X_ind.T @ a == 0,
            cp.abs(X_style.T @ a) <= style_band,
            te2 <= te ** 2]
    obj = alpha @ w
    if gamma > 0 and w_prev is not None:
        obj = obj - gamma * cp.norm1(w - w_prev)             # 换手惩罚
    prob = cp.Problem(cp.Maximize(obj), cons)
    for solver in (cp.CLARABEL, cp.SCS, cp.ECOS):           # TE是二次约束(SOCP)→需锥求解器,非OSQP
        try:
            prob.solve(solver=solver, verbose=False)
            if w.value is not None and not np.isnan(w.value).any():
                return np.clip(w.value, 0, None)
        except Exception:
            continue
    return b                                                # 全失败:回退持基准
