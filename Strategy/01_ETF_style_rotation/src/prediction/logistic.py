"""滚动窗口 Logistic 分类预测 (报告 三.(二))。

P(y_{t+1}=1) = sigma(b0 + b_M·X^M_t + b_AR·X^AR_t)
  X^M_t  : 当期5类宏观综合得分
  X^AR_t : 该因子近 ar_lags 周收益 (滞后项)
滚动 window=104 周训练, 预测下一周; >0.5 判为 1。
严格防未来函数: 训练集仅含 (X_s, y_{s+1}), s+1 <= t; 预测目标为 y_{t+1}。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def build_feature_matrix(macro_scores: pd.DataFrame,
                         factor_return: pd.Series,
                         ar_lags: int = 4) -> pd.DataFrame:
    """X_t = [5类宏观得分_t, r_t, r_{t-1}, ..., r_{t-ar_lags+1}], 对齐到周索引。"""
    X = macro_scores.copy()
    for lag in range(ar_lags):
        X[f"ar_{lag}"] = factor_return.shift(lag)
    return X


def rolling_logistic_predict(label: pd.Series,
                             X: pd.DataFrame,
                             window: int = 104,
                             threshold: float = 0.5) -> pd.DataFrame:
    """返回 DataFrame[pred, prob, actual], index = 被预测周 t+1。

    label: 当期事实标签 y_t (与 X 同索引)
    在每个 t, 用 {(X_s, y_{s+1}) : t-window <= s < t} 训练, 以 X_t 预测 y_{t+1}。
    """
    idx = X.index
    y_next = label.shift(-1)          # y_next[s] = y_{s+1}
    preds, probs, actuals, when = [], [], [], []

    for ti in range(window, len(idx) - 1):
        t = idx[ti]
        train_slice = slice(ti - window, ti)          # s in [t-window, t)
        Xtr = X.iloc[train_slice]
        ytr = y_next.iloc[train_slice]
        mask = Xtr.notna().all(axis=1) & ytr.notna()
        Xtr, ytr = Xtr[mask], ytr[mask]
        if X.iloc[ti].isna().any() or len(ytr) < window // 2:
            continue
        if ytr.nunique() < 2:                          # 单一类别窗口: 直接外推
            p = float(ytr.iloc[0])
        else:
            model = LogisticRegression(max_iter=2000)
            model.fit(Xtr.values, ytr.values.astype(int))
            p = float(model.predict_proba(X.iloc[[ti]].values)[0, 1])
        preds.append(1.0 if p >= threshold else 0.0)
        probs.append(p)
        actuals.append(label.iloc[ti + 1] if ti + 1 < len(idx) else np.nan)
        when.append(idx[ti + 1])

    return pd.DataFrame({"pred": preds, "prob": probs, "actual": actuals},
                        index=pd.DatetimeIndex(when) if when else None)


def predict_all(factor_returns: pd.DataFrame,
                macro_scores: pd.DataFrame,
                labels: dict[str, pd.DataFrame],
                window: int = 104,
                ar_lags: int = 4) -> dict:
    """对 10因子 × 3标签 逐一滚动预测。

    返回 {label_name: {factor: DataFrame[pred, prob, actual]}}。
    Composite = 三个标签 pred 之和 ∈ {0,1,2,3}, 由 composite_score() 汇总。
    """
    common = factor_returns.index.intersection(macro_scores.index)
    fr = factor_returns.loc[common]
    ms = macro_scores.loc[common]
    out = {}
    for lname, ldf in labels.items():
        out[lname] = {}
        for fac in fr.columns:
            X = build_feature_matrix(ms, fr[fac], ar_lags)
            out[lname][fac] = rolling_logistic_predict(ldf.loc[common, fac], X, window)
    return out


def composite_score(pred_results: dict) -> pd.DataFrame:
    """Composite = Cumulative + SingleWeek + Trend, index=周, columns=因子。"""
    parts = []
    for lname, by_factor in pred_results.items():
        df = pd.DataFrame({fac: r["pred"] for fac, r in by_factor.items() if r is not None and len(r)})
        parts.append(df)
    total = parts[0]
    for p in parts[1:]:
        total = total.add(p, fill_value=np.nan)
    return total


def accuracy_table(pred_results: dict) -> pd.DataFrame:
    """样本外准确率 (对账 表6: Trend ≈ 70%)。行=因子, 列=标签。"""
    rows = {}
    for lname, by_factor in pred_results.items():
        for fac, r in by_factor.items():
            if r is None or not len(r):
                continue
            ok = r.dropna(subset=["actual"])
            rows.setdefault(fac, {})[lname] = (ok["pred"] == ok["actual"]).mean()
    return pd.DataFrame(rows).T
