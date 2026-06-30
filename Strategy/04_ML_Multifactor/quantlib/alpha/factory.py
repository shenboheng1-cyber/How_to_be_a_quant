# -*- coding: utf-8 -*-
"""
quantlib.alpha.factory —— 批量评估 + 多重检验校正
================================================================
把因子工厂生成的上百个 alpha 全部跑同一条检验流水线，产出排序大表，
并做【多重检验校正】——这是把"造一百个"从数据挖掘变成严谨研究的关键。

核心诚实点：测 100 个因子，即使全是噪声，按 |t|>2 也会有约 5 个"显著"。
所以要问：扣掉多重检验后，真正扛得住的有几个？（Bonferroni / BH-FDR）
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .. import preprocess, evaluate


def sample_to_panel(matrix: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """把日频 alpha 宽矩阵在调仓日采样，对齐到研究面板的行。"""
    reb = pd.to_datetime(pd.Index(panel["trddt"].unique()))
    sub = matrix.reindex(index=reb)
    long = sub.stack(dropna=False).rename("v").reset_index()
    long.columns = ["trddt", "stkcd", "v"]
    key = panel[["trddt", "stkcd"]].merge(long, on=["trddt", "stkcd"], how="left")
    return pd.Series(key["v"].values, index=panel.index)


def evaluate_alphas(panel: pd.DataFrame, registry: dict, M, freq: str = "M",
                    do_neutralize: bool = True, verbose: bool = True) -> pd.DataFrame:
    """对 registry 里每个 alpha 跑 预处理→IC→多空，汇成排序表（按|ICIR|）。

    M : 宽矩阵对象（quantlib.alpha.matrices.load_matrices 的产物）。
    """
    rows = []
    for i, (name, fn) in enumerate(registry.items(), 1):
        try:
            raw = sample_to_panel(fn(M), panel)
            if raw.notna().sum() < 5000:
                raise ValueError("有效值过少")
            f = preprocess.preprocess_factor(panel, raw, do_neutralize=do_neutralize)
            ic = evaluate.compute_ic(panel, f)
            s = evaluate.ic_summary(ic, freq)
            qs = evaluate.quantile_summary(evaluate.quantile_returns(panel, f, 10), freq)
            ls = qs.loc["多空(QN-Q1)"]
            rows.append({"因子": name, "RankIC": s["IC均值"], "ICIR": s["ICIR"],
                         "t值": s["t值"], "IC>0占比": s["IC>0占比"],
                         "多空年化": round(ls["年化收益"], 4), "多空夏普": ls["夏普"]})
        except Exception as e:
            rows.append({"因子": name, "RankIC": np.nan, "ICIR": np.nan,
                         "t值": np.nan, "IC>0占比": np.nan, "多空年化": np.nan,
                         "多空夏普": np.nan})
        if verbose and i % 20 == 0:
            print(f"  已评估 {i}/{len(registry)} ...")
    tbl = pd.DataFrame(rows)
    return tbl.reindex(tbl["ICIR"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def factor_correlation(panel: pd.DataFrame, registry: dict, M, names: list,
                       do_neutralize: bool = True) -> pd.DataFrame:
    """计算给定若干因子两两的平均横截面相关，用于揭示"幸存者是否高度冗余"。"""
    facs = {}
    for n in names:
        raw = sample_to_panel(registry[n](M), panel)
        facs[n] = preprocess.preprocess_factor(panel, raw, do_neutralize=do_neutralize).values
    df = pd.DataFrame(facs)
    df["dt"] = panel["trddt"].values
    df = df.dropna()
    corr = df.groupby("dt")[names].corr().groupby(level=1).mean().loc[names, names]
    return corr


def _p_from_t(t: float) -> float:
    """由 t 值近似双尾 p 值（正态近似，用标准库 erf，免 scipy）。"""
    if not np.isfinite(t):
        return np.nan
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def multiple_testing_summary(tvals, alpha: float = 0.05) -> dict:
    """多重检验校正：报告原始显著、期望假阳、Bonferroni、BH-FDR 通过数。"""
    t = np.array([x for x in tvals if np.isfinite(x)])
    n = len(t)
    p = np.array([_p_from_t(x) for x in t])
    # Benjamini-Hochberg
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * np.arange(1, n + 1) / n
    passed = ranked <= thresh
    n_fdr = int(np.where(passed)[0].max() + 1) if passed.any() else 0
    return {
        "因子总数": n,
        "原始显著(|t|>1.96)": int((p < alpha).sum()),
        "纯噪声期望假阳": round(n * alpha, 1),
        "Bonferroni通过": int((p < alpha / n).sum()),
        "BH-FDR通过": n_fdr,
    }
