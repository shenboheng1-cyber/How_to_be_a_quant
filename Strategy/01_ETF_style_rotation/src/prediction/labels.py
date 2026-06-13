"""三种风格标签构造 (报告 三.(二))。

输入 fr: 周度风格因子收益 DataFrame, index=周调仓日, columns=10个风格因子。
输出与输入同形状的 0/1 标签 (窗口不足处为 NaN)。
注意: 标签是"当期事实", 模型用 X_t 预测 y_{t+1}; 移位在 logistic.py 中处理。
"""
import pandas as pd


def cumulative_label(fr: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """该因子4周滚动累计收益 > 全部因子4周累计收益的截面中位数 ? 1 : 0"""
    cum = fr.rolling(window).sum()
    med = cum.median(axis=1)
    lab = cum.gt(med, axis=0).astype(float)
    return lab.where(cum.notna())


def single_week_label(fr: pd.DataFrame) -> pd.DataFrame:
    """本周收益 > 0 ? 1 : 0"""
    return (fr > 0).astype(float).where(fr.notna())


def trend_label(fr: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """本周收益 > 4周滚动均值 ? 1 : 0"""
    ma = fr.rolling(window).mean()
    return (fr > ma).astype(float).where(ma.notna() & fr.notna())


def build_all_labels(fr: pd.DataFrame, window: int = 4) -> dict[str, pd.DataFrame]:
    return {
        "cumulative": cumulative_label(fr, window),
        "single_week": single_week_label(fr),
        "trend": trend_label(fr, window),
    }
