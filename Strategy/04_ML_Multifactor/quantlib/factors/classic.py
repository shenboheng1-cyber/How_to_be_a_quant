# -*- coding: utf-8 -*-
"""
quantlib.factors.classic —— 经典异象因子库（L1 baseline）
================================================================
每个因子是纯函数 f(panel)->pd.Series（与 panel 行对齐），返回【原始】因子值。
去极值/中性化/标准化交给 preprocess，本文件只负责"因子的定义与方向"。

【铁律：符号约定】所有因子都构造成"值越大 ⇒ 预期收益越高"。
于是 RankIC 为正 = 因子方向与经济假设一致；为负 = 反着来。
A股是检验场：很多美股异象（如动量）在A股是反的，这恰恰是有意思的研究点。

每个因子都附【经济假设】——面试时你要能讲出来，而不是"我试了一堆指标"。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def size(panel: pd.DataFrame) -> pd.Series:
    """规模因子。假设：小市值溢价（小盘长期跑赢）。方向：-log(总市值)。
    A股小市值效应历史上极强，是最经典的本土异象之一。"""
    return -np.log(panel["total_mktcap"])


def ep(panel: pd.DataFrame) -> pd.Series:
    """盈利收益率 EP=1/PE。假设：价值溢价，便宜（高EP）的股票跑赢。只取正PE。"""
    pe = panel["pe_ttm"].where(panel["pe_ttm"] > 0)
    return 1.0 / pe


def bp(panel: pd.DataFrame) -> pd.Series:
    """账面市值比 BP=1/PB。假设：价值溢价（Fama-French HML 的核心）。"""
    pb = panel["pb"].where(panel["pb"] > 0)
    return 1.0 / pb


def sp(panel: pd.DataFrame) -> pd.Series:
    """销售收益率 SP=1/PS。假设：价值溢价，营收便宜者跑赢。"""
    ps = panel["ps_ttm"].where(panel["ps_ttm"] > 0)
    return 1.0 / ps


def momentum(panel: pd.DataFrame) -> pd.Series:
    """12-1 月动量。假设（美股）：过去赢家继续赢。
    注意：A股动量常常失效甚至反向，是本项目要诚实检验的点。"""
    return panel["mom_12_1"]


def reversal(panel: pd.DataFrame) -> pd.Series:
    """1 月短期反转。假设：上月涨多的下月回落。方向取负。A股反转极强。"""
    return -panel["rev_1m"]


def low_vol(panel: pd.DataFrame) -> pd.Series:
    """低波动异象。假设：低波动股票风险调整后跑赢（Frazzini-Pedersen BAB）。方向取负。"""
    return -panel["vol_60"]


def illiquidity(panel: pd.DataFrame) -> pd.Series:
    """Amihud 非流动性。假设：流动性溢价，越不流动要求越高回报。值越大越不流动。"""
    return panel["amihud"]


def low_turnover(panel: pd.DataFrame) -> pd.Series:
    """低换手。假设：高换手=高投机/情绪，后续跑输（A股散户特征明显）。方向取负。"""
    return -panel["turn_1m"]


def max_ret(panel: pd.DataFrame) -> pd.Series:
    """彩票因子（Bali 2011 MAX）。假设：近月有过暴涨的'彩票股'被高估，后续跑输。方向取负。"""
    return -panel["max_ret"]


# 因子注册表：引擎据此批量跑。key=因子名，value=(函数, 中文名)
REGISTRY = {
    "size":         (size,         "市值"),
    "ep":           (ep,           "盈利收益率EP"),
    "bp":           (bp,           "账面市值比BP"),
    "sp":           (sp,           "销售收益率SP"),
    "momentum":     (momentum,     "12-1月动量"),
    "reversal":     (reversal,     "1月反转"),
    "low_vol":      (low_vol,      "低波动"),
    "illiquidity":  (illiquidity,  "Amihud非流动性"),
    "low_turnover": (low_turnover, "低换手"),
    "max_ret":      (max_ret,      "彩票MAX"),
}
