# -*- coding: utf-8 -*-
"""
quantlib.factors.behavioral —— 原创行为因子库（L2）
================================================================
围绕"锚定效应 / 52周高点"的一组行为金融因子。核心研究问题：

  美股的 52周高点异象（George & Hwang 2004）在【散户主导、动量失效】的 A 股
  是否成立？方向如何？它是不是只是动量/市值的伪装？

符号约定同 classic：因子值越大 ⇒ 预期收益越高（待检验，A股可能相反）。
"""
from __future__ import annotations
import pandas as pd


def w52_high(panel: pd.DataFrame) -> pd.Series:
    """距52周高点 = 当前复权价 / 过去252日最高复权价 ∈(0,1]。

    经济假设（锚定/反应不足）：投资者把52周高点当心理天花板，好消息推动股价
    接近高点时因锚定而不敢追高、反应不足 → 接近高点者后续继续涨。
    （美股成立；A股待检验——这是 L2 的头牌研究问题。）"""
    return panel["w52high"]


def range_position(panel: pd.DataFrame) -> pd.Series:
    """52周区间位置 = (price - 252日最低) / (252日最高 - 252日最低) ∈[0,1]。

    w52_high 的姊妹因子：同样刻画"价格在历史区间中的相对高度"，但同时考虑下沿。
    用于稳健性对照——两个口径结论是否一致。"""
    return panel["range_pos"]


REGISTRY = {
    "w52_high":       (w52_high,       "距52周高点"),
    "range_position": (range_position, "52周区间位置"),
}
