# -*- coding: utf-8 -*-
"""
quantlib.alpha.alphanet —— AlphaNet 式端到端特征提取（华泰思路）
================================================================
把每只股票近 W 天的价量矩阵，经"特征提取层"压成向量，再喂神经网络。
提取层(确定性，非学习)：特征两两 ts_corr + 各自 ts_std/decay/mean —— 捕捉价量交互。
随后 MLP 学非线性组合预测未来收益。

这里用已有算子(operators)在 (日期×股票) 宽矩阵上做滚动提取，在调仓日采样，
等价于"每个(股,日)用过去W天算提取特征"。torch 建网见 research/15。
"""
from __future__ import annotations
from . import operators as op

W = 30


_FKEYS = ["open", "high", "low", "close", "vwap", "volume", "ret"]


def _field(M, a):
    return {"open": M.open, "high": M.high, "low": M.low, "close": M.close,
            "vwap": M.vwap, "volume": M.volume, "ret": M.returns}[a]


def extract_features(M=None) -> dict:
    """返回 {名称: func(M)->矩阵} 的 AlphaNet 提取特征（惰性，逐个算省内存）。
    单特征：波动/线性衰减/均值；特征对：滚动相关（价量交互核心）。"""
    reg = {}
    for a in _FKEYS:
        reg[f"std_{a}"] = (lambda M_, _a=a: op.ts_std(_field(M_, _a), W))
        reg[f"decay_{a}"] = (lambda M_, _a=a: op.decay_linear(_field(M_, _a), W))
        reg[f"mean_{a}"] = (lambda M_, _a=a: op.ts_mean(_field(M_, _a), W))
    for i in range(len(_FKEYS)):
        for j in range(i + 1, len(_FKEYS)):
            ki, kj = _FKEYS[i], _FKEYS[j]
            reg[f"corr_{ki}_{kj}"] = (lambda M_, _i=ki, _j=kj: op.ts_corr(_field(M_, _i), _field(M_, _j), W))
    return reg
