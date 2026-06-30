# -*- coding: utf-8 -*-
"""
quantlib.alpha.alphas —— 因子工厂：批量生成 100+ 因子
================================================================
每个 alpha 是 func(M) -> 宽矩阵（日期×股票）的因子值。两个来源：

  A) CURATED：手写的公式化 alpha（借鉴公开的 WorldQuant「101 Formulaic Alphas」，
     已剔除需要行业分类的款；并加入若干自研表达式）。
  B) 组合生成器 generate()：基础信号 × 一元算子 × 窗口，系统化批量产出。

合并得 build_registry(M) -> {名称: func}。这是"系统化生成的因子库"，不是"挑出的赢家"，
配合多重检验校正使用 —— 详见 research/03。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import operators as op

EPS = 1e-9


# ============ A) 手写公式化 alpha ============
def a_co_strength(M):   return (M.close - M.open) / (M.high - M.low + 1e-3)      # 日内强弱
def a_hl_vwap(M):       return (M.high * M.low) ** 0.5 - M.vwap                  # 中枢偏离
def a_open_vol(M):      return -op.ts_corr(M.open, M.volume, 10)                 # 开盘-量背离
def a_rank_low(M):      return -op.ts_rank(op.rank(M.low), 9)
def a_vol_price(M):     return np.sign(op.delta(M.volume, 1)) * (-op.delta(M.close, 1))
def a_pv_corr(M):       return -op.ts_corr(op.rank(M.close), op.rank(M.volume), 6)
def a_ret_rev(M):       return -op.delta(M.close, 5) / (op.delay(M.close, 5) + EPS)
def a_low_close(M):     return -((M.low - M.close) * (M.open ** 3)) / \
                               ((M.low - M.high) * (M.close ** 3) + EPS)
def a_high_decay(M):    return -op.decay_linear(op.delta(M.high, 2), 5)
def a_amihud(M):        return op.ts_mean(M.returns.abs() / (M.amount + EPS), 20)
def a_vwap_mom(M):      return op.delta(M.vwap, 5) / (op.delay(M.vwap, 5) + EPS)
def a_vol_std(M):       return -op.ts_std(M.returns, 20)
def a_turn_rank(M):     return -op.ts_rank(M.volume / (M.adv20 + EPS), 10)
def a_max_pos(M):       return op.ts_argmax(M.high, 20) / 20.0
def a_corr_vwap_vol(M): return -op.ts_corr(M.vwap, M.volume, 12)
def a_close_acc(M):     return op.delta(op.delta(M.close, 1), 1) / (M.close + EPS)

# ===== 拓展：每个因子附经济假设。挑了若干 A 股相对少被挖的角度 =====
L2 = np.log(2)

# 隔夜 vs 日内收益分解（Lou-Polk-Skouras 2019：隔夜与日内由不同投资者驱动）
def a_overnight_rev(M):  on = M.open / op.delay(M.close, 1) - 1; return -op.ts_mean(on, 5)        # 隔夜跳空反转
def a_intraday_mom(M):   idr = M.close / M.open - 1; return op.ts_mean(idr, 5)                    # 日内动量延续
def a_overnight_mom(M):  on = M.open / op.delay(M.close, 1) - 1; return op.ts_sum(on, 60)         # 隔夜动量(机构信息)
def a_on_id_diverge(M):  on = M.open/op.delay(M.close,1)-1; idr = M.close/M.open-1; return -op.ts_corr(on, idr, 20)
def a_gap_fill(M):       gap = M.open/op.delay(M.close,1)-1; return -gap * (M.close/M.open - 1)   # 跳空当日回补

# 区间/日内波动（用 OHLC 估计，比收盘价波动信息更全）
def a_parkinson(M):      hl = np.log(M.high/M.low); return -op.ts_mean(hl*hl, 20)                 # Parkinson 低波
def a_garman_klass(M):   hl=np.log(M.high/M.low); co=np.log(M.close/M.open); return -op.ts_mean(0.5*hl*hl-(2*L2-1)*co*co, 20)
def a_vol_of_vol(M):     rng = (M.high-M.low)/(M.close+EPS); return -op.ts_std(rng, 20)           # 波动的波动
def a_range_expand(M):   rng = M.high-M.low; return -(rng/(op.ts_mean(rng,20)+EPS))              # 区间扩张后反转
def a_range_trend(M):    rng = (M.high-M.low)/(M.close+EPS); return op.delta(op.ts_mean(rng,5), 5)

# 收益分布形态（偏度/峰度/上下行不对称——博彩与崩盘偏好）
def a_ret_skew(M):       return -op.ts_skew(M.returns, 20)                                        # 高正偏被高估
def a_ret_kurt(M):       return -op.ts_kurt(M.returns, 20)                                        # 厚尾风险
def a_downside_vol(M):   neg = M.returns.where(M.returns < 0, 0.0); return -op.ts_std(neg, 20)    # 下行波动
def a_vol_asym(M):       pos=M.returns.where(M.returns>0,0.0); neg=M.returns.where(M.returns<0,0.0); return op.ts_std(pos,20)-op.ts_std(neg,20)
def a_max_range(M):      return -(op.ts_max(M.returns, 20) - op.ts_min(M.returns, 20))            # 收益极差(彩票)

# 路径效率/趋势质量（同样涨幅，平滑趋势 vs 锯齿，含义不同）
def a_efficiency(M):     net=(M.close-op.delay(M.close,20)).abs(); path=op.ts_sum(op.delta(M.close,1).abs(),20); return net/(path+EPS)
def a_variance_ratio(M): r1=op.ts_std(M.returns,20); r5=op.ts_std(M.close/op.delay(M.close,5)-1,20); return -(r5/(np.sqrt(5)*r1+EPS))
def a_up_day_ratio(M):   up=(M.returns>0).astype(float); return op.ts_mean(up,20)-0.5            # 上涨日占比
def a_price_curvature(M):return op.delta(op.delta(op.ts_mean(M.close,5),5),5)/(M.close+EPS)

# 量价动态/弹性（成交结构的变化比水平更有信息）
def a_turnover_accel(M): t=M.volume/(M.adv20+EPS); return op.delta(op.ts_mean(t,5),5)            # 换手加速
def a_volume_shock(M):   return -(M.volume-op.ts_mean(M.volume,20))/(op.ts_std(M.volume,20)+EPS) # 放量后反转
def a_vp_elasticity(M):  return -op.ts_corr(op.delta(np.log(M.volume+1),1), M.returns.abs(), 20) # 量价弹性
def a_amihud_trend(M):   il=M.returns.abs()/(M.amount+EPS); return op.delta(op.ts_mean(il,10),10) # 流动性恶化
def a_dvol_skew(M):      return -op.ts_skew(M.amount, 20)                                         # 成交额偏度
def a_volume_cv(M):      return op.ts_std(M.volume,20)/(op.ts_mean(M.volume,20)+EPS)              # 成交不稳定

# 自相关/微观结构（短期反转、知情交易的痕迹）
def a_ret_autocorr(M):   return -op.ts_corr(M.returns, op.delay(M.returns,1), 20)                 # 负自相关=反转
def a_vol_lead_ret(M):   return -op.ts_corr(op.delay(M.volume,1), M.returns, 20)                  # 量领先价
def a_signed_vol(M):     sv=np.sign(M.returns)*M.volume; return -op.ts_mean(sv,10)/(op.ts_mean(M.volume,10)+EPS)
def a_clv_smooth(M):     clv=((M.close-M.low)-(M.high-M.close))/(M.high-M.low+EPS); return op.ts_mean(clv,10)

# 价位锚定 / VWAP
def a_vwap_revert(M):    return -op.ts_mean((M.close-M.vwap)/(M.vwap+EPS), 5)                     # 偏离VWAP回归
def a_high_anchor(M):    return M.close/(op.ts_max(M.high,60)+EPS)                                # 距60日高(锚定)
def a_low_anchor(M):     return -(M.close/(op.ts_min(M.low,60)+EPS))                              # 距60日低
def a_drawdown(M):       return (M.close-op.ts_max(M.close,60))/(op.ts_max(M.close,60)+EPS)       # 当前回撤深度
def a_intraday_vol(M):   idr = M.close/M.open - 1; return -op.ts_std(idr, 20)                     # 日内波动(低波)

CURATED = {
    "co_strength": a_co_strength, "hl_vwap": a_hl_vwap, "open_vol": a_open_vol,
    "rank_low": a_rank_low, "vol_price": a_vol_price, "pv_corr": a_pv_corr,
    "ret_rev": a_ret_rev, "low_close": a_low_close, "high_decay": a_high_decay,
    "amihud_d": a_amihud, "vwap_mom": a_vwap_mom, "vol_std": a_vol_std,
    "turn_rank": a_turn_rank, "max_pos": a_max_pos, "corr_vwap_vol": a_corr_vwap_vol,
    "close_acc": a_close_acc,
    # —— 拓展 34 个 ——
    "overnight_rev": a_overnight_rev, "intraday_mom": a_intraday_mom,
    "overnight_mom": a_overnight_mom, "on_id_diverge": a_on_id_diverge, "gap_fill": a_gap_fill,
    "parkinson": a_parkinson, "garman_klass": a_garman_klass, "vol_of_vol": a_vol_of_vol,
    "range_expand": a_range_expand, "range_trend": a_range_trend,
    "ret_skew": a_ret_skew, "ret_kurt": a_ret_kurt, "downside_vol": a_downside_vol,
    "vol_asym": a_vol_asym, "max_range": a_max_range,
    "efficiency": a_efficiency, "variance_ratio": a_variance_ratio, "up_day_ratio": a_up_day_ratio,
    "price_curvature": a_price_curvature,
    "turnover_accel": a_turnover_accel, "volume_shock": a_volume_shock, "vp_elasticity": a_vp_elasticity,
    "amihud_trend": a_amihud_trend, "dvol_skew": a_dvol_skew, "volume_cv": a_volume_cv,
    "ret_autocorr": a_ret_autocorr, "vol_lead_ret": a_vol_lead_ret, "signed_vol": a_signed_vol,
    "clv_smooth": a_clv_smooth,
    "vwap_revert": a_vwap_revert, "high_anchor": a_high_anchor, "low_anchor": a_low_anchor,
    "drawdown": a_drawdown, "intraday_vol": a_intraday_vol,
}


# ============ B) 组合生成器 ============
def _base_signals(M) -> dict:
    """基础信号矩阵（因子工厂的'原材料'）。"""
    return {
        "ret1":     M.returns,
        "mom20":    M.close / op.delay(M.close, 20) - 1,
        "clo":      M.close,
        "vol":      M.volume,
        "dvol":     M.amount,
        "vwgap":    (M.close - M.vwap) / (M.vwap + EPS),
        "hl":       (M.high - M.low) / (M.close + EPS),
        "cogap":    (M.close - M.open) / (M.open + EPS),
        "illiq":    M.returns.abs() / (M.amount + EPS),
        "turn":     M.volume / (M.adv20 + EPS),
    }


def _unary_ops() -> dict:
    """作用在单个信号上的一元算子（生成新因子）。late-binding 用默认参数固定。"""
    return {
        "rank":   lambda x: op.rank(x),
        "neg":    lambda x: -x,
        "d5":     lambda x: op.delta(x, 5),
        "d10":    lambda x: op.delta(x, 10),
        "d20":    lambda x: op.delta(x, 20),
        "tsr10":  lambda x: op.ts_rank(x, 10),
        "tsr20":  lambda x: op.ts_rank(x, 20),
        "z20":    lambda x: (x - op.ts_mean(x, 20)) / (op.ts_std(x, 20) + EPS),
        "mean5r": lambda x: x / (op.ts_mean(x, 5) + EPS),
    }


def generate(M) -> dict:
    """基础信号 × 一元算子 → 批量因子（≈ 10×8 = 80 个）。"""
    signals, ops = _base_signals(M), _unary_ops()
    reg = {}
    for sname, sig in signals.items():
        for oname, fn in ops.items():
            reg[f"{sname}_{oname}"] = (lambda M_, _sig=sig, _fn=fn: _fn(_sig))
    return reg


def build_registry(M) -> dict:
    """合并 手写 + 组合生成 的全部因子，返回 {名称: func(M)->矩阵}。"""
    reg = dict(CURATED)
    reg.update(generate(M))
    return reg
