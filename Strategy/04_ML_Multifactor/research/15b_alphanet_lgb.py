# -*- coding: utf-8 -*-
"""
研究脚本 15b —— 同 AlphaNet 特征下的 LightGBM 基线（csmar 环境，无 torch）
读 research/15 存盘的特征，跑 LightGBM walk-forward，与 AlphaNet(MLP) 对比。
用法：DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib python research/15b_alphanet_lgb.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import ml, evaluate

d = np.load("results/15_alphanet_data.npz", allow_pickle=True)
X, y, dates = d["X"].astype(float), d["y"].astype(float), pd.to_datetime(d["dates"]).values
pred = ml.walk_forward_predict(X, y, dates, ml.lgb_model(), init=36, embargo=1, step=3)
oos = ~np.isnan(pred)
panel = pd.DataFrame({"trddt": dates[oos], "fwd_ret": y[oos]})
f = pd.Series(pred[oos])
ic = evaluate.ic_summary(evaluate.compute_ic(panel, f))
ls = evaluate.quantile_summary(evaluate.quantile_returns(panel, f, 10)).loc["多空(QN-Q1)"]
print(f"同特征 LightGBM: RankIC={ic['IC均值']} ICIR={ic['ICIR']} t={ic['t值']} 多空夏普={ls['夏普']}")
