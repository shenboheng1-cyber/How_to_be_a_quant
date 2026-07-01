# -*- coding: utf-8 -*-
"""
研究脚本 15 —— AlphaNet 端到端深度学习（vs LightGBM）
================================================================
AlphaNet 提取特征(价量两两相关/波动/衰减) → torch MLP，防泄漏 walk-forward。
对比同特征下的 LightGBM，以及参照 231 因子模型(IR 1.08)。

用法(base 环境有 torch)：/opt/anaconda3/bin/python research/15_alphanet.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 防 torch+lightgbm 的 libomp 冲突(OMP #15)静默崩溃

import numpy as np, pandas as pd
import torch, torch.nn as nn
from quantlib import data, universe, preprocess, evaluate, ml
from quantlib.alpha import matrices, alphanet, factory

FREQ, START, END = "M", "2015-01-01", "2025-12-31"


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(d), nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def torch_wf(X, y, dates, init=36, embargo=1, step=12, epochs=25, bs=8192):
    uniq = sorted(pd.unique(dates))
    pred = np.full(len(y), np.nan)
    Xf = np.nan_to_num(X).astype("float32")
    torch.manual_seed(0)
    i = 0 + init
    while i < len(uniq):
        test = uniq[i:i + step]
        train = uniq[:max(0, i - embargo)]
        tr = np.where(np.isin(dates, train) & ~np.isnan(y))[0]
        te = np.isin(dates, test)
        if len(tr) > 5000:
            m = MLP(Xf.shape[1])
            opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
            Xt = torch.tensor(Xf[tr]); yt = torch.tensor(y[tr].astype("float32"))
            m.train()
            for ep in range(epochs):                       # minibatch
                perm = torch.randperm(len(tr))
                for b in range(0, len(tr), bs):
                    idx = perm[b:b + bs]
                    opt.zero_grad()
                    loss = ((m(Xt[idx]) - yt[idx]) ** 2).mean()
                    loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                pred[te] = m(torch.tensor(Xf[te])).numpy()
        print(f"  fold@{pd.Timestamp(uniq[i]).date()} 训练{len(tr)}行", flush=True)
        i += step
    return pred


def main():
    t = time.time()
    M = matrices.load_matrices(START, END)
    feats = alphanet.extract_features(M)
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    cols = {}
    for nm, fn in feats.items():                       # 惰性：逐个算→采样→丢弃，省内存
        raw = factory.sample_to_panel(fn(M), panel)
        cols[nm] = preprocess.preprocess_factor(panel, raw, do_neutralize=True).values
    X = np.column_stack(list(cols.values()))
    y = ml.make_label(panel)
    dates = panel["trddt"].values
    print(f"AlphaNet 特征 {X.shape} | {time.time()-t:.0f}s", flush=True)
    # 存盘：供 csmar 环境单独跑 LightGBM 对比（torch 与 lightgbm 同进程会因 libomp 冲突崩溃）
    np.savez("results/15_alphanet_data.npz", X=X.astype("float32"),
             y=y.astype("float32"), dates=dates.astype("datetime64[D]").astype(str))

    preds = {"AlphaNet(MLP)": torch_wf(X, y, dates)}
    pd.set_option("display.unicode.east_asian_width", True)
    rows = {}
    for name, pred in preds.items():
        oos = ~np.isnan(pred)
        sub = panel[oos].reset_index(drop=True)
        f = pd.Series(pred[oos])
        ic = evaluate.ic_summary(evaluate.compute_ic(sub, f))
        ls = evaluate.quantile_summary(evaluate.quantile_returns(sub, f, 10)).loc["多空(QN-Q1)"]
        rows[name] = {"RankIC": ic["IC均值"], "ICIR": ic["ICIR"], "t值": ic["t值"], "多空夏普": ls["夏普"]}
    print("\n" + "=" * 60, "\nAlphaNet(MLP) 样本外表现\n", "=" * 60, sep="")
    print(pd.DataFrame(rows).T.to_string())
    print("\n参照：231因子 LightGBM 多空夏普≈2.96、ICIR≈1.03（同特征 LightGBM 见 research/15b）")
    pd.DataFrame(rows).T.to_csv("results/15_alphanet.csv", encoding="utf-8-sig")
    print("已保存 results/15_alphanet.csv")


if __name__ == "__main__":
    main()
