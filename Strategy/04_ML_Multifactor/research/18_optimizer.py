# -*- coding: utf-8 -*-
"""
研究脚本 18 —— 风险模型 + 组合优化器 vs naive top-decile（L5）
================================================================
同一个 alpha，对比两种组合构建：
  A) naive top-decile 等权(现有)
  B) 优化器(行业/风格中性 + 权重上限 + 风险约束)
看 IR / 回撤 / 最大权重 / 行业集中度 改善。

用法(base 环境有 cvxpy)：/opt/anaconda3/bin/python research/18_optimizer.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import data, universe, preprocess, backtest, altdata, riskmodel, optimizer
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]


def main():
    t0 = time.time()
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)

    # 风格暴露(标准化, 不中性) + alpha(部分风格的行业+市值中性合成)
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    aset = ["reversal", "low_turnover", "low_vol", "illiquidity", "ep"]
    A = [preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel),
                                      industry_col="industry", do_neutralize=True) for k in aset]
    panel["alpha"] = pd.concat(A, axis=1).mean(axis=1).values

    print(f"风险模型估计 ... {time.time()-t0:.0f}s")
    f_df, resid, ind_levels = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid)
    sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)        # PIT：用上一期特质风险
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")

    bench = backtest.load_benchmark("000905", FREQ)
    dates = sorted(panel["trddt"].unique())
    rows = []
    w_prev = None
    for dt in dates[24:]:                                          # 跳过风险模型预热
        g = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna()].copy()
        if len(g) < 200:
            continue
        # 候选:alpha 前 500
        cand = g.nlargest(500, "alpha").reset_index(drop=True)
        # baseline: top decile 等权
        td = g.nlargest(max(1, len(g) // 10), "alpha")
        td_ret = td["fwd_ret"].mean()
        # 优化器输入
        Xind = pd.get_dummies(cand["industry"]).astype(float)
        cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = cand[style_cols].fillna(0.0).values
        d = cand["specvar"].fillna(cand["specvar"].median()).fillna(0.04).values
        w = optimizer.optimize(cand["alpha"].values, Xind.values, Xs, F, d, cap=0.03, lam=8.0)
        opt_ret = float(np.nansum(w * cand["fwd_ret"].values))
        # 换手 & 集中度
        wser = pd.Series(w, index=cand["stkcd"].values)
        to = 0.0 if w_prev is None else 0.5 * (wser.subtract(w_prev, fill_value=0).abs().sum())
        rows.append({"dt": dt, "opt": opt_ret, "td": td_ret, "bench": bench.get(dt, np.nan),
                     "to": to, "maxw": w.max(), "nhold": int((w > 1e-4).sum())})
        w_prev = wser
    R = pd.DataFrame(rows).set_index("dt")

    def stats(col, cost):
        r = R[col] - (R["to"] * cost if col == "opt" else 0)        # 优化器扣换手成本
        m = backtest.metrics(r, FREQ)
        ir = backtest.info_ratio(r - R["bench"], FREQ)
        return {**m, "对500_IR": ir}

    pd.set_option("display.unicode.east_asian_width", True)
    out = pd.DataFrame({
        "naive top-decile": {**backtest.metrics(R["td"], FREQ), "对500_IR": backtest.info_ratio(R["td"] - R["bench"], FREQ),
                             "年化换手": "—", "平均持仓": int(0.1 * panel.groupby("trddt").size().mean()), "最大权重": "等权"},
        "优化器(中性+约束)": {**stats("opt", 0.003), "年化换手": round(R["to"].mean() * 12, 1),
                          "平均持仓": int(R["nhold"].mean()), "最大权重": round(R["maxw"].mean(), 3)},
    }).T
    print("\n" + "=" * 70, "\n组合优化 vs naive top-decile（同 alpha，样本外，优化器扣千3换手）\n", "=" * 70, sep="")
    print(out.to_string())
    out.to_csv("results/18_optimizer.csv", encoding="utf-8-sig")
    print(f"\n完成 {time.time()-t0:.0f}s，已保存")


if __name__ == "__main__":
    main()
