# -*- coding: utf-8 -*-
"""
研究脚本 29 —— 产品级「中证1000 指数增强」(优化器版)
================================================================
持中证1000市值域(rank 800-1800,市值加权合成基准),约束优化:
  max alpha·w  s.t. 行业中性 + 风格中性 + 跟踪误差预算 + 个股主动权重上限 + 多头。
alpha = 全231因子 LightGBM walk-forward OOS 预测(results/lgb_oos_pred.parquet)。
给绝对(产品净值)+ 相对(超额/IR/超额回撤)双口径。

用法：/opt/anaconda3/bin/python research/29_csi1000_product.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, backtest, fundamentals, altdata,
                      riskmodel, optimizer)
from quantlib.factors import classic

FREQ, C, PPY = "M", 0.003, 12
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]
LO, HI = 800, 1800          # 中证1000 市值域(剔除沪深300+中证500大盘、剔最小微盘)


def main():
    t0 = time.time()
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    pred = pd.read_parquet("results/lgb_oos_pred.parquet"); pred["trddt"] = pred["trddt"].astype("datetime64[ns]")
    panel = panel.merge(pred, on=["stkcd", "trddt"], how="left").rename(columns={"lgb": "alpha"})
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    panel["caprank"] = panel.groupby("trddt")["total_mktcap"].rank(ascending=False, method="first")
    i1000 = backtest.load_benchmark("000852", FREQ)
    print(f"准备完成 {time.time()-t0:.0f}s，逐月优化 ...", flush=True)

    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")

    TES = [0.03, 0.05]
    rows, wprev = [], None
    for dt in sorted(panel[panel["alpha"].notna()]["trddt"].unique()):
        m = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna() &
                  (panel["caprank"] > LO) & (panel["caprank"] <= HI)].copy()
        if len(m) < 200: continue
        b = (m["total_mktcap"] / m["total_mktcap"].sum()).values          # 市值加权合成基准
        Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values
        rec = {"dt": dt, "bench": float(np.nansum(b * fwd)), "i1000": i1000.get(dt, np.nan)}
        for te in TES:
            w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, d,
                                            active_cap=0.02, te=te, style_band=0.10)
            ws = pd.Series(w, index=m["stkcd"].values)
            to = 0.0 if (wprev is None or te != TES[0]) else 0.5 * ws.subtract(wprev, fill_value=0).abs().sum()
            rec[f"opt{te}"] = float(np.nansum(w * fwd)); rec[f"to{te}"] = to
            if te == TES[0]: wprev = ws
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("dt")

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 80, "\n中证1000 指数增强(优化器版,OOS,扣换手) —— 双口径\n", "=" * 80, sep="")
    for te in TES:
        port = R[f"opt{te}"] - R[f"to{te}"] * C
        exS = (port - R["bench"]).dropna()              # 对合成基准(优化器控制的)
        exI = (port - R["i1000"]).dropna()              # 对真实中证1000指数
        navp = (1 + port).cumprod(); navx = (1 + exS).cumprod()
        ann = (1 + port).prod() ** (PPY / len(port)) - 1
        print(f"\n--- 跟踪误差预算 TE={te:.0%} ---")
        print(f"  产品绝对: 年化{ann:.1%}  夏普{port.mean()/port.std(ddof=1)*np.sqrt(PPY):.2f}  "
              f"最大回撤{(navp/navp.cummax()-1).min():.1%}")
        print(f"  对合成基准: 超额{exS.mean()*PPY:.1%}  IR{exS.mean()/exS.std(ddof=1)*np.sqrt(PPY):.2f}  "
              f"跟踪误差{exS.std(ddof=1)*np.sqrt(PPY):.1%}  超额回撤{(navx/navx.cummax()-1).min():.1%}  月胜率{(exS>0).mean():.0%}")
        print(f"  对真实中证1000: 超额{exI.mean()*PPY:.1%}  IR{exI.mean()/exI.std(ddof=1)*np.sqrt(PPY):.2f}  年化换手{R[f'to{te}'].mean()*PPY:.1f}x")
    # 存净值(TE3%)
    te = TES[0]; port = R[f"opt{te}"] - R[f"to{te}"] * C
    pd.DataFrame({"指增产品": (1 + port).cumprod(), "合成基准": (1 + R["bench"]).cumprod(),
                  "中证1000": (1 + R["i1000"]).cumprod(),
                  "超额净值": (1 + (port - R["bench"])).cumprod()}).to_csv("results/29_nav.csv", encoding="utf-8-sig")
    print(f"\n完成 {time.time()-t0:.0f}s，净值存 results/29_nav.csv")


if __name__ == "__main__":
    main()
