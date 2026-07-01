# -*- coding: utf-8 -*-
"""
研究脚本 29 —— 中证1000 指增 + 【拥挤/波动择时降主动风险】
================================================================
基线 = 29_csi1000_product.py (TE=3% 优化器版指增)。
本脚本在基线【超额收益流】上叠加时序降仓 (regime)：
  scaled_excess_t = scale_t * excess_t,  scale_t 用滞后信号, cap=1 (只减不加)。
两个信号:
  (A) vol_target  : 超额流近期实现波动越高→scale 越低 (target = 全样本中位波动)
  (B) crowding+derisk : 因子收益两两相关进入历史高位 (拥挤/踩踏) → 降仓到 low_expo
对比降仓前后: 超额% / IR / 超额回撤% / 超额卡玛。

用法：/opt/anaconda3/bin/python research/29_crowdtiming.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, backtest, fundamentals, altdata,
                      riskmodel, optimizer, regime)
from quantlib.factors import classic

FREQ, C, PPY = "M", 0.003, 12
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]
LO, HI = 800, 1800
TE = 0.03                       # 只跑 TE=3% 这一个口径


def summarize(ex, label):
    ex = ex.dropna()
    nav = (1 + ex).cumprod()
    mdd = (nav / nav.cummax() - 1).min()
    ann = ex.mean() * PPY
    ir = ex.mean() / ex.std(ddof=1) * np.sqrt(PPY)
    te = ex.std(ddof=1) * np.sqrt(PPY)
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    print(f"  {label:28s} 超额{ann:6.1%}  IR{ir:5.2f}  TE{te:5.1%}  超额回撤{mdd:6.1%}  "
          f"卡玛{calmar:5.2f}  月胜率{(ex>0).mean():4.0%}")
    return dict(ann=ann, ir=ir, te=te, mdd=mdd, calmar=calmar)


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

    rows, wprev = [], None
    for dt in sorted(panel[panel["alpha"].notna()]["trddt"].unique()):
        m = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna() &
                  (panel["caprank"] > LO) & (panel["caprank"] <= HI)].copy()
        if len(m) < 200: continue
        b = (m["total_mktcap"] / m["total_mktcap"].sum()).values
        Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values
        w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, d,
                                        active_cap=0.02, te=TE, style_band=0.10)
        ws = pd.Series(w, index=m["stkcd"].values)
        to = 0.0 if wprev is None else 0.5 * ws.subtract(wprev, fill_value=0).abs().sum()
        rows.append({"dt": dt, "bench": float(np.nansum(b * fwd)),
                     "i1000": i1000.get(dt, np.nan),
                     "opt": float(np.nansum(w * fwd)), "to": to})
        wprev = ws
    R = pd.DataFrame(rows).set_index("dt")

    # ---- 基线净超额流 (扣换手成本) ----
    port = R["opt"] - R["to"] * C
    exI = (port - R["i1000"]).dropna()              # 对真实中证1000 (净超额)
    bench_to = R["to"]                              # 用于成本重算

    # ============ 两个择时信号 (滞后, cap=1, 只减不加) ============
    # 信号A: vol_target —— target=全样本超额波动中位, 高波动月降仓
    realized = exI.rolling(6, min_periods=3).std().shift(1) * np.sqrt(PPY)
    tgt = realized.median()                          # 自适应目标 ≈ 长期超额波动
    scaleA = (tgt / realized).clip(upper=1.0).fillna(1.0)   # cap=1 只减不加

    # 信号B: crowding_index + derisk —— 因子(行业+风格)收益两两相关进高位→降仓
    crowd = regime.crowding_index(f_df[cols], lookback=12)
    crowd.index = pd.to_datetime(crowd.index)
    crowd = crowd.reindex(exI.index)
    scaleB = regime.derisk(exI, crowd, hi_quantile=0.8, low_expo=0.5)

    # 信号C: A 与 B 取较低 (两者都触发才不降, 任一触发即降) → min
    scaleC = pd.concat([scaleA.reindex(exI.index), scaleB.reindex(exI.index)], axis=1).min(axis=1).fillna(1.0)

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 92)
    print(f"中证1000 指增 + 拥挤/波动择时 (OOS 2018-2025, TE={TE:.0%}, 扣 {C:.1%} 换手) —— 对真实中证1000")
    print("=" * 92)
    base = summarize(exI, "基线 (无降仓)")

    print("\n--- 降仓后 (scaled_excess = scale_t * excess_t) ---")
    resA = summarize(scaleA.reindex(exI.index).fillna(1.0) * exI, "A. vol_target")
    resB = summarize(scaleB.reindex(exI.index).fillna(1.0) * exI, "B. crowding+derisk")
    resC = summarize(scaleC * exI, "C. min(A,B) 任一触发即降")

    # 降仓触发统计
    print(f"\n  触发降仓占比: A {(scaleA<0.999).mean():.0%}  B {(scaleB<0.999).mean():.0%}  C {(scaleC<0.999).mean():.0%}")
    print(f"  年化换手(基线): {bench_to.mean()*PPY:.1f}x")
    # 2024 踩踏期 (2024-01~2024-02) 检查
    sub = exI.loc["2024-01-01":"2024-06-30"]
    if len(sub):
        print(f"  2024H1 基线超额: {sub.sum():.1%}  A降仓后: {(scaleA.reindex(sub.index).fillna(1.0)*sub).sum():.1%}  "
              f"B: {(scaleB.reindex(sub.index).fillna(1.0)*sub).sum():.1%}")

    print(f"\n完成 {time.time()-t0:.0f}s")
    # 把关键结果存盘供报告
    out = pd.DataFrame({"exI_base": exI,
                        "scaleA": scaleA.reindex(exI.index), "scaleB": scaleB.reindex(exI.index),
                        "scaleC": scaleC,
                        "exI_A": scaleA.reindex(exI.index).fillna(1.0)*exI,
                        "exI_B": scaleB.reindex(exI.index).fillna(1.0)*exI,
                        "exI_C": scaleC*exI})
    out.to_csv("results/29_crowdtiming.csv", encoding="utf-8-sig")
    return base, resA, resB, resC


if __name__ == "__main__":
    main()
