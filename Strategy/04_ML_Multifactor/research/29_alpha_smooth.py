# -*- coding: utf-8 -*-
"""
研究脚本 29_alpha_smooth —— 中证1000指增 + 【alpha 时序平滑】杠杆
================================================================
基线 29_csi1000_product.py 的改进：优化前对每只股票的 lgb 信号做时序 EWMA
(按 stkcd 分组, halflife = 2 / 3 个月), 用平滑后的信号当 alpha 再跑优化器。
EWMA 只用当期及过去 (pandas ewm 天然因果), 无前视。

对比 平滑(hl=2/3) vs 不平滑 的 超额/IR/换手, 报告更优的 halflife。

用法：/opt/anaconda3/bin/python research/29_alpha_smooth.py
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
LO, HI = 800, 1800
TE = 0.03                          # 报告口径固定 TE=3%
HALFLIVES = [2, 3]                 # 试 EWMA halflife = 2, 3 个月


def smooth_alpha(panel, hl):
    """按 stkcd 时序 EWMA 平滑 lgb 原始信号。pandas ewm 只用当期及过去, 无前视。
    返回与 panel 行对齐的平滑 alpha 序列。"""
    df = panel[["stkcd", "trddt", "lgb_raw"]].copy()
    df = df.sort_values(["stkcd", "trddt"])
    df["sm"] = (df.groupby("stkcd")["lgb_raw"]
                  .transform(lambda s: s.ewm(halflife=hl, min_periods=1).mean()))
    return df["sm"].reindex(panel.index)


def main():
    t0 = time.time()
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    pred = pd.read_parquet("results/lgb_oos_pred.parquet"); pred["trddt"] = pred["trddt"].astype("datetime64[ns]")
    panel = panel.merge(pred, on=["stkcd", "trddt"], how="left").rename(columns={"lgb": "lgb_raw"})
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    panel["caprank"] = panel.groupby("trddt")["total_mktcap"].rank(ascending=False, method="first")
    i1000 = backtest.load_benchmark("000852", FREQ)

    # 预计算各种 alpha 变体: raw(不平滑) + 每个 halflife 的平滑
    alpha_variants = {"raw": panel["lgb_raw"].copy()}
    for hl in HALFLIVES:
        alpha_variants[f"hl{hl}"] = smooth_alpha(panel, hl)
    print(f"准备完成 {time.time()-t0:.0f}s，逐月优化 ...", flush=True)

    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")

    variants = list(alpha_variants.keys())
    rows = []
    wprev = {v: None for v in variants}     # 各变体各自的上期权重(算换手)
    has_pred = panel["lgb_raw"].notna()
    for dt in sorted(panel[has_pred]["trddt"].unique()):
        msk = ((panel["trddt"] == dt) & has_pred & panel["industry"].notna() &
               (panel["caprank"] > LO) & (panel["caprank"] <= HI))
        m = panel[msk].copy()
        if len(m) < 200:
            continue
        idx = m.index
        b = (m["total_mktcap"] / m["total_mktcap"].sum()).values
        Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values
        rec = {"dt": dt, "bench": float(np.nansum(b * fwd)), "i1000": i1000.get(dt, np.nan)}
        for v in variants:
            av = alpha_variants[v].reindex(idx).values
            # 平滑后偶有 NaN(该股首次出现时 min_periods=1 应无 NaN, 但稳妥处理)
            av = np.where(np.isnan(av), np.nan_to_num(av, nan=0.0), av)
            w = optimizer.optimize_enhanced(av, b, Xind.values, Xs, F, d,
                                            active_cap=0.02, te=TE, style_band=0.10)
            ws = pd.Series(w, index=m["stkcd"].values)
            to = 0.0 if wprev[v] is None else 0.5 * ws.subtract(wprev[v], fill_value=0).abs().sum()
            rec[f"opt_{v}"] = float(np.nansum(w * fwd)); rec[f"to_{v}"] = to
            wprev[v] = ws
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("dt")

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 84, "\n中证1000指增 + alpha时序平滑(EWMA) —— TE=3% 对真实中证1000\n", "=" * 84, sep="")
    summary = {}
    for v in variants:
        port = R[f"opt_{v}"] - R[f"to_{v}"] * C        # 扣换手成本
        gross = R[f"opt_{v}"]                          # 未扣成本
        exI = (port - R["i1000"]).dropna()             # 净超额(对真实中证1000)
        exI_g = (gross - R["i1000"]).dropna()          # 毛超额
        navx = (1 + exI).cumprod()
        te_real = exI.std(ddof=1) * np.sqrt(PPY)
        ir = exI.mean() / exI.std(ddof=1) * np.sqrt(PPY)
        ann_ex = exI.mean() * PPY
        ann_ex_g = exI_g.mean() * PPY
        mdd = (navx / navx.cummax() - 1).min()
        turn = R[f"to_{v}"].mean() * PPY
        summary[v] = dict(ex_net=ann_ex, ex_gross=ann_ex_g, ir=ir, te=te_real, mdd=mdd, turn=turn)
        label = {"raw": "不平滑(基线)", "hl2": "EWMA hl=2", "hl3": "EWMA hl=3"}[v]
        print(f"\n--- {label} ---")
        print(f"  毛超额{ann_ex_g:.2%}  净超额{ann_ex:.2%}  IR{ir:.2f}  跟踪误差{te_real:.2%}  "
              f"超额回撤{mdd:.2%}  年化换手{turn:.1f}x  月胜率{(exI>0).mean():.0%}")

    # 选更优 halflife (净超额优先, 同时看 IR/换手)
    best_hl = max(HALFLIVES, key=lambda h: summary[f"hl{h}"]["ir"])
    print("\n" + "=" * 84)
    print(f"对比基线(不平滑): 净超额{summary['raw']['ex_net']:.2%} IR{summary['raw']['ir']:.2f} 换手{summary['raw']['turn']:.1f}x")
    for h in HALFLIVES:
        s = summary[f"hl{h}"]
        print(f"  hl={h}: 净超额{s['ex_net']:.2%} IR{s['ir']:.2f} 换手{s['turn']:.1f}x "
              f"(IR {'+' if s['ir']>=summary['raw']['ir'] else ''}{s['ir']-summary['raw']['ir']:.2f}, "
              f"换手{(s['turn']/summary['raw']['turn']-1)*100:+.0f}%)")
    print(f"\n>> 更优 halflife = {best_hl} 个月")
    print(f"完成 {time.time()-t0:.0f}s")

    return summary, best_hl


if __name__ == "__main__":
    main()
