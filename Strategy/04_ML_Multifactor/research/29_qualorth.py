# -*- coding: utf-8 -*-
"""
研究脚本 29_qualorth —— 中证1000指增 + 【集成正交质量腿】
================================================================
基线 29 的 alpha = 纯 LGB OOS 预测(偏动量小盘)。
本脚本构造一个正交的质量价值合成 qual：
  用 f_gross_prof, f_low_lev, f_accruals, f_ep, f_bp, f_roe 六个因子，
  各自 preprocess_factor(行业+市值中性) → 按横截面IC符号对齐 → zscore → 等权平均。
再 alpha = w_lgb*zscore(lgb) + (1-w_lgb)*qual，试 w_lgb=1.0(纯LGB基线)/0.7/0.5。
对比 超额/IR/超额回撤/换手。

性能：质量腿的行业中性化 preprocess 只在【可投资域】(caprank 800-1800,约1000股/月)
上做，而非全 5000 股面板——既符合"投资域内中性"的标准做法，又快 5x。

用法：/opt/anaconda3/bin/python -u research/29_qualorth.py
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
QUAL_FACTORS = ["f_gross_prof", "f_low_lev", "f_accruals", "f_ep", "f_bp", "f_roe"]
W_LGBS = [1.0, 0.7, 0.5]   # 1.0 = 纯LGB基线对照


def log(m):
    sys.stderr.write(f"[{time.time()-T0:7.1f}s] {m}\n"); sys.stderr.flush()


def xs_zscore(panel, s):
    df = pd.DataFrame({"trddt": panel["trddt"].values, "f": np.asarray(s)})
    return df.groupby("trddt")["f"].transform(preprocess.zscore).values


def build_qual(sub):
    """在可投资域子面板 sub 上构造集成正交质量腿 qual（与 sub 行对齐）。"""
    fwd = sub["fwd_ret"].values
    legs = []
    for fname in QUAL_FACTORS:
        raw = fundamentals.REGISTRY[fname][0](sub)
        z = preprocess.preprocess_factor(sub, raw, industry_col="industry", do_neutralize=True)
        tmp = pd.DataFrame({"z": z.values, "fwd": fwd}).dropna()
        ic = np.corrcoef(tmp["z"], tmp["fwd"])[0, 1] if len(tmp) > 100 else 0.0
        sign = 1.0 if ic >= 0 else -1.0
        legs.append(sign * z.values)
        log(f"    {fname:14s} IC={ic:+.4f} sign={sign:+.0f}")
    Q = np.nanmean(np.vstack(legs), axis=0)
    qual = xs_zscore(sub, Q)
    return qual


def main():
    global T0
    T0 = time.time()
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    panel = fundamentals.attach(panel)
    pred = pd.read_parquet("results/lgb_oos_pred.parquet"); pred["trddt"] = pred["trddt"].astype("datetime64[ns]")
    panel = panel.merge(pred, on=["stkcd", "trddt"], how="left").rename(columns={"lgb": "lgb_raw"})
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    panel["caprank"] = panel.groupby("trddt")["total_mktcap"].rank(ascending=False, method="first")
    i1000 = backtest.load_benchmark("000852", FREQ)
    log(f"准备完成，面板 {panel.shape}")

    # 风险模型（全panel，与基线一致）
    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")
    log("风险模型完成")

    # 可投资域子面板：质量腿+合成都在这里做（快且符合域内中性）
    sub = panel[panel["lgb_raw"].notna() & panel["industry"].notna() &
                (panel["caprank"] > LO) & (panel["caprank"] <= HI)].copy().reset_index(drop=True)
    log(f"可投资域子面板 {sub.shape}，构造质量腿 ...")
    sub["qual"] = build_qual(sub)
    sub["lgb_z"] = xs_zscore(sub, sub["lgb_raw"].values)
    log("质量腿完成，逐月优化 (TE=3%) ...")

    te = 0.03
    results = {}
    dts = sorted(sub["trddt"].unique())
    for w_lgb in W_LGBS:
        rows, wprev = [], None
        for dt in dts:
            m = sub[sub["trddt"] == dt].copy()
            if len(m) < 200: continue
            lz = preprocess.zscore(pd.Series(m["lgb_z"].values)).values
            qz = preprocess.zscore(pd.Series(np.nan_to_num(m["qual"].values))).values
            alpha = w_lgb * lz + (1.0 - w_lgb) * qz
            b = (m["total_mktcap"] / m["total_mktcap"].sum()).values
            Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
            F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
            Xs = m[style_cols].fillna(0.0).values
            d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
            fwd = m["fwd_ret"].values
            w = optimizer.optimize_enhanced(alpha, b, Xind.values, Xs, F, d,
                                            active_cap=0.02, te=te, style_band=0.10)
            ws = pd.Series(w, index=m["stkcd"].values)
            to = 0.0 if wprev is None else 0.5 * ws.subtract(wprev, fill_value=0).abs().sum()
            rows.append({"dt": dt, "bench": float(np.nansum(b * fwd)), "i1000": i1000.get(dt, np.nan),
                         "opt": float(np.nansum(w * fwd)), "to": to})
            wprev = ws
        results[w_lgb] = pd.DataFrame(rows).set_index("dt")
        log(f"w_lgb={w_lgb} 优化完成 ({len(rows)}月)")

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 88, "\n中证1000指增 + 集成正交质量腿 (OOS 2018-2025, TE=3%, 扣0.3%换手) —— 对真实中证1000\n", "=" * 88, sep="")
    summary = {}
    for w_lgb in W_LGBS:
        R = results[w_lgb]
        port = R["opt"] - R["to"] * C
        exI = (port - R["i1000"]).dropna()
        navx = (1 + exI).cumprod()
        ex = exI.mean() * PPY
        ir = exI.mean() / exI.std(ddof=1) * np.sqrt(PPY)
        teI = exI.std(ddof=1) * np.sqrt(PPY)
        mdd = (navx / navx.cummax() - 1).min()
        turn = R["to"].mean() * PPY
        gross_ex = (R["opt"] - R["i1000"]).dropna().mean() * PPY
        tag = "纯LGB(基线复现)" if w_lgb == 1.0 else f"w_lgb={w_lgb} (LGB+质量腿)"
        print(f"\n--- {tag} ---")
        print(f"  对真实中证1000: 净超额{ex:.2%}  IR{ir:.2f}  跟踪误差{teI:.2%}  "
              f"超额回撤{mdd:.2%}  年化换手{turn:.1f}x  月胜率{(exI>0).mean():.0%}")
        print(f"  (毛超额 未扣成本: {gross_ex:.2%})")
        summary[w_lgb] = dict(ex=ex*100, ir=ir, te=teI*100, mdd=mdd*100, turn=turn, gross=gross_ex*100)
    print("\nSUMMARY_JSON " + repr(summary))

    out = {}
    for w_lgb in W_LGBS:
        R = results[w_lgb]; port = R["opt"] - R["to"] * C
        out[f"超额净值_w{w_lgb}"] = (1 + (port - R["i1000"])).cumprod()
    pd.DataFrame(out).to_csv("results/29_qualorth_nav.csv", encoding="utf-8-sig")
    log("完成，净值存 results/29_qualorth_nav.csv")
    return summary


if __name__ == "__main__":
    main()
