# -*- coding: utf-8 -*-
"""
研究脚本 31 —— 中证1000 指增（真实成分权重版）
================================================================
用 iFinD 拉的真实中证1000成分权重(raw/IFIND_CSI1000_Weights.parquet, 半年快照)
替代之前的"市值域合成基准"。预期:对真实中证1000的超额回撤从 −13% 压到 ~−4%、IR 更可信。
alpha = 全231因子 LightGBM walk-forward OOS(results/lgb_oos_pred.parquet)。

用法：/opt/anaconda3/bin/python research/31_csi1000_real.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import data, universe, preprocess, backtest, altdata, riskmodel, optimizer
from quantlib.factors import classic

FREQ, C, PPY = "M", 0.003, 12
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]


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
    i1000 = backtest.load_benchmark("000852", FREQ)

    # 真实中证1000成分(半年快照) → as-of
    cw = pd.read_parquet(os.path.join(data.DATA_ROOT, "raw", "IFIND_CSI1000_Weights.parquet"))
    cw["trddt"] = pd.to_datetime(cw["trddt"]); cw["stkcd"] = cw["stkcd"].astype(str).str.zfill(6)
    snaps = np.sort(cw["trddt"].unique())
    print(f"准备完成 {time.time()-t0:.0f}s，真实成分快照 {len(snaps)} 个，逐月优化 ...", flush=True)

    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")

    TES = [0.03, 0.05]
    rows, wprev = [], None
    for dt in sorted(panel[panel["alpha"].notna()]["trddt"].unique()):
        sn = snaps[snaps <= np.datetime64(dt)]
        if not len(sn):
            continue
        cons = cw[cw["trddt"] == sn[-1]][["stkcd", "weight"]]                 # 最近一次调样的真实成分
        g = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna()]
        m = g.merge(cons, on="stkcd", how="inner")
        if len(m) < 300:
            continue
        b = (m["weight"] / m["weight"].sum()).values                          # 真实权重(归一)
        Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values
        rec = {"dt": dt, "bench": float(np.nansum(b * fwd)), "i1000": i1000.get(dt, np.nan), "ncons": len(m)}
        for te in TES:
            w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, d, active_cap=0.02, te=te, style_band=0.10)
            ws = pd.Series(w, index=m["stkcd"].values)
            to = 0.0 if (wprev is None or te != TES[0]) else 0.5 * ws.subtract(wprev, fill_value=0).abs().sum()
            rec[f"opt{te}"] = float(np.nansum(w * fwd)); rec[f"to{te}"] = to
            if te == TES[0]: wprev = ws
        rows.append(rec)
    R = pd.DataFrame(rows).set_index("dt")

    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n平均成分覆盖(∩可投): {R['ncons'].mean():.0f} 只")
    print("=" * 78, "\n中证1000 指增（真实成分权重，OOS，扣换手）\n", "=" * 78, sep="")
    for te in TES:
        port = R[f"opt{te}"] - R[f"to{te}"] * C
        exI = (port - R["i1000"]).dropna()
        navp = (1 + port).cumprod(); navx = (1 + exI).cumprod()
        ann = (1 + port).prod() ** (PPY / len(port)) - 1
        print(f"\n--- TE={te:.0%} ---")
        print(f"  产品绝对: 年化{ann:.1%} 夏普{port.mean()/port.std(ddof=1)*np.sqrt(PPY):.2f} 最大回撤{(navp/navp.cummax()-1).min():.1%}")
        print(f"  对真实中证1000: 超额{exI.mean()*PPY:.2%} IR{exI.mean()/exI.std(ddof=1)*np.sqrt(PPY):.2f} "
              f"跟踪误差{exI.std(ddof=1)*np.sqrt(PPY):.1%} 超额回撤{(navx/navx.cummax()-1).min():.1%} 月胜率{(exI>0).mean():.0%} 换手{R[f'to{te}'].mean()*PPY:.1f}x")
    te = TES[0]; port = R[f"opt{te}"] - R[f"to{te}"] * C
    pd.DataFrame({"指增产品": (1 + port).cumprod(), "中证1000": (1 + R["i1000"]).cumprod(),
                  "超额净值": (1 + (port - R["i1000"])).cumprod()}).to_csv("results/31_nav.csv", encoding="utf-8-sig")
    print(f"\n完成 {time.time()-t0:.0f}s，净值存 results/31_nav.csv")


if __name__ == "__main__":
    main()
