# -*- coding: utf-8 -*-
"""
研究脚本 32 —— 中证1000 指增 最终版（真实成分 + 换手惩罚）
================================================================
真实成分(IFIND_CSI1000_Weights) + 换手惩罚(optimize_enhanced gamma)。
换手惩罚是 workflow 选出的承重杠杆:降churn、提净超额、降回撤。扫 gamma 取最优。

用法：/opt/anaconda3/bin/python research/32_csi1000_final.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import data, universe, preprocess, backtest, altdata, riskmodel, optimizer
from quantlib.factors import classic

FREQ, C, PPY, TE = "M", 0.003, 12, 0.03
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
    cw = pd.read_parquet(os.path.join(data.DATA_ROOT, "raw", "IFIND_CSI1000_Weights.parquet"))
    cw["trddt"] = pd.to_datetime(cw["trddt"]); cw["stkcd"] = cw["stkcd"].astype(str).str.zfill(6)
    snaps = np.sort(cw["trddt"].unique())
    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")
    dates = sorted(panel[panel["alpha"].notna()]["trddt"].unique())
    print(f"准备完成 {time.time()-t0:.0f}s，扫 gamma ...", flush=True)
    pd.set_option("display.unicode.east_asian_width", True)

    def run(gamma):
        rows, wprev = [], None
        for dt in dates:
            sn = snaps[snaps <= np.datetime64(dt)]
            if not len(sn): continue
            cons = cw[cw["trddt"] == sn[-1]][["stkcd", "weight"]]
            g = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna()]
            m = g.merge(cons, on="stkcd", how="inner")
            if len(m) < 300: continue
            b = (m["weight"] / m["weight"].sum()).values
            Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
            F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
            Xs = m[style_cols].fillna(0.0).values
            d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
            wp = b if (gamma > 0 and wprev is None) else (wprev.reindex(m["stkcd"]).fillna(0).values if gamma > 0 else None)
            w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, d,
                                            active_cap=0.02, te=TE, style_band=0.10, gamma=gamma, w_prev=wp)
            ws = pd.Series(w, index=m["stkcd"].values)
            to = 0.0 if wprev is None else 0.5 * ws.subtract(wprev, fill_value=0).abs().sum()
            rows.append({"dt": dt, "port": float(np.nansum(w * m["fwd_ret"].values)), "to": to,
                         "i1000": i1000.get(dt, np.nan)}); wprev = ws
        R = pd.DataFrame(rows).set_index("dt")
        net = R["port"] - R["to"] * C
        ex = (net - R["i1000"]).dropna(); nav = (1 + ex).cumprod()
        return {"净超额": ex.mean() * PPY, "IR": ex.mean() / ex.std(ddof=1) * np.sqrt(PPY),
                "跟踪误差": ex.std(ddof=1) * np.sqrt(PPY), "超额回撤": (nav / nav.cummax() - 1).min(),
                "月胜率": (ex > 0).mean(), "年化换手": R["to"].mean() * PPY}, net, R["i1000"]

    res, best = {}, None
    for g in [0.0, 0.003, 0.005]:
        r, net, bench = run(g)
        res[f"gamma={g}"] = r
        if best is None or r["IR"] > res[f"gamma={best[0]}"]["IR"]:
            best = (g, net, bench)
        print(f"  gamma={g}: IR{r['IR']:.2f} 净超额{r['净超额']:.2%} 换手{r['年化换手']:.1f}x 超额回撤{r['超额回撤']:.1%}", flush=True)

    out = pd.DataFrame(res).T
    for c in ["净超额", "跟踪误差", "超额回撤"]: out[c] = (out[c] * 100).round(1).astype(str) + "%"
    out["IR"] = out["IR"].round(2); out["月胜率"] = (out["月胜率"] * 100).round(0).astype(int).astype(str) + "%"
    out["年化换手"] = out["年化换手"].round(1)
    print("\n" + "=" * 76, "\n中证1000指增 最终版(真实成分+换手惩罚, 对真实中证1000, OOS, 扣换手)\n", "=" * 76, sep="")
    print(out.to_string())
    g, net, bench = best
    pd.DataFrame({"指增产品": (1 + net).cumprod(), "中证1000": (1 + bench).cumprod(),
                  "超额净值": (1 + (net - bench)).cumprod()}).to_csv("results/32_nav.csv", encoding="utf-8-sig")
    print(f"\n最优 gamma={g}；完成 {time.time()-t0:.0f}s，净值存 results/32_nav.csv")


if __name__ == "__main__":
    main()
