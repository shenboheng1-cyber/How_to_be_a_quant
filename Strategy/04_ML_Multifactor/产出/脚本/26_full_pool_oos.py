# -*- coding: utf-8 -*-
"""
研究脚本 26 —— 全 231 因子的 OOS（回应"是不是只在10个里选"）
================================================================
用 08_features.parquet 的全部 231 个因子(已标准化)+标签 y，两种 OOS：
  (a) LightGBM walk-forward(purged CV)——本就是合法OOS，强方法
  (b) 贪心逐期选择——在全 231 里选(对比之前 32 因子版 ICIR 0.64)
再把 (a) 的 OOS alpha 部署到 市场中性 / 中证800指增，给真实可交易指标。

用法：/opt/anaconda3/bin/python research/26_full_pool_oos.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, evaluate, backtest, ml,
                      altdata, riskmodel, optimizer)
from quantlib.factors import classic

FREQ, C, WARM, PPY = "M", 0.003, 36, 12
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]


def rank_ic(df, fcol, ycol="y"):
    d = df[[ "trddt", fcol, ycol]].dropna()
    return d.groupby("trddt").apply(lambda x: x[fcol].rank().corr(x[ycol].rank()))


def metrics_full(r, bench=None):
    r = r.dropna(); n = len(r)
    ann = (1 + r).prod() ** (PPY / n) - 1; vol = r.std(ddof=1) * np.sqrt(PPY)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(PPY)
    nav = (1 + r).cumprod(); mdd = (nav / nav.cummax() - 1).min()
    cal = ann / abs(mdd) if mdd else np.nan
    if bench is not None:
        b = bench.reindex(r.index); ex = (r - b).dropna()
        exann = (1 + r).prod() ** (PPY / n) - (1 + b.reindex(r.index)).prod() ** (PPY / n); win = (ex > 0).mean()
    else:
        exann = ann; win = (r > 0).mean()
    return {"年化": ann, "波动": vol, "夏普": sharpe, "回撤": mdd, "卡玛": cal, "超额": exann, "胜率": win}


def main():
    t0 = time.time()
    feat = pd.read_parquet("results/08_features.parquet")
    fcols = [c for c in feat.columns if c not in ("stkcd", "trddt", "y")]
    print(f"特征 {feat.shape[0]}行 × {len(fcols)}因子；加载 {time.time()-t0:.0f}s", flush=True)

    # (a) LightGBM walk-forward
    X = feat[fcols].values.astype("float32"); y = feat["y"].values
    dates = feat["trddt"].values
    pred = ml.walk_forward_predict(X, y, dates, ml.lgb_model(), init=WARM, step=3)
    feat["lgb"] = pred
    f_oos = feat[feat["lgb"].notna()]
    ic_lgb = rank_ic(f_oos, "lgb").dropna()
    print(f"\n(a) LightGBM walk-forward 全231: RankIC {ic_lgb.mean():.4f}  ICIR {ic_lgb.mean()/ic_lgb.std():.3f}  "
          f"t {ic_lgb.mean()/ic_lgb.std()*np.sqrt(len(ic_lgb)):.1f}  期 {len(ic_lgb)}", flush=True)

    # (b) 贪心 OOS 在全 231 里选
    IC = pd.DataFrame({c: rank_ic(feat, c) for c in fcols}).sort_index()
    print(f"231因子IC矩阵完成 {time.time()-t0:.0f}s", flush=True)
    alpha = pd.Series(np.nan, index=feat.index)
    for t in sorted(feat["trddt"].unique()):
        win = IC.loc[IC.index < t].dropna(axis=1, how="all")
        if len(win) < WARM: continue
        signs = np.sign(win.mean()); aligned = win.mul(signs, axis=1).fillna(0.0)
        cands, sel, best = list(aligned.columns), [], -np.inf
        cur = pd.Series(0.0, index=aligned.index)
        while cands and len(sel) < 12:
            scs = {c: ((cur + aligned[c]).mean() / (cur + aligned[c]).std()) for c in cands}
            c = max(scs, key=scs.get)
            if scs[c] <= best + 1e-4 and len(sel) >= 3: break
            best = scs[c]; sel.append(c); cands.remove(c); cur = cur + aligned[c]
        idx = feat.index[feat["trddt"] == t]
        feat.loc[idx, "grd"] = (sum(signs[c] * feat.loc[idx, c] for c in sel) / len(sel)).values
    ic_grd = rank_ic(feat[feat["grd"].notna()], "grd").dropna()
    print(f"(b) 贪心OOS 全231: RankIC {ic_grd.mean():.4f}  ICIR {ic_grd.mean()/ic_grd.std():.3f}  "
          f"(对比32因子版 0.64)", flush=True)

    # ---- 部署 LightGBM OOS alpha ----
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    a = feat[["stkcd", "trddt", "lgb"]].copy(); a["trddt"] = a["trddt"].astype("datetime64[ns]")
    panel = panel.merge(a, on=["stkcd", "trddt"], how="left").rename(columns={"lgb": "alpha"})
    oos = panel[panel["alpha"].notna()]
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    i500 = backtest.load_benchmark("000905", FREQ); i800 = backtest.load_benchmark("000906", FREQ)
    i1000 = backtest.load_benchmark("000852", FREQ)

    rows, prev = [], set()
    for dt, g in oos.dropna(subset=["fwd_ret"]).groupby("trddt"):
        top = g.nlargest(max(1, len(g) // 10), "alpha"); cur = set(top["stkcd"])
        to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows.append({"dt": dt, "g": top["fwd_ret"].mean(), "to": to}); prev = cur
    L = pd.DataFrame(rows).set_index("dt"); v1 = L["g"] - L["to"] * C
    d = pd.concat([v1, i1000], axis=1).dropna(); beta = float(np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0])
    v2 = (v1 - beta * i1000.reindex(v1.index) - beta * 0.03 / 12).dropna()

    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")
    con = data.connect()
    bw = con.sql(f"""SELECT CAST(Enddt AS DATE) dt, Stkcd stkcd, TRY_CAST(Weight AS DOUBLE) w
                     FROM '{data.DATA_ROOT}/raw/IDX_Smprat.parquet' WHERE Indexcd='000906'""").df()
    con.close(); bw["dt"] = bw["dt"].astype("datetime64[ns]"); bw = bw.dropna()
    snaps = np.sort(bw["dt"].unique()); erows, wprev = [], None
    for dt in sorted(panel[panel["alpha"].notna()]["trddt"].unique()):
        s = snaps[snaps <= np.datetime64(dt)]
        if not len(s): continue
        cons = bw[bw["dt"] == s[-1]][["stkcd", "w"]]
        g = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna()]
        m = g.merge(cons, on="stkcd", how="inner")
        if len(m) < 200: continue
        b = (m["w"] / m["w"].sum()).values
        Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values; dd = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, dd, active_cap=0.02, te=0.03, style_band=0.10)
        ws = pd.Series(w, index=m["stkcd"].values)
        to = 0.0 if wprev is None else 0.5 * ws.subtract(wprev, fill_value=0).abs().sum()
        erows.append({"dt": dt, "r": float(np.nansum(w * m["fwd_ret"].values)) - to * C}); wprev = ws
    v3 = pd.DataFrame(erows).set_index("dt")["r"]

    res = {"V1 多头·全A": metrics_full(v1, i500), "V2 市场中性·IM贴水3%": metrics_full(v2),
           "V3 中证800指增 TE3%": metrics_full(v3, i800.reindex(v3.index))}
    out = pd.DataFrame(res).T
    for c in ["年化", "波动", "回撤", "超额"]:
        out[c] = (out[c] * 100).round(1).astype(str) + "%"
    out["夏普"] = out["夏普"].round(2); out["卡玛"] = out["卡玛"].round(2)
    out["胜率"] = (out["胜率"] * 100).round(0).astype(int).astype(str) + "%"
    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 80, "\n全231因子 LightGBM walk-forward OOS alpha 部署（扣0.3%换手）\n", "=" * 80, sep="")
    print(out.to_string()); out.to_csv("results/26_fullpool.csv", encoding="utf-8-sig")
    pd.DataFrame({"V1多头": (1 + v1).cumprod(), "V2中性": (1 + v2).cumprod(), "V3指增": (1 + v3).cumprod(),
                  "中证500": (1 + i500.reindex(v1.index)).cumprod(), "中证800": (1 + i800.reindex(v1.index)).cumprod()
                  }).to_csv("results/26_nav.csv", encoding="utf-8-sig")
    print(f"\n完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
