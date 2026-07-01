# -*- coding: utf-8 -*-
"""
研究脚本 22 —— 最佳几版策略的完整指标 + 净值
================================================================
对比 4 版（均扣换手成本 C=0.3%/单边换手）：
  V1 多头·全A(正交核)            —— 激进、小盘、扛beta
  V2 市场中性·IM对冲 贴水3%(正交核) —— 最佳风险调整
  V3 中证800指增 TE=3%(正交核)     —— 可规模化机构版
  V4 多头·全A(231-LGB,旧旗舰)      —— 参照
指标：年化/波动/夏普/最大回撤/卡玛/超额/胜率。存净值到 results/22_nav.csv。

用法：/opt/anaconda3/bin/python research/22_best_versions.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, evaluate, backtest,
                      fundamentals, events, altdata, riskmodel, optimizer)
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
C = 0.003
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]
CORE = [("max_ret", classic.REGISTRY["max_ret"][0]), ("lockup", events.REGISTRY["e_lockup"][0]),
        ("research", altdata.REGISTRY["a_research"][0]), ("size", classic.REGISTRY["size"][0]),
        ("viol", events.REGISTRY["e_viol"][0]), ("accruals", fundamentals.REGISTRY["f_accruals"][0]),
        ("profit_growth", fundamentals.REGISTRY["f_profit_growth"][0]),
        ("earn_mgmt", events.REGISTRY["e_earn_mgmt"][0]), ("reversal", classic.REGISTRY["reversal"][0]),
        ("illiquidity", classic.REGISTRY["illiquidity"][0])]
PPY = 12


def metrics_full(r, bench=None):
    r = r.dropna()
    n = len(r)
    ann = (1 + r).prod() ** (PPY / n) - 1
    vol = r.std(ddof=1) * np.sqrt(PPY)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(PPY)
    nav = (1 + r).cumprod()
    mdd = (nav / nav.cummax() - 1).min()
    calmar = ann / abs(mdd) if mdd else np.nan
    if bench is not None:
        b = bench.reindex(r.index)
        ex = (r - b).dropna()
        exann = (1 + r).prod() ** (PPY / n) - (1 + b.reindex(r.index)).prod() ** (PPY / n)
        win = (ex > 0).mean()
    else:
        exann = ann; win = (r > 0).mean()
    return {"年化收益": ann, "年化波动": vol, "夏普": sharpe, "最大回撤": mdd,
            "卡玛": calmar, "超额": exann, "胜率": win}


def long_stream(panel, col):
    rows, prev = [], set()
    for dt, g in panel.dropna(subset=[col, "fwd_ret"]).groupby("trddt"):
        top = g.nlargest(max(1, len(g) // 10), col)
        cur = set(top["stkcd"])
        to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows.append({"dt": dt, "g": top["fwd_ret"].mean(), "to": to}); prev = cur
    return pd.DataFrame(rows).set_index("dt")


def main():
    t0 = time.time()
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel = fundamentals.attach(panel); panel = events.attach_events(panel); panel = altdata.attach_altfactors(panel)
    Z = []
    for name, fn in CORE:
        z = preprocess.preprocess_factor(panel, fn(panel), industry_col="industry", do_neutralize=True)
        Z.append(pd.Series((z * np.sign(evaluate.compute_ic(panel, z).mean())).values, name=name))
    panel["alpha"] = pd.concat(Z, axis=1).mean(axis=1).values
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    i500 = backtest.load_benchmark("000905", FREQ); i800 = backtest.load_benchmark("000906", FREQ)
    i1000 = backtest.load_benchmark("000852", FREQ)
    print(f"alpha 就绪 {time.time()-t0:.0f}s", flush=True)

    nav = {}
    # V1 多头·全A(核心)
    L = long_stream(panel, "alpha"); v1 = L["g"] - L["to"] * C
    # V2 市场中性 IM 贴水3%
    d = pd.concat([v1, i1000], axis=1).dropna(); beta = float(np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0])
    v2 = (v1 - beta * i1000.reindex(v1.index) - beta * 0.03 / 12).dropna()
    # V4 231-LGB 多头(读09)
    bt = pd.read_csv("results/09_backtest.csv", index_col=0, parse_dates=True)
    v4 = bt["long_g"] - bt["to_l"] * C

    # V3 中证800指增 TE=3%(核心) —— 重跑优化器循环取净值
    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")
    con = data.connect()
    bw = con.sql(f"""SELECT CAST(Enddt AS DATE) dt, Stkcd stkcd, TRY_CAST(Weight AS DOUBLE) w
                     FROM '{data.DATA_ROOT}/raw/IDX_Smprat.parquet' WHERE Indexcd='000906'""").df()
    con.close(); bw["dt"] = bw["dt"].astype("datetime64[ns]"); bw = bw.dropna()
    snaps = np.sort(bw["dt"].unique()); dates = sorted(panel["trddt"].unique())
    rows, wprev = [], None
    for dt in dates[24:]:
        snap = snaps[snaps <= np.datetime64(dt)]
        if not len(snap): continue
        cons = bw[bw["dt"] == snap[-1]][["stkcd", "w"]]
        g = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna()]
        m = g.merge(cons, on="stkcd", how="inner")
        if len(m) < 200: continue
        b = (m["w"] / m["w"].sum()).values
        Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        dd = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        Xs = m[style_cols].fillna(0.0).values
        w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, dd,
                                        active_cap=0.02, te=0.03, style_band=0.10)
        ws = pd.Series(w, index=m["stkcd"].values)
        to = 0.0 if wprev is None else 0.5 * ws.subtract(wprev, fill_value=0).abs().sum()
        rows.append({"dt": dt, "r": float(np.nansum(w * m["fwd_ret"].values)) - to * C}); wprev = ws
    E = pd.DataFrame(rows).set_index("dt"); v3 = E["r"]

    res = {
        "V1 多头·全A(正交核)": metrics_full(v1, i500),
        "V2 市场中性·IM对冲贴水3%": metrics_full(v2),
        "V3 中证800指增 TE=3%": metrics_full(v3, i800.reindex(v3.index)),
        "V4 多头·全A(231-LGB,旧)": metrics_full(v4, i500),
    }
    out = pd.DataFrame(res).T
    for c in ["年化收益", "年化波动", "最大回撤", "超额"]:
        out[c] = (out[c] * 100).round(1).astype(str) + "%"
    out["夏普"] = out["夏普"].round(2); out["卡玛"] = out["卡玛"].round(2)
    out["胜率"] = (out["胜率"] * 100).round(0).astype(int).astype(str) + "%"
    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 84, "\n最佳几版策略 完整指标（均扣 0.3% 单边换手成本；样本外）\n", "=" * 84, sep="")
    print(out.to_string())
    out.to_csv("results/22_best_versions.csv", encoding="utf-8-sig")

    # 净值（绝对收益口径）供画图
    NAV = pd.DataFrame({
        "V1多头全A": (1 + v1).cumprod(), "V2市场中性": (1 + v2).cumprod(),
        "V3中证800指增": (1 + v3).cumprod(), "V4多头231": (1 + v4).cumprod(),
        "中证500": (1 + i500.reindex(v1.index)).cumprod(), "中证800": (1 + i800.reindex(v1.index)).cumprod(),
    })
    NAV.to_csv("results/22_nav.csv", encoding="utf-8-sig")
    print(f"\n净值已存 results/22_nav.csv；完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
