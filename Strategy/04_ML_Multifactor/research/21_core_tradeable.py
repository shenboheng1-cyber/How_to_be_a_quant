# -*- coding: utf-8 -*-
"""
研究脚本 21 —— 10因子正交核 在可交易组合里的表现
================================================================
用 research/20 选出的 10 因子正交核当 alpha，重跑：
  A) 市场中性（IC/IM 期货对冲，扣贴水）—— 对比 231-LGB 的夏普 1.55/1.29
  B) 中证800 指数增强（行业/风格中性 + TE预算）—— 对比 231 价量composite 的 IR 0.34
假设：正交核基本面/事件多 → 在大盘更 work → 指增 IR 能升。

用法：/opt/anaconda3/bin/python research/21_core_tradeable.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, evaluate, backtest,
                      fundamentals, events, altdata, riskmodel, optimizer)
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
C = 0.003                                                   # 单边换手成本
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]
BENCH = "000906"                                            # 中证800
CORE = [("max_ret", classic.REGISTRY["max_ret"][0]),
        ("lockup", events.REGISTRY["e_lockup"][0]),
        ("research", altdata.REGISTRY["a_research"][0]),
        ("size", classic.REGISTRY["size"][0]),
        ("viol", events.REGISTRY["e_viol"][0]),
        ("accruals", fundamentals.REGISTRY["f_accruals"][0]),
        ("profit_growth", fundamentals.REGISTRY["f_profit_growth"][0]),
        ("earn_mgmt", events.REGISTRY["e_earn_mgmt"][0]),
        ("reversal", classic.REGISTRY["reversal"][0]),
        ("illiquidity", classic.REGISTRY["illiquidity"][0])]


def long_stream(panel, alpha_col):
    """多头 top-decile 月收益 + 单边换手。"""
    rows, prev = [], set()
    for dt, g in panel.dropna(subset=[alpha_col, "fwd_ret"]).groupby("trddt"):
        k = max(1, len(g) // 10)
        top = g.nlargest(k, alpha_col)
        cur = set(top["stkcd"])
        to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows.append({"dt": dt, "long_g": top["fwd_ret"].mean(), "to_l": to})
        prev = cur
    return pd.DataFrame(rows).set_index("dt")


def load_bench_weights():
    con = data.connect()
    df = con.sql(f"""SELECT CAST(Enddt AS DATE) AS dt, Stkcd AS stkcd, TRY_CAST(Weight AS DOUBLE) AS w
                     FROM '{data.DATA_ROOT}/raw/IDX_Smprat.parquet' WHERE Indexcd='{BENCH}'""").df()
    con.close()
    df["dt"] = df["dt"].astype("datetime64[ns]")
    return df.dropna()


def main():
    t0 = time.time()
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel = fundamentals.attach(panel)
    panel = events.attach_events(panel)
    panel = altdata.attach_altfactors(panel)

    # 10因子正交核 alpha（符号对齐后等权）
    Z = []
    for name, fn in CORE:
        z = preprocess.preprocess_factor(panel, fn(panel), industry_col="industry", do_neutralize=True)
        z = z * np.sign(evaluate.compute_ic(panel, z).mean())
        Z.append(pd.Series(z.values, name=name))
    panel["alpha"] = pd.concat(Z, axis=1).mean(axis=1).values
    # 风格暴露
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    print(f"正交核 alpha 就绪 {time.time()-t0:.0f}s", flush=True)
    pd.set_option("display.unicode.east_asian_width", True)

    # ============ A) 市场中性 ============
    L = long_stream(panel, "alpha")
    long_net = L["long_g"] - L["to_l"] * C
    i500 = backtest.load_benchmark("000905", FREQ).reindex(L.index)
    i1000 = backtest.load_benchmark("000852", FREQ).reindex(L.index)

    def beta(idx):
        d = pd.concat([long_net, idx], axis=1).dropna()
        return float(np.polyfit(d.iloc[:, 1], d.iloc[:, 0], 1)[0])

    rows = {}
    for name, idx in [("IC对冲(中证500)", i500), ("IM对冲(中证1000)", i1000)]:
        b = beta(idx)
        for basis in [0.0, 0.03, 0.06]:
            neu = long_net - b * idx - b * basis / 12
            m = backtest.metrics(neu, FREQ)
            rows[f"{name} 贴水{basis:.0%}"] = {"beta": round(b, 2), "净年化": m["年化"],
                                              "净夏普": m["夏普"], "最大回撤": m["最大回撤"]}
    print("\n" + "=" * 66, "\nA) 市场中性 — 正交核 alpha（对比231-LGB: IM贴水0%夏普1.55/贴水6%夏普1.03）\n", "=" * 66, sep="")
    print(pd.DataFrame(rows).T.to_string())
    mlong = backtest.metrics(long_net, FREQ)
    print(f"  [多头扛市场] 年化{mlong['年化']:.1%} 夏普{mlong['夏普']:.2f} 回撤{mlong['最大回撤']:.0%}")

    # ============ B) 中证800 指数增强 ============
    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid)
    sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")
    bw = load_bench_weights()
    snaps = np.sort(bw["dt"].unique())
    dates = sorted(panel["trddt"].unique())
    TES = [0.03, 0.05, 0.08]
    erows = []
    for dt in dates[24:]:
        snap = snaps[snaps <= np.datetime64(dt)]
        if not len(snap):
            continue
        cons = bw[bw["dt"] == snap[-1]][["stkcd", "w"]]
        g = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna()]
        m = g.merge(cons, on="stkcd", how="inner")
        if len(m) < 200:
            continue
        b = (m["w"] / m["w"].sum()).values
        Xind = pd.get_dummies(m["industry"]).astype(float)
        cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values
        rec = {"dt": dt, "bench": float(np.nansum(b * fwd))}
        for te in TES:
            w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, d,
                                            active_cap=0.02, te=te, style_band=0.10)
            rec[f"opt{te}"] = float(np.nansum(w * fwd))
        nv = m.nlargest(max(1, len(m) // 3), "alpha")
        rec["naive"] = nv["fwd_ret"].mean()
        erows.append(rec)
    R = pd.DataFrame(erows).set_index("dt")

    def rep(col):
        ex = (R[col] - R["bench"]).dropna()
        nav = (1 + ex).cumprod()
        return {"年化超额": round(ex.mean() * 12, 4), "跟踪误差": round(ex.std(ddof=1) * np.sqrt(12), 3),
                "信息比IR": round(ex.mean() / ex.std(ddof=1) * np.sqrt(12), 2),
                "超额回撤": round((nav / nav.cummax() - 1).min(), 3)}

    cols = {"naive倾斜(成分内top30%)": rep("naive")}
    for te in TES:
        cols[f"优化器 TE={te:.0%}"] = rep(f"opt{te}")
    print("\n" + "=" * 66, "\nB) 中证800 指数增强 — 正交核 alpha（对比231价量composite最佳 IR 0.34）\n", "=" * 66, sep="")
    print(pd.DataFrame(cols).T.to_string())
    pd.DataFrame(cols).T.to_csv("results/21_core_enhanced.csv", encoding="utf-8-sig")
    pd.DataFrame(rows).T.to_csv("results/21_core_neutral.csv", encoding="utf-8-sig")
    print(f"\n完成 {time.time()-t0:.0f}s，已保存")


if __name__ == "__main__":
    main()
