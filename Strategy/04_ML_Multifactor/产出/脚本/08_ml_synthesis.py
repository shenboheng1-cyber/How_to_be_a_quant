# -*- coding: utf-8 -*-
"""
研究脚本 08 —— L3：机器学习因子合成（防泄漏）
================================================================
把 ~250 个代表性因子组装成特征矩阵，用 LightGBM 合成，全程 purged walk-forward。
对比 LightGBM / 岭回归 / IC加权 / 等权 四种合成；看特征重要性里订单流因子排第几。

用法：python research/08_ml_synthesis.py
"""
import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from quantlib import data, universe, preprocess, evaluate, microstructure, ml
from quantlib.alpha import matrices, alphas, gtja191, factory
from quantlib.factors import classic, behavioral

FREQ, START, END = "M", "2015-01-01", "2025-12-31"
FEAT_CACHE = "results/08_features.parquet"


def assemble():
    if os.path.exists(FEAT_CACHE):
        print("载入特征缓存 ...")
        df = pd.read_parquet(FEAT_CACHE)
        feat_cols = [c for c in df.columns if c not in ("stkcd", "trddt", "y")]
        return df, feat_cols

    print("组装特征矩阵（首次较慢）...")
    catalog = json.load(open("results/micro_catalog.json"))["factors"]
    panel = data.load_research_panel(FREQ, START, END)
    micro = microstructure.load_specs(catalog, FREQ, START, END)
    panel = panel.merge(micro, on=["stkcd", "trddt"], how="left")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    M = matrices.load_matrices(START, END)

    raw = {}
    for nm, (fn, _) in classic.REGISTRY.items():     raw["cls_" + nm] = fn(panel)
    for nm, (fn, _) in behavioral.REGISTRY.items():  raw["beh_" + nm] = fn(panel)
    for nm, fn in alphas.CURATED.items():            raw["hw_" + nm] = factory.sample_to_panel(fn(M), panel)
    for nm, fn in alphas.generate(M).items():        raw["fac_" + nm] = factory.sample_to_panel(fn(M), panel)
    # GTJA：取上次 ICIR 最强的 40 个
    try:
        g = pd.read_csv("results/04_gtja191_summary.csv")
        top_gtja = g.reindex(g["ICIR"].abs().sort_values(ascending=False).index)["因子"].head(40).tolist()
    except Exception:
        top_gtja = [f"gtja{i}" for i in [95, 62, 99, 70, 16, 105, 42, 140, 83, 90]]
    greg = gtja191.build_registry(M)
    for nm in top_gtja:
        if nm in greg: raw["gtja_" + nm] = factory.sample_to_panel(greg[nm](M), panel)
    # 微观结构（带 sign）
    for s in catalog:
        if s["name"] in panel.columns: raw["mic_" + s["name"]] = panel[s["name"]] * s["sign"]

    print(f"  原始特征 {len(raw)} 个，预处理中 ...")
    out = pd.DataFrame({"stkcd": panel["stkcd"].values, "trddt": panel["trddt"].values})
    out["y"] = ml.make_label(panel)
    for k, v in raw.items():
        out[k] = preprocess.preprocess_factor(panel, v, do_neutralize=True).values
    os.makedirs("results", exist_ok=True)
    out.to_parquet(FEAT_CACHE)
    feat_cols = list(raw.keys())
    return out, feat_cols


def eval_signal(sig, panel_oos):
    f = pd.Series(sig)
    ic = evaluate.compute_ic(panel_oos, f)
    s = evaluate.ic_summary(ic)
    qs = evaluate.quantile_summary(evaluate.quantile_returns(panel_oos, f, 10))
    ls = qs.loc["多空(QN-Q1)"]
    nav = evaluate.long_short_nav(evaluate.quantile_returns(panel_oos, f, 10))
    return s, ls, nav


def main():
    t0 = time.time()
    df, feat = assemble()
    X = df[feat].values
    y = df["y"].values
    dates = df["trddt"].values
    print(f"特征矩阵 {X.shape} | {time.time()-t0:.0f}s\n")

    print("walk-forward 预测（LightGBM / 岭回归）...")
    pred_lgb = ml.walk_forward_predict(X, y, dates, ml.lgb_model(), init=36, embargo=1, step=3)
    pred_rdg = ml.walk_forward_predict(X, y, dates, ml.ridge_model(20.0), init=36, embargo=1, step=3)
    sig_eq = ml.equal_weight_signal(X)
    sig_ic = ml.ic_weight_signal(X, y, dates, init=36)

    oos = ~np.isnan(pred_lgb)
    # panel 仅含评估所需列（fwd_ret/trddt/total_mktcap）— 从缓存重建一个最小 panel
    base = data.load_research_panel(FREQ, START, END)
    base = base.merge(df[["stkcd", "trddt"]], on=["stkcd", "trddt"], how="right")  # 对齐行序
    base = base.reset_index(drop=True)
    panel_oos = base[oos].reset_index(drop=True)

    methods = {"LightGBM": pred_lgb, "岭回归(线性)": pred_rdg, "IC加权": sig_ic, "等权": sig_eq}
    rows, navs = [], {}
    for name, sig in methods.items():
        s, ls, nav = eval_signal(sig[oos], panel_oos)
        rows.append({"合成方法": name, "RankIC": s["IC均值"], "ICIR": s["ICIR"],
                     "t值": s["t值"], "多空年化": round(ls["年化收益"], 4), "多空夏普": ls["夏普"]})
        navs[name] = nav
    tbl = pd.DataFrame(rows)
    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 70, "\nL3 因子合成对比（样本外，", oos.sum(), "行）\n", "=" * 70, sep="")
    print(tbl.to_string(index=False))

    # 特征重要性（描述性：在全部可用标签上训一个 LGB 取 gain）
    print("\n特征重要性(LightGBM gain) Top 20：")
    import lightgbm as lgb
    m = ~np.isnan(y)
    gbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                            min_child_samples=200, subsample=0.8, colsample_bytree=0.7,
                            reg_lambda=5.0, importance_type="gain", n_jobs=-1, verbose=-1)
    gbm.fit(X[m], y[m])
    imp = pd.Series(gbm.feature_importances_, index=feat).sort_values(ascending=False)
    top = imp.head(20)
    for k, v in top.items():
        tag = "★微观" if k.startswith("mic_") else ""
        print(f"  {k:<28} {v:>10.0f}  {tag}")
    micro_in_top = [k for k in imp.head(30).index if k.startswith("mic_")]
    print(f"\n★ 微观结构因子进前30的: {micro_in_top}")

    # 图：多空净值 + 重要性
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for nm, nav in navs.items():
        a1.plot(nav.index, nav.values, label=nm, lw=1.5)
    a1.set_title("OOS long-short NAV by synthesis method"); a1.legend(); a1.set_yscale("log"); a1.grid(alpha=.3)
    colors = ["#C44E52" if k.startswith("mic_") else "#4C72B0" for k in top.index[::-1]]
    a2.barh(range(len(top)), top.values[::-1], color=colors)
    a2.set_yticks(range(len(top))); a2.set_yticklabels(top.index[::-1], fontsize=7)
    a2.set_title("Feature importance (red = microstructure)"); a2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("results/08_ml_synthesis.png", bbox_inches="tight")
    print("\n已保存 results/08_ml_synthesis.png 与特征缓存")
    tbl.to_csv("results/08_ml_synthesis_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
