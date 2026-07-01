# -*- coding: utf-8 -*-
"""
研究脚本 27 —— 任务1：分散 alpha + 拥挤/波动降仓，砍 alpha 自身回撤
================================================================
诊断:多头 −25.6% 回撤里 ~2/3 是 alpha 自身(动量小盘踩踏),不是 beta。
对策:① 把 LGB(动量小盘) 和 质量/价值(防御,2024扛得住) 在个股层面分散合成;
      ② 波动目标降仓(高波时减仓)。看回撤/卡玛改善。
顺带把 LGB OOS 预测存盘(results/lgb_oos_pred.parquet)供任务2复用。

用法：/opt/anaconda3/bin/python research/27_diversify_derisk.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, evaluate, backtest, ml,
                      fundamentals, altdata, regime)

FREQ, C, WARM, PPY = "M", 0.003, 36, 12
QUAL = ["f_gross_prof", "f_low_lev", "f_accruals", "f_ep", "f_bp", "f_roe"]   # 价值+质量 防御腿


def zx(panel, col):
    return panel.groupby("trddt")[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))


def long_stream(panel, col):
    rows, prev = [], set()
    for dt, g in panel.dropna(subset=[col, "fwd_ret"]).groupby("trddt"):
        top = g.nlargest(max(1, len(g) // 10), col); cur = set(top["stkcd"])
        to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows.append({"dt": dt, "g": top["fwd_ret"].mean(), "to": to}); prev = cur
    L = pd.DataFrame(rows).set_index("dt")
    return L["g"] - L["to"] * C


def mfull(r):
    r = r.dropna(); n = len(r); ann = (1 + r).prod() ** (PPY / n) - 1
    vol = r.std(ddof=1) * np.sqrt(PPY); sh = r.mean() / r.std(ddof=1) * np.sqrt(PPY)
    nav = (1 + r).cumprod(); mdd = (nav / nav.cummax() - 1).min()
    return {"年化": ann, "波动": vol, "夏普": sh, "最大回撤": mdd, "卡玛": ann / abs(mdd), "胜率": (r > 0).mean()}


def main():
    t0 = time.time()
    # 1) LGB walk-forward 预测(并存盘)
    feat = pd.read_parquet("results/08_features.parquet")
    fcols = [c for c in feat.columns if c not in ("stkcd", "trddt", "y")]
    feat["lgb"] = ml.walk_forward_predict(feat[fcols].values.astype("float32"), feat["y"].values,
                                          feat["trddt"].values, ml.lgb_model(), init=WARM, step=3)
    feat[["stkcd", "trddt", "lgb"]].dropna().to_parquet("results/lgb_oos_pred.parquet")
    print(f"LGB预测完成+存盘 {time.time()-t0:.0f}s", flush=True)

    # 2) 面板 + 质量腿
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel); panel = fundamentals.attach(panel)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    a = feat[["stkcd", "trddt", "lgb"]].copy(); a["trddt"] = a["trddt"].astype("datetime64[ns]")
    panel = panel.merge(a, on=["stkcd", "trddt"], how="left")
    Q = []
    for k in QUAL:
        z = preprocess.preprocess_factor(panel, fundamentals.REGISTRY[k][0](panel), industry_col="industry", do_neutralize=True)
        Q.append((z * np.sign(evaluate.compute_ic(panel, z).mean())).values)
    panel["qual"] = np.nanmean(np.column_stack(Q), axis=1)
    panel = panel[panel["lgb"].notna()].copy()
    panel["z_lgb"] = zx(panel, "lgb"); panel["z_qual"] = zx(panel, "qual")
    panel["blend"] = 0.5 * panel["z_lgb"] + 0.5 * panel["z_qual"].fillna(0)

    # 3) 三条腿 + 混合
    r_lgb = long_stream(panel, "z_lgb"); r_qual = long_stream(panel, "z_qual"); r_bl = long_stream(panel, "blend")
    corr = pd.concat([r_lgb, r_qual], axis=1).dropna().corr().iloc[0, 1]

    # 4) 波动目标降仓(只减不加,cap=1.0)叠加在混合上
    scale = regime.vol_target(r_bl, target_ann=0.15, lookback=6, cap=1.0)
    r_bl_vt = (scale * r_bl).dropna()

    pd.set_option("display.unicode.east_asian_width", True)
    res = {"① LGB多头(动量小盘)": mfull(r_lgb), "② 质量价值多头(防御)": mfull(r_qual),
           "③ 50/50 分散混合": mfull(r_bl), "④ 混合+波动目标降仓": mfull(r_bl_vt)}
    out = pd.DataFrame(res).T
    for c in ["年化", "波动", "最大回撤"]:
        out[c] = (out[c] * 100).round(1).astype(str) + "%"
    out["夏普"] = out["夏普"].round(2); out["卡玛"] = out["卡玛"].round(2)
    out["胜率"] = (out["胜率"] * 100).round(0).astype(int).astype(str) + "%"
    print("\n" + "=" * 78, "\n任务1：分散 + 降仓（扣0.3%换手；目标砍 alpha 自身回撤）\n", "=" * 78, sep="")
    print(out.to_string())
    print(f"\nLGB 与 质量价值 月收益相关 = {corr:.2f}（越低越分散）")
    # 2024 踩踏期对比
    crash = pd.concat([r_lgb.rename("LGB"), r_qual.rename("质量"), r_bl.rename("混合")], axis=1).loc["2023-12":"2024-02"]
    print("\n2023-12~2024-02 小盘踩踏期月收益:"); print((crash * 100).round(1).to_string())
    out.to_csv("results/27_diversify.csv", encoding="utf-8-sig")
    pd.DataFrame({"LGB": (1 + r_lgb).cumprod(), "质量": (1 + r_qual).cumprod(),
                  "混合": (1 + r_bl).cumprod(), "混合+降仓": (1 + r_bl_vt).cumprod()}).to_csv("results/27_nav.csv", encoding="utf-8-sig")
    print(f"\n完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
