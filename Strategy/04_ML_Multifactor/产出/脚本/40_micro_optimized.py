# -*- coding: utf-8 -*-
"""
研究脚本 40 —— 微盘多头 + 分行业中性/CDaR 优化(抗过拟合)
================================================================
基座 = LGB-231 多头 top-decile(天然微盘)。4 配置对比:
  base / +分行业中性 / +CDaR减震 / +两者。
抗过拟合:CDaR floor 固定 15%/年(不 in-sample 挑最优),额外给:
  ① 剔除组合最佳2个月后的卡玛(看是否压在少数月);
  ② 前后半样本各自卡玛(时间稳健性)。
所有配置同 CDaR活跃期(2020+)公平比,并诚实标注该期不含2018熊市。

用法：/opt/anaconda3/bin/python research/40_micro_optimized.py
"""
import sys, os, time, importlib.util, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import data, universe, altdata
spec = importlib.util.spec_from_file_location("cdar", os.path.join(os.path.dirname(__file__), "35_cdar_cvar_alloc.py"))
cdar = importlib.util.module_from_spec(spec); spec.loader.exec_module(cdar)
C, PPY, FLOOR = cdar.C, cdar.PPY, 0.15


def ind_neutral_long(panel, col="lgb"):
    rows, prev = {}, set()
    for dt, x in panel.dropna(subset=[col, "fwd_ret", "industry"]).groupby("trddt"):
        sel = x.groupby("industry", group_keys=False).apply(
            lambda d: d.nlargest(max(1, round(len(d) * 0.1)), col), include_groups=False)
        cur = set(sel["stkcd"]); to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows[dt] = sel["fwd_ret"].mean() - to * C; prev = cur
    return pd.Series(rows).sort_index()


def M(r):
    r = pd.Series(r).dropna(); n = len(r)
    ann = (1 + r).prod() ** (PPY / n) - 1; vol = r.std(ddof=1) * np.sqrt(PPY)
    nav = (1 + r).cumprod(); mdd = (nav / nav.cummax() - 1).min()
    return ann, vol, ann / vol, mdd, ann / abs(mdd)


def cal(r):
    r = pd.Series(r).dropna(); nav = (1 + r).cumprod(); mdd = (nav / nav.cummax() - 1).min()
    ann = (1 + r).prod() ** (PPY / len(r)) - 1
    return ann / abs(mdd) if mdd else np.nan


def main():
    t0 = time.time()
    panel = data.load_research_panel("M", "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel); panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    pred = pd.read_parquet("results/lgb_oos_pred.parquet"); pred["trddt"] = pred["trddt"].astype("datetime64[ns]")
    panel = panel.merge(pred, on=["stkcd", "trddt"], how="left")
    print(f"面板就绪 {time.time()-t0:.0f}s", flush=True)

    base = cdar.long_decile(panel, "lgb", largest=True)            # 基座微盘多头
    indn = ind_neutral_long(panel, "lgb")                          # 分行业中性
    lowvol = cdar.long_decile(panel[panel["lgb"].notna()], "vol_60", largest=False)
    idx = base.index

    def cdar_on(main_leg):                                         # 3腿 min-CDaR(floor), 滚动walk-forward
        legs = pd.DataFrame({"主": main_leg.reindex(idx), "低波": lowvol.reindex(idx), "现金": 0.0}).dropna()
        port, _ = cdar.rolling_alloc(legs, cdar.solve_cdar_exact, window=36, min_train=24,
                                     mode="min_risk", ret_floor=FLOOR / PPY, alpha=0.05)
        return port

    streams = {"① base 微盘多头": base, "② +分行业中性": indn,
               "③ +CDaR减震": cdar_on(base), "④ +行业中性+CDaR": cdar_on(indn)}
    active = streams["③ +CDaR减震"].index                          # 同CDaR活跃期(2020+)
    mid = active[len(active) // 2]

    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n评估期(同CDaR活跃期): {pd.Timestamp(active[0]).date()} ~ {pd.Timestamp(active[-1]).date()}  ⚠️不含2018熊市,偏乐观")
    print(f"{'配置':22s}{'年化':>7}{'波动':>7}{'夏普':>6}{'回撤':>8}{'卡玛':>6}{'剔最佳2月卡玛':>13}{'前半/后半卡玛':>16}")
    for nm, s in streams.items():
        r = pd.Series(s).reindex(active).dropna()
        a, v, sh, m, c = M(r)
        drop2 = cal(r.drop(r.nlargest(2).index))
        c1 = cal(r[r.index <= mid]); c2 = cal(r[r.index > mid])
        print(f"{nm:22s}{a*100:6.1f}%{v*100:6.1f}%{sh:6.2f}{m*100:7.1f}%{c:6.2f}{drop2:>13.2f}{c1:>8.2f}/{c2:<7.2f}")
    print(f"\nCDaR floor 敏感性(④行业中性+CDaR, 看是否knife-edge):")
    for fl in [0.12, 0.15, 0.18]:
        legs = pd.DataFrame({"主": indn.reindex(idx), "低波": lowvol.reindex(idx), "现金": 0.0}).dropna()
        port, _ = cdar.rolling_alloc(legs, cdar.solve_cdar_exact, window=36, min_train=24, mode="min_risk", ret_floor=fl / PPY, alpha=0.05)
        r = pd.Series(port).reindex(active).dropna(); a, v, sh, m, c = M(r)
        print(f"  floor={int(fl*100)}%: 年化{a*100:.1f}% 回撤{m*100:.1f}% 卡玛{c:.2f}")
    pd.DataFrame({k: (1 + pd.Series(v).reindex(active)).cumprod() for k, v in streams.items()}).to_csv("results/40_nav.csv", encoding="utf-8-sig")
    print(f"\n净值存 results/40_nav.csv；完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
