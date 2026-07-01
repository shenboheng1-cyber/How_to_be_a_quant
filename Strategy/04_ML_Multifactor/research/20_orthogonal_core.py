# -*- coding: utf-8 -*-
"""
研究脚本 20 —— 正交因子精选：几个因子就够了？
================================================================
问题：231 个因子里真正有用的有几个？
方法：贪心前向选择——每步加入"能最大提升复合 ICIR"的因子（自动避开冗余）。
看 ICIR 随因子数的饱和曲线，找到最小正交核，并与 231-LGB 对比。

用法：/opt/anaconda3/envs/csmar/bin/python research/20_orthogonal_core.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from quantlib import data, universe, preprocess, evaluate, fundamentals, events, altdata
from quantlib.factors import classic

FREQ, START, END = "M", "2015-01-01", "2025-12-31"


def build_pool(panel):
    """候选池：可解释的命名因子（价量+基本面+事件+另类）。"""
    pool = {}
    for name, (fn, _) in classic.REGISTRY.items():
        pool[name] = fn
    for mod, pre in [(fundamentals, "f_"), (events, "e_"), (altdata, "a")]:
        for name, (fn, _) in getattr(mod, "REGISTRY", {}).items():
            pool[pre + name if pre != "a" else name] = fn
    try:
        from quantlib.factors import behavioral
        for name, (fn, _) in behavioral.REGISTRY.items():
            pool["b_" + name] = fn
    except Exception:
        pass
    return pool


def main():
    t0 = time.time()
    panel = data.load_research_panel(FREQ, START, END)
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel = fundamentals.attach(panel)
    panel = events.attach_events(panel)
    panel = altdata.attach_altfactors(panel)
    pool = build_pool(panel)
    print(f"候选因子 {len(pool)} 个，预处理中 ... {time.time()-t0:.0f}s", flush=True)

    # 预处理 + 按标准IC符号对齐（统一成正向），存成 Z 矩阵
    Z, sIC = {}, {}
    for name, fn in pool.items():
        try:
            raw = fn(panel)
            z = preprocess.preprocess_factor(panel, raw, industry_col="industry", do_neutralize=True)
            ic = evaluate.compute_ic(panel, z).mean()
            if np.isnan(ic) or z.notna().mean() < 0.3:
                continue
            sign = np.sign(ic)
            Z[name] = (z * sign).values
            sIC[name] = abs(ic)
        except Exception as e:
            print("  跳过", name, str(e)[:40])
    Z = pd.DataFrame(Z, index=panel.index)
    print(f"有效因子 {Z.shape[1]} 个。各因子标准|IC|前10:", flush=True)
    print("  " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(sIC.items(), key=lambda x: -x[1])[:10]))

    def comp_icir(cols):
        comp = Z[cols].mean(axis=1)
        return evaluate.ic_summary(evaluate.compute_ic(panel, comp))["ICIR"]

    # 贪心前向选择
    print("\n贪心前向选择（每步加入最大化复合 ICIR 的因子）：", flush=True)
    remaining = list(Z.columns)
    selected, curve = [], []
    while remaining:
        best = max(remaining, key=lambda c: comp_icir(selected + [c]))
        ic = comp_icir(selected + [best])
        selected.append(best); remaining.remove(best)
        curve.append((len(selected), best, ic))
        print(f"  +{len(selected):2d} {best:16s} 复合ICIR={ic:.3f}", flush=True)

    peak = max(curve, key=lambda x: x[2])
    K = peak[0]
    core = [c[1] for c in curve[:K]]
    all_icir = comp_icir(list(Z.columns))
    print(f"\n>>> ICIR 峰值 {peak[2]:.3f} 在第 {K} 个因子；全部{Z.shape[1]}个等权 ICIR={all_icir:.3f}")
    print(f">>> 正交核（{K}个）: {core}")

    # 核内相关性
    cc = pd.DataFrame(Z[core]).corr()
    print(f"\n正交核两两相关 平均|corr|={cc.where(~np.eye(K, dtype=bool)).abs().stack().mean():.3f} (越低越正交)")

    # 性能对比：正交核 vs 231-LGB（08结果）
    comp = pd.Series(Z[core].mean(axis=1), index=panel.index)
    icr = evaluate.ic_summary(evaluate.compute_ic(panel, comp))
    qs = evaluate.quantile_summary(evaluate.quantile_returns(panel, comp, 10))
    q10 = qs.loc["Q10"]; ls = qs.loc["多空(QN-Q1)"]
    print("\n" + "=" * 60)
    pd.set_option("display.unicode.east_asian_width", True)
    out = pd.DataFrame({
        f"正交核({K}个,等权)": {"RankIC": icr["IC均值"], "ICIR": icr["ICIR"], "t值": icr["t值"],
                            "多头年化": round(q10["年化收益"], 4), "多空夏普": ls["夏普"], "多空回撤": ls["最大回撤"]},
        "231-LGB(全因子)": {"RankIC": 0.1063, "ICIR": 1.033, "t值": 10.07,
                          "多头年化": 0.1975, "多空夏普": 2.95, "多空回撤": -0.103},
    }).T
    print(out.to_string())
    pd.DataFrame(curve, columns=["n", "因子", "复合ICIR"]).to_csv("results/20_orthogonal_core.csv", index=False, encoding="utf-8-sig")
    print(f"\n完成 {time.time()-t0:.0f}s，曲线已存 results/20_orthogonal_core.csv")


if __name__ == "__main__":
    main()
