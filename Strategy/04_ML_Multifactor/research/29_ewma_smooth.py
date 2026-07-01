# -*- coding: utf-8 -*-
"""
研究脚本 29_ewma_smooth —— 杠杆=【alpha 时序平滑】
================================================================
在优化前，对每只股票的 lgb 信号做时序 EWMA（按 stkcd 分组，halflife 试 2/3 个月，
用 ewm(...).mean() 再 .shift(0)？ 不——为了"只用当期及过去、无前视"：
  EWMA 本身只用历史与当期；ewm().mean() 在 t 期的值只依赖 <=t 的数据，无前视。
  为额外保险（避免任何同期泄漏）我们采用：先按 stkcd 排序时间，EWMA 用 adjust=True，
  得到的平滑值仅由当期及之前的 lgb 构成 → 直接当 t 期 alpha 用（与原始 lgb 对齐方式相同，
  因为原始 lgb 本身就是 t 期信号 → t→t+1 收益）。
对比 平滑(hl=2/3) vs 不平滑 的 超额/IR/换手，报告更优 halflife。

用法：/opt/anaconda3/bin/python research/29_ewma_smooth.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, backtest, fundamentals, altdata,
                      riskmodel, optimizer)
from quantlib.factors import classic

# ---- 降低 duckdb 并发压力（与并行实验共享磁盘/temp，避免 OOM 溢写）----
# 仅包裹本脚本内的 data.connect，不改动 quantlib 下任何共享文件。
_orig_connect = data.connect
def _safe_connect():
    con = _orig_connect()
    try:
        con.sql("SET threads=2")
        con.sql("SET preserve_insertion_order=false")
        con.sql("PRAGMA temp_directory='%s'" % os.path.join(
            os.environ.get("TMPDIR", "/private/tmp/claude-501/-Users-shenboheng-CSMAR/a7ebc38d-e5d7-4391-a63d-c9b1f5693913/scratchpad"),
            "duckdb_tmp"))
        con.sql("PRAGMA max_temp_directory_size='40GiB'")
    except Exception:
        pass
    return con
data.connect = _safe_connect

FREQ, C, PPY = "M", 0.003, 12
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]
LO, HI = 800, 1800
TE = 0.03                         # 只报 TE=3% 口径
HALFLIVES = [None, 2, 3]          # None=不平滑(基线复现), 2/3=EWMA halflife(月)


def build_alpha_variants(panel):
    """返回 dict: name -> alpha Series(与 panel index 对齐)。
    EWMA 按 stkcd 分组、按时间排序，ewm(halflife).mean() 只用当期及过去(无前视)。"""
    out = {}
    # 排序以保证 ewm 沿时间正确累积
    order = panel.sort_values(["stkcd", "trddt"]).index
    raw = panel["lgb"]
    out["raw"] = raw.copy()
    g = panel.loc[order].groupby("stkcd")["lgb"]
    for hl in HALFLIVES:
        if hl is None:
            continue
        sm = g.transform(lambda s: s.ewm(halflife=hl, min_periods=1, adjust=True).mean())
        # sm 已按 order 索引对齐 panel.index（transform 保留原 index）
        out[f"ewma{hl}"] = sm.reindex(panel.index)
    return out


def run_variant(panel, alpha_col, f_df, i1000, style_cols):
    """对给定 alpha 列逐月优化，返回结果 DataFrame。"""
    rows, wprev = [], None
    dts = sorted(panel[panel[alpha_col].notna()]["trddt"].unique())
    for dt in dts:
        m = panel[(panel["trddt"] == dt) & panel[alpha_col].notna() & panel["industry"].notna() &
                  (panel["caprank"] > LO) & (panel["caprank"] <= HI)].copy()
        if len(m) < 200:
            continue
        b = (m["total_mktcap"] / m["total_mktcap"].sum()).values
        Xind = pd.get_dummies(m["industry"]).astype(float)
        cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values
        w = optimizer.optimize_enhanced(m[alpha_col].values, b, Xind.values, Xs, F, d,
                                        active_cap=0.02, te=TE, style_band=0.10)
        ws = pd.Series(w, index=m["stkcd"].values)
        to = 0.0 if wprev is None else 0.5 * ws.subtract(wprev, fill_value=0).abs().sum()
        rows.append({"dt": dt, "bench": float(np.nansum(b * fwd)), "i1000": i1000.get(dt, np.nan),
                     "port": float(np.nansum(w * fwd)), "to": to})
        wprev = ws
    return pd.DataFrame(rows).set_index("dt")


def report(R, label):
    port = R["port"] - R["to"] * C
    exS = (port - R["bench"]).dropna()
    exI = (port - R["i1000"]).dropna()
    navx = (1 + exS).cumprod()
    navxI = (1 + exI).cumprod()
    res = {
        "label": label,
        "exI": exI.mean() * PPY,
        "irI": exI.mean() / exI.std(ddof=1) * np.sqrt(PPY),
        "teI": exI.std(ddof=1) * np.sqrt(PPY),
        "mddI": (navxI / navxI.cummax() - 1).min(),
        "turn": R["to"].mean() * PPY,
        # 对合成基准（优化器实际控制的口径）
        "exS": exS.mean() * PPY,
        "irS": exS.mean() / exS.std(ddof=1) * np.sqrt(PPY),
        "teS": exS.std(ddof=1) * np.sqrt(PPY),
    }
    print(f"\n--- {label} (TE=3%, 扣{C:.1%}换手成本) ---")
    print(f"  对真实中证1000: 超额{res['exI']:.2%}  IR{res['irI']:.2f}  跟踪误差{res['teI']:.2%}  "
          f"超额回撤{res['mddI']:.2%}  年化换手{res['turn']:.1f}x")
    print(f"  对合成基准:     超额{res['exS']:.2%}  IR{res['irS']:.2f}  跟踪误差{res['teS']:.2%}")
    return res


def main():
    t0 = time.time()
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    pred = pd.read_parquet("results/lgb_oos_pred.parquet")
    pred["trddt"] = pred["trddt"].astype("datetime64[ns]")
    panel = panel.merge(pred, on=["stkcd", "trddt"], how="left")  # 列名保持 lgb
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    panel["caprank"] = panel.groupby("trddt")["total_mktcap"].rank(ascending=False, method="first")
    i1000 = backtest.load_benchmark("000852", FREQ)

    # 风险模型（用真实 fwd_ret 回归，与基线一致；alpha 平滑不影响风险模型）
    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid)
    sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")

    # 构造 alpha 变体
    variants = build_alpha_variants(panel)
    for name, s in variants.items():
        panel[f"alpha_{name}"] = s.values
    print(f"准备完成 {time.time()-t0:.0f}s，逐变体优化 ...", flush=True)

    results = {}
    label_map = {"raw": "不平滑(基线)", "ewma2": "EWMA halflife=2", "ewma3": "EWMA halflife=3"}
    for name in ["raw", "ewma2", "ewma3"]:
        t1 = time.time()
        R = run_variant(panel, f"alpha_{name}", f_df, i1000, style_cols)
        results[name] = report(R, label_map[name])
        print(f"    [{name} 用时 {time.time()-t1:.0f}s]", flush=True)

    # 汇总对比
    print("\n" + "=" * 72)
    print("汇总对比（对真实中证1000, TE=3%, 扣成本）")
    print("=" * 72)
    print(f"{'变体':<18}{'超额%':>8}{'IR':>7}{'TE%':>7}{'超额MDD%':>10}{'换手x':>8}")
    for name in ["raw", "ewma2", "ewma3"]:
        r = results[name]
        print(f"{label_map[name]:<18}{r['exI']*100:>8.2f}{r['irI']:>7.2f}{r['teI']*100:>7.2f}"
              f"{r['mddI']*100:>10.2f}{r['turn']:>8.1f}")

    # 选更优 halflife（按 IR 优先, 平手看换手）
    best = max(["ewma2", "ewma3"], key=lambda n: (results[n]["irI"], -results[n]["turn"]))
    print(f"\n更优平滑: {label_map[best]}  vs 不平滑 IR {results['raw']['irI']:.2f}→{results[best]['irI']:.2f}, "
          f"换手 {results['raw']['turn']:.1f}x→{results[best]['turn']:.1f}x")
    print(f"\n完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
