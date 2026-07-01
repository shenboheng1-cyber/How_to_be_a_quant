# -*- coding: utf-8 -*-
"""
研究脚本 35 —— 回撤控制:把回撤/尾部风险度量放进优化目标 (CDaR / CVaR)
================================================================================
方法族(Chekhlov-Uryasev-Zabarankin):CDaR/CVaR 可线性规划求解。单一收益流无法
"CDaR 优化",故做【子组合配置】版本 —— 在几条子腿之间用 CDaR/CVaR 最优配权:
  子腿:
    ① LGB 多头 top-decile   (=基线 V1多头,动量小盘 alpha)
    ② 低波防御腿            (vol_60 最低 decile 的多头,低波扛踩踏)
    ③ 现金                   (0 收益,真正的回撤杀手)
    ④ 中证1000 指数腿        (000852,可选 beta 腿)
  优化:每月用【过去 W 个月】的子腿净收益,解 min-CDaR / min-CVaR (或
        "CDaR≤上限 下最大化收益") 的 LP,得下月权重,只用过去数据、下月持有。
  对比:CDaR 目标 vs CVaR 目标 vs 基线;扫窗长/回撤上限,报均衡/激进档。

无前视要点:
  * 每条子腿的净收益流本身合法(t 期信号→t→t+1 收益,已扣换手成本)。
  * 配置权重 w_{t} 只用 <= t 的子腿收益历史(严格 .iloc[:i],不含当期未知的 t 期)。
    w_t 作用于 t→t+1 收益。子腿间换手另计成本。
  * 无任何全样本 fit / 回填。滚动窗每月重解 LP。

用法：/opt/anaconda3/bin/python research/35_cdar_cvar_alloc.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import cvxpy as cp
from quantlib import data, universe
from quantlib.backtest import load_benchmark

FREQ, C, PPY = "M", 0.003, 12
ALLOC_C = 0.001   # 子腿间再平衡的换手成本率(单边0.1%,腿层面调仓比个股便宜)


# ----------------------------------------------------------------------------- 子腿构造
def long_decile(pnl, col, largest=True, cost=C):
    """等权 top-decile 多头净收益流;换手=新进名单占比;扣 cost 成本。"""
    rows, prev = [], set()
    sub = pnl.dropna(subset=[col, "fwd_ret"])
    for dt, g in sub.groupby("trddt"):
        k = max(1, len(g) // 10)
        top = g.nlargest(k, col) if largest else g.nsmallest(k, col)
        cur = set(top["stkcd"])
        to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows.append({"dt": dt, "g": top["fwd_ret"].mean(), "to": to}); prev = cur
    L = pd.DataFrame(rows).set_index("dt")
    return (L["g"] - L["to"] * cost).rename(col)


# ----------------------------------------------------------------------------- LP: CDaR (精确 running-max)
def solve_cdar_exact(R, mode="min_risk", ret_floor=None, dd_cap=None, alpha=0.05):
    """精确 CDaR:显式 running-max M_k >= u_i (i<=k)。更紧,LP 规模 O(T^2) 约束但 T 小可接受。"""
    T, N = R.shape
    w = cp.Variable(N, nonneg=True)
    u = cp.Variable(T + 1)
    M = cp.Variable(T + 1)          # running max of u[0..k]
    z = cp.Variable()
    y = cp.Variable(T, nonneg=True)
    port = R @ w
    cons = [cp.sum(w) == 1, u[0] == 0, M[0] == 0]
    for k in range(T):
        cons += [u[k + 1] == u[k] + port[k]]
        cons += [M[k + 1] >= M[k], M[k + 1] >= u[k + 1]]   # running max
        cons += [y[k] >= (M[k + 1] - u[k + 1]) - z]         # dd_k - z
    cdar = z + 1.0 / (alpha * T) * cp.sum(y)
    exp_ret = cp.sum(port) / T
    if mode == "min_risk":
        obj = cp.Minimize(cdar)
        if ret_floor is not None:
            cons += [exp_ret >= ret_floor]
    else:
        obj = cp.Maximize(exp_ret)
        cons += [cdar <= dd_cap]
    prob = cp.Problem(obj, cons)
    for solver in (cp.CLARABEL, cp.ECOS, cp.SCS):
        try:
            prob.solve(solver=solver)
            if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception:
            continue
    if w.value is None:
        return None
    ww = np.clip(np.asarray(w.value).ravel(), 0, None); s = ww.sum()
    return ww / s if s > 1e-8 else None


# ----------------------------------------------------------------------------- LP: CVaR
def solve_cvar(R, mode="min_risk", ret_floor=None, cvar_cap=None, alpha=0.05):
    """CVaR(Rockafellar-Uryasev)对【单期损失】的尾部条件均值。损失 = -port。
       CVaR = min_z z + 1/(alpha T) sum_k max(-port_k - z, 0)。"""
    T, N = R.shape
    w = cp.Variable(N, nonneg=True)
    z = cp.Variable()
    y = cp.Variable(T, nonneg=True)
    port = R @ w
    cons = [cp.sum(w) == 1]
    for k in range(T):
        cons += [y[k] >= -port[k] - z]
    cvar = z + 1.0 / (alpha * T) * cp.sum(y)
    exp_ret = cp.sum(port) / T
    if mode == "min_risk":
        obj = cp.Minimize(cvar)
        if ret_floor is not None:
            cons += [exp_ret >= ret_floor]
    else:
        obj = cp.Maximize(exp_ret)
        cons += [cvar <= cvar_cap]
    prob = cp.Problem(obj, cons)
    for solver in (cp.CLARABEL, cp.ECOS, cp.SCS):
        try:
            prob.solve(solver=solver)
            if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception:
            continue
    if w.value is None:
        return None
    ww = np.clip(np.asarray(w.value).ravel(), 0, None); s = ww.sum()
    return ww / s if s > 1e-8 else None


# ----------------------------------------------------------------------------- 滚动配置回测
def rolling_alloc(legs, solver_fn, window=36, min_train=24, **kw):
    """
    legs: DataFrame (T x N) 子腿净收益, index=月末。
    每月 t: 用 legs.iloc[t-window:t] (严格过去) 解权重 w_t, 作用于 legs.iloc[t] (=t→t+1 净收益)。
    子腿间再平衡换手成本 = 0.5*|w_t - w_prev_drifted| * ALLOC_C。
    返回 (组合净收益 Series, 权重 DataFrame)。
    """
    dates = legs.index
    T = len(dates)
    out_r, w_hist = {}, {}
    w_prev = None
    names = legs.columns.tolist()
    for i in range(T):
        if i < min_train:
            continue
        R = legs.iloc[max(0, i - window):i].values   # 严格过去, 不含 i
        if R.shape[0] < min_train:
            continue
        w = solver_fn(R, **kw)
        if w is None:
            w = w_prev if w_prev is not None else np.ones(len(names)) / len(names)
        # 本期实现收益(用当期 legs.iloc[i], 这是 t->t+1 的已扣个股换手净收益)
        r_legs = legs.iloc[i].values
        gross = float(np.dot(w, r_legs))
        # 子腿间换手成本:与上月末【漂移后】权重比
        if w_prev is not None:
            to = 0.5 * np.abs(w - w_prev).sum()
        else:
            to = 0.5 * np.abs(w).sum()
        net = gross - to * ALLOC_C
        out_r[dates[i]] = net
        w_hist[dates[i]] = w
        # 下月的"上月权重"= 本月权重按本月各腿收益漂移
        drift = w * (1 + r_legs)
        w_prev = drift / drift.sum() if drift.sum() > 1e-9 else w
    return pd.Series(out_r).sort_index(), pd.DataFrame(w_hist, index=names).T.sort_index()


def mfull(r):
    r = pd.Series(r).dropna(); n = len(r)
    ann = (1 + r).prod() ** (PPY / n) - 1
    vol = r.std(ddof=1) * np.sqrt(PPY)
    sharpe = ann / vol if vol else np.nan
    nav = (1 + r).cumprod(); mdd = (nav / nav.cummax() - 1).min()
    cal = ann / abs(mdd) if mdd else np.nan
    return {"年化": ann, "波动": vol, "夏普": sharpe, "回撤": mdd, "卡玛": cal, "胜率": (r > 0).mean(), "n": n}


def main():
    t0 = time.time()
    # ---- 面板 + 子腿 ----
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    pred = pd.read_parquet("results/lgb_oos_pred.parquet")
    pred["trddt"] = pred["trddt"].astype("datetime64[ns]")
    panel = panel.merge(pred, on=["stkcd", "trddt"], how="left")
    print(f"面板+预测 {time.time()-t0:.0f}s", flush=True)

    r_lgb = long_decile(panel, "lgb", largest=True)                 # ① 基线腿
    sub = panel[panel["lgb"].notna()].copy()                        # 低波腿限同 alpha 覆盖期
    r_lowvol = long_decile(sub, "vol_60", largest=False)            # ② 低波腿
    i1000 = load_benchmark("000852", FREQ).reindex(r_lgb.index)     # ④ 中证1000腿

    # 对齐到基线口径(基线从 2018-02 起,首月被 pct_change 丢掉)
    base = pd.read_csv("results/26_nav.csv"); base["dt"] = pd.to_datetime(base["dt"])
    base_r = base.set_index("dt")["V1多头"].pct_change(fill_method=None).dropna()
    idx = base_r.index

    r_lgb = r_lgb.reindex(idx); r_lowvol = r_lowvol.reindex(idx); i1000 = i1000.reindex(idx)
    cash = pd.Series(0.0, index=idx, name="cash")

    print("\n子腿单独表现(样本外 2018-02~2025-11):")
    for nm, s in [("①LGB多头", r_lgb), ("②低波防御", r_lowvol), ("③现金", cash), ("④中证1000", i1000)]:
        m = mfull(s); print(f"  {nm:10s} 年化{m['年化']*100:5.1f}% 波动{m['波动']*100:5.1f}% "
                            f"夏普{m['夏普']:.2f} 回撤{m['回撤']*100:6.1f}% 卡玛{m['卡玛']:.2f}")
    print("  基线校验 corr(①,基线)=%.5f" % pd.concat([r_lgb, base_r], axis=1).dropna().corr().iloc[0, 1])
    corr = pd.concat([r_lgb.rename("LGB"), r_lowvol.rename("LowVol"), i1000.rename("CSI1k")], axis=1).dropna().corr()
    print("  子腿相关:\n", corr.round(2).to_string())

    # 两套腿:3腿(①②③) 与 4腿(①②③④)
    legs3 = pd.concat([r_lgb.rename("LGB"), r_lowvol.rename("LowVol"), cash.rename("Cash")], axis=1)
    legs4 = pd.concat([r_lgb.rename("LGB"), r_lowvol.rename("LowVol"), cash.rename("Cash"),
                       i1000.rename("CSI1k")], axis=1)

    # ---- 扫参数/变体 ----
    # min_train=12: 子腿需 12 个月历史才能配置,策略实际从 2019-02 起活跃(仍不含 2018 深熊)。
    # 纯 min-CDaR/min-CVaR 会退化到 100% 现金(回撤=0 但收益没了),故主力用
    # "min-CDaR s.t. 期望收益>=下限" 的收益地板形式;dd_cap 形式作为对照(会 whipsaw、更差)。
    MT = 12
    results = {}
    weights_store = {}
    configs = [
        # (标签, legs, solver, kwargs)  —— 退化对照
        ("CDaR min(纯), W36, a.05, 3腿",     legs3, solve_cdar_exact, dict(mode="min_risk", alpha=0.05, window=36)),
        ("CVaR min(纯), W36, a.05, 3腿",     legs3, solve_cvar,       dict(mode="min_risk", alpha=0.05, window=36)),
        # 收益地板形式(主力):floor = 年化收益下限/12
        ("CDaR floor12%/yr, W36, 3腿",       legs3, solve_cdar_exact, dict(mode="min_risk", alpha=0.05, window=36, ret_floor=0.12/12)),
        ("CDaR floor14%/yr, W36, 3腿",       legs3, solve_cdar_exact, dict(mode="min_risk", alpha=0.05, window=36, ret_floor=0.14/12)),
        ("CDaR floor19%/yr, W36, 3腿",       legs3, solve_cdar_exact, dict(mode="min_risk", alpha=0.05, window=36, ret_floor=0.19/12)),
        ("CVaR floor14%/yr, W36, 3腿",       legs3, solve_cvar,       dict(mode="min_risk", alpha=0.05, window=36, cvar_cap=None) if False else
                                             dict(mode="min_risk", alpha=0.05, window=36, ret_floor=0.14/12)),
        ("CVaR floor19%/yr, W36, 3腿",       legs3, solve_cvar,       dict(mode="min_risk", alpha=0.05, window=36, ret_floor=0.19/12)),
        # 回撤上限形式(对照:whipsaw,MDD 反而更大)
        ("CDaR cap15%, W36, 3腿",            legs3, solve_cdar_exact, dict(mode="max_ret_dd_cap", alpha=0.05, window=36, dd_cap=0.15)),
        ("CDaR cap25%, W36, 3腿",            legs3, solve_cdar_exact, dict(mode="max_ret_dd_cap", alpha=0.05, window=36, dd_cap=0.25)),
        # 窗长/尾部/4腿 敏感性
        ("CDaR floor14%/yr, W24, 3腿",       legs3, solve_cdar_exact, dict(mode="min_risk", alpha=0.05, window=24, ret_floor=0.14/12)),
        ("CDaR floor14%/yr, W48, 3腿",       legs3, solve_cdar_exact, dict(mode="min_risk", alpha=0.05, window=48, ret_floor=0.14/12)),
        ("CDaR floor14%/yr, a.10, 3腿",      legs3, solve_cdar_exact, dict(mode="min_risk", alpha=0.10, window=36, ret_floor=0.14/12)),
        ("CDaR floor14%/yr, W36, 4腿",       legs4, solve_cdar_exact, dict(mode="min_risk", alpha=0.05, window=36, ret_floor=0.14/12)),
    ]
    for lbl, legs, fn, kw in configs:
        window = kw.pop("window")
        r, w = rolling_alloc(legs, fn, window=window, min_train=MT, **kw)
        kw["window"] = window
        results[lbl] = mfull(r)
        weights_store[lbl] = (r, w)
        print(f"  解算 {lbl:30s} {time.time()-t0:.0f}s", flush=True)

    # 基线:同期(公平)与全期(任务命名的对象 卡玛0.82)
    common_idx = weights_store["CDaR floor14%/yr, W36, 3腿"][0].index
    results["★基线 LGB多头(同活跃期)"] = mfull(base_r.reindex(common_idx))
    results["★基线 LGB多头(全期2018-)"] = mfull(base_r)

    # ---- 汇总 ----
    df = pd.DataFrame(results).T
    df["年化"] = (df["年化"] * 100).round(1)
    df["波动"] = (df["波动"] * 100).round(1)
    df["回撤"] = (df["回撤"] * 100).round(1)
    df["夏普"] = df["夏普"].round(2)
    df["卡玛"] = df["卡玛"].round(2)
    df["胜率"] = (df["胜率"] * 100).round(0)
    df["n"] = df["n"].astype(int)
    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 92)
    print("方法族:CDaR/CVaR 子组合配置(滚动窗、无前视;子腿间换手0.1%)")
    print("=" * 92)
    print(df.to_string())
    df.to_csv("results/35_cdar_cvar.csv", encoding="utf-8-sig")

    # ---- 落盘 NAV + 权重(选代表性变体) ----
    AGG = "CDaR floor12%/yr, W36, 3腿"     # 激进档:回撤砍最狠
    BAL = "CDaR floor19%/yr, W36, 3腿"     # 均衡档:几乎不损收益
    nav_dict = {"基线LGB多头": (1 + base_r.reindex(common_idx)).cumprod()}
    for lbl in [AGG, BAL, "CVaR floor19%/yr, W36, 3腿"]:
        nav_dict[lbl] = (1 + weights_store[lbl][0]).cumprod()
    pd.DataFrame(nav_dict).to_csv("results/35_nav.csv", encoding="utf-8-sig")

    # 2023-11~2024-05 踩踏期:看配置在危机前后如何切现金/低波
    crash = "2023-11", "2024-05"
    print("\n2023-11~2024-05 小盘踩踏期 —— 代表变体月初腿权重(只用过去数据,无前视):")
    for lbl in [BAL, AGG]:
        w = weights_store[lbl][1].loc[crash[0]:crash[1]]
        print(f"  [{lbl}]"); print((w * 100).round(0).to_string())

    # 权重时间序列平均(诚实展示 LGB/低波/现金 的长期配比)
    print("\n各变体长期平均腿权重(%):")
    for lbl in [AGG, BAL, "CDaR floor14%/yr, W36, 3腿",
                "CDaR cap25%, W36, 3腿", "CVaR floor19%/yr, W36, 3腿"]:
        w = weights_store[lbl][1]
        print(f"  {lbl:30s} " + "  ".join(f"{c}={w[c].mean()*100:.0f}%" for c in w.columns))

    print(f"\n完成 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
