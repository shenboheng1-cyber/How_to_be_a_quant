# -*- coding: utf-8 -*-
"""
研究脚本 35 (factor_hrp) —— 回撤控制方法族：【因子层面风险管理 = 防御因子 + 因子风险平价(HRP)】
================================================================================
目标策略：全A、月频、LGB-231 多头 top-decile。
基线(样本外 2018-2025，本脚本要改善的对象):
    年化 20.9% / 波动 24.1% / 夏普 0.91 / 最大回撤 -25.6% / 卡玛 0.82
(本脚本从 lgb_oos_pred.parquet 复现主腿，得到年化19.8/夏普0.82/回撤-25.6/卡玛0.77，
 与基线口径一致，corr=1.0；细微年化差异=复现多含2018-01一期。判据一律用卡玛与效率。)

方法族做法：
  在【个股层面】构造几条与 LGB 主腿低相关的"防御腿"多头 top-decile 月收益：
    ① LGB 主腿 (动量+小盘 alpha)
    ② 低波动 low_vol   (-vol_60，只 winsor+zscore，保留防御 tilt，不做市值中性化)
    ③ 低beta 代理      (大市值 log(mktcap)，即 size+low_vol 近似的低beta腿)
    ④ 质量 quality     (gross_prof + low_lev + roe 合成)
  合成方式对比：(a)等权 (b)逆波动率 inverse-vol (c)HRP 层次风险平价(Lopez de Prado)。
  权重一律用【滚动窗口、只用过去】的协方差估计，下月持有 —— 严格无前视。

无前视自查：
  - 防御腿信号(vol_60/mktcap/基本面)都来自 data.load_research_panel 的调仓日快照，
    只含 t 及以前(vol_60 是过去60日、基本面是 PIT 次年5-1 可用)。
  - trailing beta 若用则只用【已实现】(过去)收益回归 —— 本脚本诊断后弃用(噪声大)。
  - 合成权重 w_t 用 R.iloc[i-W:i](严格不含第 i 期)的协方差估计，作用于 R.iloc[i]。
    warmup 期用等权。没有任何"全样本 fit 再回填"。

诚实结论(见文末打印)：防御腿会大幅拉低收益，且长多头腿彼此 beta 相关 0.69-0.86，
风险平价几乎无可分散，卡玛并未改善。如实报告帕累托与效率。

用法：/opt/anaconda3/bin/python research/35_factor_hrp.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from quantlib import (data, universe, preprocess, evaluate, backtest,
                      altdata, fundamentals)

FREQ, C, PPY, WARM = "M", 0.003, 12, 12
QUAL = ["f_gross_prof", "f_low_lev", "f_roe"]   # 质量腿三合一
OOS = "2018-01-01"


# ----------------------------- 工具 -----------------------------
def mfull(r):
    r = r.dropna(); n = len(r)
    ann = (1 + r).prod() ** (PPY / n) - 1
    vol = r.std(ddof=1) * np.sqrt(PPY)
    nav = (1 + r).cumprod(); mdd = (nav / nav.cummax() - 1).min()
    return {"年化": ann, "波动": vol, "夏普": ann / vol,
            "最大回撤": mdd, "卡玛": ann / abs(mdd), "胜率": (r > 0).mean()}


def zx(p, col):
    return p.groupby("trddt")[col].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))


def zonly(p, raw):
    """只 winsor + 横截面 zscore，不做市值/行业中性化——保留防御 tilt。"""
    return preprocess.preprocess_factor(p, raw, industry_col=None, do_neutralize=False)


def long_stream(p, col):
    """个股层面 top-decile 等权多头月净收益(扣 0.3% 单边换手成本)。"""
    rows, prev = [], set()
    for dt, g in p.dropna(subset=[col, "fwd_ret"]).groupby("trddt"):
        top = g.nlargest(max(1, len(g) // 10), col); cur = set(top["stkcd"])
        to = 1.0 if not prev else len(cur - prev) / len(cur)
        rows.append({"dt": dt, "g": top["fwd_ret"].mean(), "to": to}); prev = cur
    L = pd.DataFrame(rows).set_index("dt")
    return L["g"] - L["to"] * C


# ----------------------- 风险平价权重算子 -----------------------
def w_equal(cov):
    n = cov.shape[0]; return np.ones(n) / n


def w_invvol(cov):
    iv = 1.0 / np.sqrt(np.diag(cov)); return iv / iv.sum()


def w_hrp(cov):
    """HRP (Lopez de Prado 2016): 相关距离聚类 -> 拟对角 -> 递归二分。"""
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    cond = squareform(dist, checks=False)
    link = sch.linkage(cond, method="single")
    sort_ix = sch.leaves_list(sch.optimal_leaf_ordering(link, cond))
    w = pd.Series(1.0, index=sort_ix); clusters = [list(sort_ix)]

    def cvar(idx):
        c = cov[np.ix_(idx, idx)]; ivp = 1.0 / np.diag(c); ivp /= ivp.sum()
        return ivp @ c @ ivp

    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            c0, c1 = clusters[i], clusters[i + 1]
            v0, v1 = cvar(c0), cvar(c1); a = 1 - v0 / (v0 + v1)
            w[c0] *= a; w[c1] *= 1 - a
    return w.sort_index().values


WFUN = {"等权": w_equal, "逆波动": w_invvol, "HRP": w_hrp}


def combine(R, cols, method, W=24, warm=WARM):
    """滚动窗口(只用过去)协方差 -> 权重 -> 下期持有。无前视。返回合成月收益 + 平均权重。"""
    Rs = R[cols]; n = len(cols)
    wser = pd.DataFrame(index=Rs.index, columns=cols, dtype=float)
    for i in range(len(Rs)):
        if i < warm or n == 1:
            wser.iloc[i] = 1.0 / n; continue
        hist = Rs.iloc[max(0, i - W):i].values          # 严格不含第 i 期
        cov = np.cov(hist.T, ddof=1)
        wser.iloc[i] = WFUN[method](cov)
    port = (wser * Rs).sum(axis=1)
    return port, wser


# ----------------------------- 主流程 -----------------------------
def main():
    t0 = time.time()
    # 1) 面板 + 主腿 LGB(OOS 预测) + 基本面/行业
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel); panel = fundamentals.attach(panel)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    a = pd.read_parquet("results/lgb_oos_pred.parquet")
    a["trddt"] = a["trddt"].astype("datetime64[ns]")
    panel = panel.merge(a, on=["stkcd", "trddt"], how="left")
    print(f"面板加载完 {time.time()-t0:.0f}s", flush=True)

    # 2) 四条腿因子(个股层面)
    panel["z_lgb"]  = zx(panel, "lgb")                                 # ① 主腿
    panel["z_lvol"] = zonly(panel, -panel["vol_60"])                   # ② 低波动
    panel["z_large"] = zonly(panel, np.log(panel["total_mktcap"]))     # ③ 低beta代理(大市值)
    Q = []                                                             # ④ 质量三合一
    for k in QUAL:
        z = zonly(panel, fundamentals.REGISTRY[k][0](panel))
        z = z * np.sign(evaluate.compute_ic(panel, z).mean())
        Q.append(z.values)
    panel["z_qual"] = np.nanmean(np.column_stack(Q), axis=1)

    # 3) 各腿 top-decile 多头净收益流
    legs = {}
    for nm, col in [("LGB", "z_lgb"), ("LowVol", "z_lvol"),
                    ("Large", "z_large"), ("Quality", "z_qual")]:
        legs[nm] = long_stream(panel, col)
    R = pd.DataFrame(legs).dropna()
    R = R[R.index >= OOS]

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 84)
    print("各腿单独业绩(2018+，个股 top-decile 多头，扣 0.3% 换手)")
    print("=" * 84)
    leg_tab = pd.DataFrame({k: mfull(R[k]) for k in R.columns}).T
    print(fmt(leg_tab))
    print("\n防御腿与 LGB 主腿相关性(月收益):")
    print(R.corr()["LGB"].round(3).to_string())

    # 4) 合成: 全四腿风险平价(等权/逆波动/HRP)
    DEF = ["LowVol", "Large", "Quality"]
    print("\n" + "=" * 84)
    print("方案A：四腿(含主腿)整体风险平价合成 —— 主腿被稀释到 ~1/4")
    print("=" * 84)
    rowsA = {}
    for m in ["等权", "逆波动", "HRP"]:
        port, wser = combine(R, ["LGB"] + DEF, m)
        rowsA[m] = mfull(port[port.index >= OOS])
        rowsA[m]["主腿权重"] = wser["LGB"].mean()
    print(fmt(pd.DataFrame(rowsA).T, extra=["主腿权重"]))

    # 5) 合成: 防御腿(HRP)成"防御 sleeve"，再与主腿按 w_def 混合 —— 帕累托扫描
    print("\n" + "=" * 84)
    print("方案B：防御腿 HRP 合成为 sleeve，再与 LGB 主腿按 w_def 混合(扫描 w_def)")
    print("=" * 84)
    sleeve, _ = combine(R, DEF, "HRP")
    lgb = R["LGB"]; base = mfull(lgb[lgb.index >= OOS])
    a0, d0 = base["年化"], base["最大回撤"]
    rowsB = {}
    for wd in [0.0, 0.10, 0.15, 0.20, 0.30, 0.50]:
        port = ((1 - wd) * lgb + wd * sleeve)
        mm = mfull(port[port.index >= OOS])
        # 效率 = 每少赚 1pp 年化 换来的回撤下降 pp（>1 好，<1 不划算）
        drop_ret = (a0 - mm["年化"]) * 100
        cut_dd = (mm["最大回撤"] - d0) * 100   # 回撤更浅=正
        mm["效率(砍撤pp/损益pp)"] = (cut_dd / drop_ret) if drop_ret > 1e-9 else np.nan
        rowsB[f"w_def={wd:.2f}"] = mm
    tabB = pd.DataFrame(rowsB).T
    print(fmt(tabB, extra=["效率(砍撤pp/损益pp)"]))

    # 6) sleeve 用单腿 LowVol(相关性/效率最优的防御腿) 再扫一次
    print("\n" + "=" * 84)
    print("方案C：仅用 LowVol 单腿作防御 sleeve(相关最低、效率最优)")
    print("=" * 84)
    rowsC = {}
    for wd in [0.10, 0.15, 0.20, 0.30]:
        port = (1 - wd) * lgb + wd * R["LowVol"]
        mm = mfull(port[port.index >= OOS])
        drop_ret = (a0 - mm["年化"]) * 100; cut_dd = (mm["最大回撤"] - d0) * 100
        mm["效率(砍撤pp/损益pp)"] = (cut_dd / drop_ret) if drop_ret > 1e-9 else np.nan
        rowsC[f"w_def={wd:.2f}"] = mm
    print(fmt(pd.DataFrame(rowsC).T, extra=["效率(砍撤pp/损益pp)"]))

    # 7) 崩盘窗口诊断
    print("\n" + "=" * 84)
    print("2023-12~2024-02 小微盘踩踏期 各腿月收益(%)")
    print("=" * 84)
    print((R.loc["2023-12":"2024-02"] * 100).round(1).to_string())
    print("\n2018 熊市 各腿累计收益(%):")
    print(((1 + R.loc["2018"]).prod() - 1).mul(100).round(1).to_string())

    # 8) 存盘：帕累托挑两档 (激进=w_def0.30 HRP sleeve；均衡=w_def0.15 LowVol)
    aggr = ((1 - 0.30) * lgb + 0.30 * sleeve)
    bal  = (1 - 0.15) * lgb + 0.15 * R["LowVol"]
    nav = pd.DataFrame({
        "LGB主腿": (1 + lgb).cumprod(),
        "均衡(LGB+15%LowVol)": (1 + bal).cumprod(),
        "激进(LGB+30%HRP防御)": (1 + aggr).cumprod(),
    })
    nav.to_csv("results/35_factor_hrp_nav.csv", encoding="utf-8-sig")
    print(f"\n均衡档: {short(mfull(bal[bal.index>=OOS]))}")
    print(f"激进档: {short(mfull(aggr[aggr.index>=OOS]))}")
    print(f"基线主腿: {short(base)}")
    print(f"\n存盘 results/35_factor_hrp_nav.csv  用时 {time.time()-t0:.0f}s")


def fmt(tab, extra=None):
    t = tab.copy()
    for c in ["年化", "波动", "最大回撤"]:
        if c in t: t[c] = (t[c] * 100).round(1).astype(str) + "%"
    if "夏普" in t: t["夏普"] = t["夏普"].round(2)
    if "卡玛" in t: t["卡玛"] = t["卡玛"].round(2)
    if "胜率" in t: t["胜率"] = (t["胜率"] * 100).round(0).astype(int).astype(str) + "%"
    for e in (extra or []):
        if e in t: t[e] = t[e].round(2)
    return t.to_string()


def short(m):
    return (f"年化{m['年化']*100:.1f}% 夏普{m['夏普']:.2f} "
            f"回撤{m['最大回撤']*100:.1f}% 卡玛{m['卡玛']:.2f}")


if __name__ == "__main__":
    main()
