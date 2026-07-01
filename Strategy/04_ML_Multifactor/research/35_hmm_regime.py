# -*- coding: utf-8 -*-
"""
研究脚本 35 —— HMM / Regime-Switching 回撤控制
================================================================
方法族：状态切换 Regime-Switching。在市场(中证1000)月收益(+已实现波动)
上滚动拟合 2-3 状态高斯 HMM，识别 "risk-off" 高波/负收益状态，在该状态把
多头 top-decile 组合敞口降到 low ∈ {0, 0.3, 0.5}。

无前视保障(逐条自查)：
  1. 市场特征只用中证1000自身收益/已实现波动，全部为【当期及之前】。
  2. HMM 采用【扩张窗】：在每个调仓日 t，仅用 data[: t] 拟合 HMM 并
     predict 当前状态 posterior，严禁全样本 fit 一次回填。
  3. 状态→敞口的映射再 .shift(1)：t 月策略收益用的仓位 = t-1 已知状态所决定，
     严格滞后一期，避免用到 t 月本身的市场收益。
  4. "哪个状态是 risk-off" 也只用扩张窗内已见样本的状态均值来判定(标签不稳，
     用 posterior 均值最低者 = risk-off)。

用法：/opt/anaconda3/bin/python research/35_hmm_regime.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from quantlib import backtest

np.random.seed(0)
PPY = 12

# ----------------------------------------------------------------------
# 1. 数据：策略净收益流 + 市场(中证1000)已实现收益/波动
# ----------------------------------------------------------------------
nav = pd.read_csv("results/26_nav.csv", index_col=0, parse_dates=True)
strat = nav["V1多头"].pct_change(fill_method=None).dropna()          # 已扣0.3%换手的净流

# 市场收益：用中证1000。load_benchmark 返回【未来一期】收益(t→t+1)，
# 我们要 "月末 t 已实现" 口径 → shift(1)。市场历史回溯到 2015 做 HMM burn-in。
mkt_fwd = backtest.load_benchmark("000852", "M", "2015-01-01", "2025-12-31")
mkt = mkt_fwd.shift(1).dropna()                                       # 值@t = t-1→t 已实现收益
mkt.name = "mkt"

# 市场已实现波动(近3个月，滞后到 t 为止全是过去信息)
mkt_vol = mkt.rolling(3, min_periods=2).std()

feat_full = pd.DataFrame({"ret": mkt, "vol": mkt_vol}).dropna()       # HMM 输入特征(市场层面)


# ----------------------------------------------------------------------
# 2. 滚动/扩张窗 HMM：每个 t 只用 data[:t] 拟合，输出【下一期】敞口的原料
# ----------------------------------------------------------------------
def rolling_hmm_riskoff(feat: pd.DataFrame, n_states: int, use_vol: bool,
                        criterion: str = "sharpe", min_train: int = 24,
                        engine: str = "hmm"):
    """
    返回 Series: prob_riskoff@t = 在 t 月末(用 data[:t] 拟合)判定当前处于
    risk-off 状态的后验概率。全程扩张窗、无前视。
    risk-off 状态判定(用训练窗内 posterior 加权统计量，无前视)：
      criterion='ret'    : 收益均值最低的状态(⚠ 常把"低漂移震荡态"误判为 risk-off，
                            导致长期低仓，不是真 regime 择时)。
      criterion='sharpe' : (收益均值 − 波动均值) 最低的状态 = 低收益+高波的【真崩盘态】。
                            需要 use_vol=True 才有 vol 维度可分离。
    engine: 'hmm' 用 hmmlearn.GaussianHMM; 'gmm' 用 sklearn GaussianMixture(2态代理)。
    """
    cols = ["ret", "vol"] if use_vol else ["ret"]
    X_all = feat[cols].values
    idx = feat.index
    out = pd.Series(np.nan, index=idx, name="p_off")

    if engine == "hmm":
        from hmmlearn.hmm import GaussianHMM
    else:
        from sklearn.mixture import GaussianMixture

    for i in range(len(idx)):
        if i + 1 < min_train:                    # 训练样本不足，视为 risk-on(不减仓)
            out.iloc[i] = 0.0
            continue
        Xtr = X_all[: i + 1]                     # 只用到 t 为止(含 t)
        # 标准化(用训练窗自身统计量，无前视)
        mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
        Xz = (Xtr - mu) / sd
        try:
            if engine == "hmm":
                model = GaussianHMM(n_components=n_states, covariance_type="diag",
                                    n_iter=100, random_state=0, tol=1e-3)
            else:
                model = GaussianMixture(n_components=n_states, covariance_type="diag",
                                        random_state=0, max_iter=200)
            model.fit(Xz)
            post = model.predict_proba(Xz)                  # (T, n_states)
            ret_tr = Xtr[:, 0]
            mret = (post * ret_tr[:, None]).sum(0) / (post.sum(0) + 1e-12)
            if criterion == "sharpe" and use_vol:
                vol_tr = Xtr[:, 1]
                mvol = (post * vol_tr[:, None]).sum(0) / (post.sum(0) + 1e-12)
                score = mret - mvol                         # 低收益+高波=最差
            else:
                score = mret
            off = int(np.argmin(score))
            out.iloc[i] = post[-1, off]                     # 当前(t)处于 risk-off 的后验
        except Exception:
            out.iloc[i] = out.iloc[i - 1] if i > 0 and not np.isnan(out.iloc[i - 1]) else 0.0
    return out


# ----------------------------------------------------------------------
# 3. 敞口叠加 + 业绩度量
# ----------------------------------------------------------------------
def build_exposure(p_off: pd.Series, thr: float, low: float) -> pd.Series:
    """risk-off 后验 > thr → 敞口 low，否则 1.0。再 .shift(1) 严格滞后一期。"""
    raw = pd.Series(1.0, index=p_off.index)
    raw[p_off > thr] = low
    return raw.shift(1)                                     # t 月仓位用 t-1 状态


def metrics(r: pd.Series, base_cost_bps: float = 0.0):
    r = r.dropna()
    if len(r) < 3:
        return dict(ann=np.nan, vol=np.nan, sharpe=np.nan, mdd=np.nan, calmar=np.nan)
    n = (1 + r).cumprod()
    ann = (1 + r).prod() ** (PPY / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(PPY)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(PPY)
    mdd = (n / n.cummax() - 1).min()
    return dict(ann=ann, vol=vol, sharpe=sharpe, mdd=mdd, calmar=ann / abs(mdd))


def apply_timing(strat_r: pd.Series, expo: pd.Series, switch_cost: float = 0.001):
    """
    敞口应用到策略净流。择时换手成本：敞口变动时按 |Δexpo|*switch_cost 计,
    近似 top-decile 组合整体加减仓的交易摩擦(默认 10bp 单边等价，双边≈)。
    敞口<1 的空出部分假设放【现金/零收益】(不加杠杆，cap=1)。
    """
    e = expo.reindex(strat_r.index).ffill().fillna(1.0)
    timed = e * strat_r
    dexp = e.diff().abs().fillna(0.0)
    timed = timed - dexp * switch_cost
    return timed, e


# ----------------------------------------------------------------------
# 4. 扫参数：states x feature x low x thr，报告帕累托
# ----------------------------------------------------------------------
def run():
    base = metrics(strat)
    print("=" * 78)
    print("基线 V1多头(样本外 2018-2025): 年化%.1f%% 波动%.1f%% 夏普%.2f 回撤%.1f%% 卡玛%.2f"
          % (base["ann"] * 100, base["vol"] * 100, base["sharpe"], base["mdd"] * 100, base["calmar"]))
    print("=" * 78)

    # 配置：criterion='sharpe' 需 use_vol=True(要 vol 维度分离崩盘态);
    #       criterion='ret' 保留作对照(会暴露"长期低仓陷阱")。
    configs = []
    for engine in ["hmm", "gmm"]:
        for n_states in [2, 3]:
            configs.append((engine, n_states, True, "sharpe"))
        configs.append((engine, 3, True, "ret"))          # ret 口径对照(仅3态)

    # 预计算每个 config 的 risk-off 后验(最贵的一步)
    p_cache = {}
    for engine, n_states, use_vol, crit in configs:
        key = (engine, n_states, use_vol, crit)
        p_cache[key] = rolling_hmm_riskoff(feat_full, n_states, use_vol,
                                           criterion=crit, engine=engine)

    rows = []
    for (engine, n_states, use_vol, crit), p_off in p_cache.items():
        for thr in [0.5, 0.7]:
            for low in [0.0, 0.3, 0.5]:
                expo = build_exposure(p_off, thr, low)
                timed, e = apply_timing(strat, expo)
                m = metrics(timed)
                e_al = e.reindex(strat.index).ffill().fillna(1.0)
                frac_off = float((e_al < 1.0).mean())
                rows.append({
                    "engine": engine, "states": n_states, "crit": crit,
                    "thr": thr, "low": low, "off_frac": round(frac_off, 2),
                    "ann": round(m["ann"] * 100, 1), "vol_ann": round(m["vol"] * 100, 1),
                    "sharpe": round(m["sharpe"], 2), "mdd": round(m["mdd"] * 100, 1),
                    "calmar": round(m["calmar"], 2),
                    "d_ann": round((m["ann"] - base["ann"]) * 100, 1),
                    "d_mdd": round((m["mdd"] - base["mdd"]) * 100, 1),
                })
    res = pd.DataFrame(rows)
    # 效率：每少赚1pp收益换来几pp回撤下降 (d_mdd>0 = 回撤变浅)
    res["eff"] = np.where(res["d_ann"] < 0,
                          (res["d_mdd"]) / (-res["d_ann"]).clip(lower=0.1),
                          np.inf)
    res = res.sort_values(["calmar"], ascending=False).reset_index(drop=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    print("\n全部变体(按卡玛降序):")
    print(res.to_string())

    res.to_csv("results/35_hmm_regime.csv", index=False, encoding="utf-8-sig")

    # 帕累托选档：只在【真崩盘态】口径(crit=sharpe, off_frac 合理 0.05~0.4)里挑，
    # 排除"长期低仓陷阱"(off_frac 过高说明不是真择时)。
    pool = res[(res["crit"] == "sharpe") & (res["off_frac"] <= 0.40) & (res["off_frac"] >= 0.05)]
    # 均衡档：几乎不损收益(d_ann>=-3) 里卡玛最高；激进档:卡玛最高(容忍更大收益牺牲换回撤)
    balanced = pool[pool["d_ann"] >= -3.0].sort_values("calmar", ascending=False).head(1)
    if len(balanced) == 0:
        balanced = pool.sort_values("d_ann", ascending=False).head(1)
    aggressive = pool.sort_values(["calmar"], ascending=False).head(1)
    print("\n" + "-" * 78)
    print("均衡档(几乎不损收益):"); print(balanced.to_string(index=False))
    print("\n激进档(回撤砍最狠 / 卡玛最高):"); print(aggressive.to_string(index=False))
    print("-" * 78)

    # 诊断:最优均衡档在 2018/2024 的表现 & 减仓时点
    if len(balanced):
        b = balanced.iloc[0]
        p_off = p_cache[(b["engine"], b["states"], True, b["crit"])]
        expo = build_exposure(p_off, b["thr"], b["low"])
        timed, e = apply_timing(strat, expo)
        e_al = e.reindex(strat.index).ffill().fillna(1.0)
        off_dates = e_al[e_al < 1.0].index
        print("\n均衡档减仓月份(共%d个):" % len(off_dates),
              [d.strftime("%Y-%m") for d in off_dates])
        for yr in ["2018", "2022", "2024"]:
            rb = strat.loc[yr + "-01":yr + "-12"]
            rt = timed.loc[yr + "-01":yr + "-12"].dropna()
            if len(rb) and len(rt):
                nb = (1 + rb).cumprod(); nt = (1 + rt).cumprod()
                print("  %s: 基线回撤%.1f%% -> 择时回撤%.1f%% | 基线收益%.1f%% -> 择时收益%.1f%%"
                      % (yr, (nb / nb.cummax() - 1).min() * 100,
                         (nt / nt.cummax() - 1).min() * 100,
                         ((1 + rb).prod() - 1) * 100, ((1 + rt).prod() - 1) * 100))
    return res, base, p_cache, balanced, aggressive


if __name__ == "__main__":
    run()
