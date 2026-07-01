"""
35_drawdown_feedback.py  —  回撤反馈控制 (Drawdown-feedback overlay)

方法族: Nystrup-Boyd-Lindström 连续回撤反馈 + Grossman-Zhou/CPPI + Kaminski-Lo 止损。
标的: results/26_nav.csv 的 "V1多头" 月度净收益流 (已扣0.3%换手).
基线(样本外2018-2025): 年化20.9% / 波动24.1% / 夏普0.91 / 最大回撤-25.6% / 卡玛0.82.

核心思想: 把"当前(滞后的已实现)回撤"当作反馈信号来收敛敞口. 敞口在 [floor,1] 之间,
剩余 (1-scale) 放现金(此处保守设现金收益=0; 加RF只会更好).

无前视铁律:
  月 t 的敞口 scale_t 只能用 <= 月 t-1 末 的已实现净值/回撤. 我们用叠加组合(overlay)自身
  的净值峰值来算回撤, 且 scale_t = g(DD_{t-1}), 所有反馈量 .shift(1). CPPI 的 floor/NAV 也用
  t-1 末的 overlay 净值. 止损用 t-1 末的累计回撤. 全部滞后一期, 无窥视.
"""
import numpy as np
import pandas as pd

RF_ANNUAL = 0.0          # 现金月收益(保守取0); 设>0只会抬高结果
CASH_M = (1 + RF_ANNUAL) ** (1 / 12) - 1


# ----------------------------- 载入基线净收益流 -----------------------------
def load_base():
    nav = pd.read_csv("results/26_nav.csv")
    nav["dt"] = pd.to_datetime(nav["dt"])
    nav = nav.set_index("dt")
    r = nav["V1多头"].pct_change(fill_method=None).dropna()
    return r


# ----------------------------- 绩效指标 -----------------------------
def stats(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (12 / n) - 1
    vol = r.std() * np.sqrt(12)
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    # 下行波动 / Sortino
    downside = r[r < 0].std() * np.sqrt(12)
    sortino = r.mean() * 12 / downside if downside > 0 else np.nan
    return dict(ann=ann, vol=vol, sharpe=sharpe, mdd=mdd, calmar=calmar, sortino=sortino, n=n)


# ----------------------------- overlay 通用引擎 -----------------------------
def run_overlay(r: pd.Series, scale_fn):
    """
    scale_fn(state) -> scale in [0,1], 只看 t-1 末状态.
    state 包含到 t-1 末的 overlay 已实现净值路径, 因此天然无前视.
    返回: 叠加后月收益流 + 每月敞口序列.
    """
    r = r.dropna()
    idx = r.index
    ov_nav = [1.0]          # overlay 组合净值(含现金), t=0 起点
    peak = 1.0
    scales = []
    out = []
    for i, dt in enumerate(idx):
        cur_nav = ov_nav[-1]
        peak = max(peak, cur_nav)
        dd_now = cur_nav / peak - 1.0          # 到 t-1 末的已实现回撤 (<=0)
        state = dict(nav=cur_nav, peak=peak, dd=dd_now, i=i)
        s = float(np.clip(scale_fn(state), 0.0, 1.0))
        scales.append(s)
        # 本月收益: s 敞口在策略, (1-s) 在现金
        rm = s * r.iloc[i] + (1 - s) * CASH_M
        out.append(rm)
        ov_nav.append(cur_nav * (1 + rm))
    ro = pd.Series(out, index=idx)
    sc = pd.Series(scales, index=idx)
    return ro, sc


def turnover_of_scale(sc: pd.Series) -> float:
    """敞口的月均绝对变动 = 择时带来的额外换手(单边近似)."""
    return sc.diff().abs().mean()


# =========================================================================
# (a) 连续回撤反馈: scale = clip(1 - k*|DD|, floor, 1),  DD 为滞后已实现回撤
# =========================================================================
def variant_a(r, k, floor):
    def fn(st):
        return 1 - k * max(0.0, -st["dd"])   # -dd = 回撤幅度(正)
    return run_overlay(r, fn)


# =========================================================================
# (b) CPPI: 敞口 = clip( m*(NAV - floor_level)/NAV , 0, 1 )
#     floor_level = peak*(1-tol) (棘轮式 floor, 随峰值抬升). NAV/peak 均为 t-1 末.
# =========================================================================
def variant_cppi(r, m, tol):
    def fn(st):
        floor_level = st["peak"] * (1 - tol)
        cushion = st["nav"] - floor_level
        if cushion <= 0:
            return 0.0
        return m * cushion / st["nav"]
    return run_overlay(r, fn)


# =========================================================================
# (c) Kaminski-Lo 止损: 累计回撤 <= -thr 则清/半仓, 冷静 N 月, 之后按趋势(近M月>0)回场
# =========================================================================
def variant_klstop(r, thr, cool, exit_scale=0.0, trend_win=1):
    """
    thr: 触发止损的回撤阈值(正数, 如0.15).  exit_scale: 触发后敞口(0=清仓,0.5=半仓).
    cool: 触发后强制持有低敞口的最少月数.  trend_win: 回场需近 trend_win 个月 overlay收益>0.
    用状态机, 全部基于 t-1 末已实现量.
    """
    r = r.dropna(); idx = r.index
    ov_nav = 1.0; peak = 1.0
    in_stop = False; cool_left = 0
    recent = []                    # overlay 已实现月收益(用于趋势回场)
    scales = []; out = []
    for i, dt in enumerate(idx):
        peak = max(peak, ov_nav)
        dd_now = ov_nav / peak - 1.0
        # 决策(只看 t-1 末)
        if not in_stop:
            if dd_now <= -thr:
                in_stop = True; cool_left = cool; s = exit_scale
            else:
                s = 1.0
        else:
            if cool_left > 0:
                cool_left -= 1; s = exit_scale
            else:
                # 冷静期满, 看趋势是否转正
                trend_ok = len(recent) >= trend_win and all(x > 0 for x in recent[-trend_win:])
                if trend_ok:
                    in_stop = False; s = 1.0
                else:
                    s = exit_scale
        s = float(np.clip(s, 0, 1)); scales.append(s)
        rm = s * r.iloc[i] + (1 - s) * CASH_M
        out.append(rm); recent.append(rm); ov_nav *= (1 + rm)
    return pd.Series(out, index=idx), pd.Series(scales, index=idx)


# ----------------------------- 主流程: 扫参 + 报告 -----------------------------
def fmt(s, name, sc=None, base=None):
    extra = ""
    if sc is not None:
        extra += f" turn={turnover_of_scale(sc):.3f} avgexp={sc.mean():.2f}"
    line = (f"{name:<26} ann={s['ann']*100:5.1f}% vol={s['vol']*100:5.1f}% "
            f"shrp={s['sharpe']:.2f} mdd={s['mdd']*100:6.1f}% calmar={s['calmar']:.2f} "
            f"sortino={s['sortino']:.2f}")
    if base is not None:
        d_ann = (s['ann'] - base['ann']) * 100
        d_mdd = (s['mdd'] - base['mdd']) * 100   # >0 = 回撤变浅(好)
        eff = (-d_mdd) / (-d_ann) if d_ann < -1e-9 else np.inf  # 每少赚1pp收益换来几pp回撤下降
        line += f" | dANN={d_ann:+.1f} dMDD={d_mdd:+.1f} eff={eff:.2f}"
    return line + extra


def main():
    r = load_base()
    base = stats(r)
    print("="*140)
    print(fmt(base, "BASELINE(V1多头)"))
    print("  eff = 每少赚1pp年化 换来 几pp最大回撤下降 (越高越划算)")
    print("="*140)

    results = {"BASELINE": (base, None, r)}

    print("\n--- (a) 连续回撤反馈  scale=clip(1-k*|DD|, floor, 1) ---")
    for k in [1.0, 1.5, 2.0, 3.0]:
        for floor in [0.3, 0.5]:
            ro, sc = variant_a(r, k, floor)
            # 应用 floor 下限
            sc2 = sc.clip(lower=floor)
            ro2, sc2b = run_overlay(r, lambda st, k=k, fl=floor: max(fl, 1 - k*max(0.0,-st["dd"])))
            s = stats(ro2)
            nm = f"a k={k} floor={floor}"
            print("  " + fmt(s, nm, sc2b, base))
            results[nm] = (s, sc2b, ro2)

    print("\n--- (b) CPPI  exposure=clip(m*(NAV-peak*(1-tol))/NAV,0,1) ---")
    for tol in [0.10, 0.15]:
        for m in [2, 3]:
            ro, sc = variant_cppi(r, m, tol)
            s = stats(ro)
            nm = f"cppi m={m} tol={tol}"
            print("  " + fmt(s, nm, sc, base))
            results[nm] = (s, sc, ro)

    print("\n--- (c) Kaminski-Lo 止损  (thr触发, cool冷静, trend回场) ---")
    for thr in [0.12, 0.15, 0.20]:
        for exits in [0.0, 0.5]:
            for cool in [1, 2, 3]:
                ro, sc = variant_klstop(r, thr, cool, exit_scale=exits, trend_win=1)
                s = stats(ro)
                nm = f"kl thr={thr} exit={exits} cool={cool}"
                print("  " + fmt(s, nm, sc, base))
                results[nm] = (s, sc, ro)

    # -------- 选帕累托: 均衡档(几乎不损收益, 卡玛最高且dANN>=-2pp) 与 激进档(回撤砍最狠) --------
    print("\n" + "="*140)
    print("挑选:")
    rows = []
    for nm, (s, sc, ro) in results.items():
        if nm == "BASELINE":
            continue
        d_ann = (s['ann']-base['ann'])*100; d_mdd=(s['mdd']-base['mdd'])*100
        eff = (-d_mdd)/(-d_ann) if d_ann < -1e-9 else np.inf
        rows.append((nm, s['ann'], s['sharpe'], s['mdd'], s['calmar'], d_ann, d_mdd, eff,
                     turnover_of_scale(sc) if sc is not None else 0))
    df = pd.DataFrame(rows, columns=["name","ann","sharpe","mdd","calmar","dANN","dMDD","eff","turn"])

    # 均衡档: 损收益不超过 2pp, 在其中选卡玛最高
    balanced = df[df["dANN"] >= -2.0].sort_values("calmar", ascending=False)
    print("\n[均衡档候选 (dANN>=-2pp), 按卡玛降序 top5]")
    print(balanced.head(5).to_string(index=False))

    # 激进档: 回撤砍最狠(mdd 最浅/最大), 但要求还有正的效率(eff>=1, 即少赚1pp至少换1pp回撤)
    aggr = df[df["eff"] >= 1.0].sort_values("mdd", ascending=False)
    print("\n[激进档候选 (eff>=1, 回撤砍最狠), 按mdd降序 top5]")
    print(aggr.head(5).to_string(index=False))

    print("\n[全局卡玛最高 top5]")
    print(df.sort_values("calmar", ascending=False).head(5).to_string(index=False))

    df.to_csv("results/35_drawdown_feedback.csv", index=False)
    print("\n保存 results/35_drawdown_feedback.csv")
    return results, base, df


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
