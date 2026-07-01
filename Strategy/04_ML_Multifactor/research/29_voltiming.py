# -*- coding: utf-8 -*-
"""
研究脚本 29_voltiming —— 中证1000 指增 + 【拥挤/波动择时降主动风险】杠杆
================================================================
在基线 29_csi1000_product (TE=3% 优化器版) 之上，对【超额收益流】做时序降仓：
高波动 / 高拥挤月份把主动仓位 scale 调低 (scaled_excess = scale_t * excess_t)，
scale 用滞后信号、cap=1 (只减不加)，等价于风险高的月份缩小相对基准偏离。

三种 scale 信号 (全滞后，无前视)：
  A) vol_target  : 对超额流做波动目标，目标=超额历史波动，cap=1
  B) crowding    : 因子收益两两相关上升=拥挤=踩踏 → 历史高位月降到 low_expo
  C) combo       : A、B 取较小者 (任一风险高就降)

对每种信号，对比降仓前后的 超额/IR/超额回撤/超额卡玛(对合成基准 + 对真实中证1000)。
换手按 w_real=b+scale*(w-b) 重算 (scale 变动也产生换手)。

用法：/opt/anaconda3/bin/python research/29_voltiming.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from quantlib import (data, universe, preprocess, backtest, fundamentals, altdata,
                      riskmodel, optimizer, regime)
from quantlib.factors import classic

FREQ, C, PPY = "M", 0.003, 12
STYLE = ["reversal", "low_turnover", "low_vol", "size", "bp", "illiquidity"]
LO, HI = 800, 1800
TE = 0.03                      # 只跑 TE=3% 口径


def metrics(ex, ppy=PPY):
    """超额年化/IR/TE/超额回撤/超额卡玛/月胜率。ex 为月度超额序列。"""
    ex = ex.dropna()
    ann = ex.mean() * ppy
    te = ex.std(ddof=1) * np.sqrt(ppy)
    ir = ex.mean() / ex.std(ddof=1) * np.sqrt(ppy)
    navx = (1 + ex).cumprod()
    mdd = (navx / navx.cummax() - 1).min()
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    return dict(ann=ann, te=te, ir=ir, mdd=mdd, calmar=calmar, win=(ex > 0).mean())


def main():
    t0 = time.time()
    panel = data.load_research_panel(FREQ, "2015-01-01", "2025-12-31")
    panel = universe.filter_universe(panel, min_list_days=120, verbose=False)
    panel = altdata.attach_industry(panel)
    panel["trddt"] = panel["trddt"].astype("datetime64[ns]")
    pred = pd.read_parquet("results/lgb_oos_pred.parquet"); pred["trddt"] = pred["trddt"].astype("datetime64[ns]")
    panel = panel.merge(pred, on=["stkcd", "trddt"], how="left").rename(columns={"lgb": "alpha"})
    for k in STYLE:
        panel["sty_" + k] = preprocess.preprocess_factor(panel, classic.REGISTRY[k][0](panel), do_neutralize=False).values
    style_cols = ["sty_" + k for k in STYLE]
    panel["caprank"] = panel.groupby("trddt")["total_mktcap"].rank(ascending=False, method="first")
    i1000 = backtest.load_benchmark("000852", FREQ)
    print(f"准备完成 {time.time()-t0:.0f}s，逐月优化 ...", flush=True)

    f_df, resid, _ = riskmodel.factor_returns(panel, style_cols, "industry")
    sv = riskmodel.specific_var(panel, resid); sv["specvar"] = sv.groupby("stkcd")["specvar"].shift(1)
    panel = panel.merge(sv, on=["stkcd", "trddt"], how="left")

    rows, wprev = [], None
    for dt in sorted(panel[panel["alpha"].notna()]["trddt"].unique()):
        m = panel[(panel["trddt"] == dt) & panel["alpha"].notna() & panel["industry"].notna() &
                  (panel["caprank"] > LO) & (panel["caprank"] <= HI)].copy()
        if len(m) < 200: continue
        b = (m["total_mktcap"] / m["total_mktcap"].sum()).values
        Xind = pd.get_dummies(m["industry"]).astype(float); cols = list(Xind.columns) + style_cols
        F = riskmodel.factor_cov(f_df.loc[:dt].iloc[:-1]).reindex(index=cols, columns=cols).fillna(0).values
        Xs = m[style_cols].fillna(0.0).values
        d = m["specvar"].fillna(m["specvar"].median()).fillna(0.04).values
        fwd = m["fwd_ret"].values
        w = optimizer.optimize_enhanced(m["alpha"].values, b, Xind.values, Xs, F, d,
                                        active_cap=0.02, te=TE, style_band=0.10)
        ws = pd.Series(w, index=m["stkcd"].values)
        bs = pd.Series(b,  index=m["stkcd"].values)
        rows.append({"dt": dt, "bench": float(np.nansum(b * fwd)),
                     "i1000": i1000.get(dt, np.nan), "port": float(np.nansum(w * fwd)),
                     "w": ws, "b": bs})
        wprev = ws
    R = pd.DataFrame(rows).set_index("dt")

    # ---- 月度主动权重序列 (用于按 scale 重算换手) ----
    # 把每期的 w, b 存成 dict，scale 后 w_real = b + scale*(w-b)
    ws_list = R["w"].tolist(); bs_list = R["b"].tolist(); idx = R.index

    # ---- 基线超额流 (对合成基准, 这是优化器直接控制的; 同时给对真实1000) ----
    exS_raw = (R["port"] - R["bench"])                  # 主动收益流 (scale 作用对象)
    exI_raw = (R["port"] - R["i1000"])                  # 对真实中证1000

    # ============ 构造三种 scale 信号 (滞后, cap=1) ============
    # A) vol_target: 目标=超额历史波动 (整段中位水平)，scale=tgt/realized，cap=1
    realized = exS_raw.rolling(6, min_periods=3).std().shift(1) * np.sqrt(PPY)
    tgt = (exS_raw.std(ddof=1) * np.sqrt(PPY))           # 用全样本超额波动作目标(标定到≈1)
    scale_A = (tgt / realized).clip(upper=1.0).fillna(1.0)

    # B) crowding + derisk: 因子收益拥挤度高位月降到 low_expo=0.5
    crow = regime.crowding_index(f_df, lookback=12).reindex(idx)
    thr = crow.expanding(min_periods=12).quantile(0.80)
    scale_B = pd.Series(1.0, index=idx)
    scale_B[crow.shift(1) > thr.shift(1)] = 0.5

    # C) combo: 两者取较小 (任一风险高就降)
    scale_C = pd.concat([scale_A, scale_B], axis=1).min(axis=1).reindex(idx).fillna(1.0)

    scales = {"基线(scale=1)": pd.Series(1.0, index=idx),
              "A_vol_target": scale_A, "B_crowding": scale_B, "C_combo": scale_C}

    pd.set_option("display.unicode.east_asian_width", True)
    print("\n" + "=" * 96)
    print("中证1000 指增 (TE=3%, OOS) + 拥挤/波动择时降主动风险 —— 降仓前后对比")
    print("=" * 96)
    print(f"{'信号':<16}{'平均scale':>9}{'超额%(合成)':>12}{'IR':>6}{'TE%':>7}"
          f"{'超额回撤%':>10}{'卡玛':>6}{'超额%(真1000)':>14}{'IR真':>6}{'净超额%':>9}{'换手x':>7}")
    out = {}
    for name, sc in scales.items():
        sc = sc.reindex(idx).clip(upper=1.0).fillna(1.0)
        # scaled 超额 (对两个基准)
        exS = (sc * exS_raw).dropna()
        # 对真实1000: port_real = bench + scale*(port-bench); 减真实1000
        port_real = R["bench"] + sc * exS_raw
        exI = (port_real - R["i1000"]).dropna()
        # 换手: w_real_t = b_t + sc_t*(w_t-b_t)，相邻期半和绝对差
        to_series = []
        wreal_prev = None
        for i in range(len(idx)):
            wr = bs_list[i] + sc.iloc[i] * (ws_list[i] - bs_list[i])
            if wreal_prev is None:
                to = 0.0
            else:
                to = 0.5 * wr.subtract(wreal_prev, fill_value=0).abs().sum()
            to_series.append(to); wreal_prev = wr
        to_ser = pd.Series(to_series, index=idx)
        ann_to = to_ser.mean() * PPY
        # 净超额: 扣换手成本 (相对基线净额, 这里直接在超额上扣 to*C)
        net_exI = (port_real - R["i1000"] - to_ser * C).dropna()

        mS = metrics(exS); mI = metrics(exI)
        out[name] = dict(scale=sc.mean(), exS=mS, exI=mI,
                         net=net_exI.mean() * PPY, to=ann_to)
        print(f"{name:<16}{sc.mean():>9.2f}{mS['ann']*100:>12.1f}{mS['ir']:>6.2f}"
              f"{mS['te']*100:>7.1f}{mS['mdd']*100:>10.1f}{mS['calmar']:>6.2f}"
              f"{mI['ann']*100:>14.1f}{mI['ir']:>6.2f}{out[name]['net']*100:>9.1f}{ann_to:>7.1f}")

    print("\n注: '超额%(合成)' 对优化器控制的市值加权合成基准; '超额%(真1000)' 对真实中证1000指数(000852)")
    print("    净超额% = 对真实1000超额 扣 换手×0.3% 成本后年化; cap=1 表示只降不加杠杆")
    print(f"\n完成 {time.time()-t0:.0f}s")

    # 存净值对比 (基线 vs combo)
    base_sc = scales["基线(scale=1)"].reindex(idx).fillna(1.0)
    best_name = max(["A_vol_target", "B_crowding", "C_combo"], key=lambda k: out[k]["exI"]["calmar"])
    print(f"按对真实1000超额卡玛, 最优降仓信号 = {best_name} "
          f"(卡玛 {out[best_name]['exI']['calmar']:.2f} vs 基线 {out['基线(scale=1)']['exI']['calmar']:.2f})")


if __name__ == "__main__":
    main()
