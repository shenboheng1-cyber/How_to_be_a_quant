"""因子研究: 复现用户4个新因子 + 拆子区间(2019-2024 vs 2023-2024 vs 2025), 找能补前期收益的因子。

关键: 多空全期收益会被 2025 主导; 真正该看每个因子在 2023-2024(前期趴平期)的多空收益。
全 PIT(只用<=t)。月度, 五分组多空 Q1-Q5。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from etf_factor_strategy.data import load_etf_universe, load_nav_prices, DEFAULT_DATA_DIR
from etf_factor_strategy.engine import compute_factor_panel, _datewise_z


def month_ends(idx):
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).max().values)


def factor_panels(prices):
    """返回 {factor: 宽panel(date×etf)}, 全部 PIT。"""
    px = prices.sort_index().astype(float)
    dret = px.pct_change(fill_method=None)
    pool = dret.mean(axis=1)                                   # 等权ETF池日收益
    out = {}
    # 现有动量(用引擎, 取综合 combo 三因子的合成分代理: 这里单独要三因子,改取long格式合成)
    # 新因子:
    out["dd_resilience_252"] = px / px.rolling(252, min_periods=120).max() - 1.0
    # 滚动相关/beta: 在月末算, 先按日算滚动会重; 用 pandas rolling cov/var
    win = 120
    cov = dret.rolling(win, min_periods=80).cov(pool)
    varp = pool.rolling(win, min_periods=80).var()
    beta = cov.div(varp, axis=0)
    ret120_etf = px / px.shift(win) - 1.0
    ret120_pool = (1 + pool).rolling(win).apply(np.prod, raw=True) - 1.0
    out["resid_mom_120"] = ret120_etf.sub(beta.mul(ret120_pool, axis=0))
    # 相关
    std_e = dret.rolling(win, min_periods=80).std()
    corr = cov.div(std_e.mul(pool.rolling(win, min_periods=80).std(), axis=0))
    out["low_corr_120"] = -corr
    # 下跌日 beta
    downmask = pool < 0
    dret_d = dret.where(downmask)
    pool_d = pool.where(downmask)
    cov_d = dret_d.rolling(win, min_periods=40).cov(pool_d)
    var_d = pool_d.rolling(win, min_periods=40).var()
    out["high_down_beta_120"] = cov_d.div(var_d, axis=0)
    return out, px


def eval_factor(fac_panel, mret, me, lo, hi):
    """月度五分组多空 Q1-Q5 (Q1=因子最高). 返回区间年化多空 + Rank IC。"""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    seg_ls, seg_ic, top = [], [], []
    for t in me:
        if t not in fac_panel.index:
            sub = fac_panel.loc[:t]
            if sub.empty: continue
            f = sub.iloc[-1]
        else:
            f = fac_panel.loc[t]
        f = f.dropna()
        nxt = mret.index[mret.index > t]
        if len(nxt) == 0 or len(f) < 25:
            continue
        fwd = mret.loc[nxt[0]].reindex(f.index)
        d = pd.DataFrame({"f": f, "r": fwd}).dropna()
        if len(d) < 25 or not (lo <= t <= hi):
            continue
        q = pd.qcut(d["f"], 5, labels=False, duplicates="drop")
        ls = d["r"][q == 4].mean() - d["r"][q == 0].mean()
        seg_ls.append(ls)
        seg_ic.append(d["f"].corr(d["r"], method="spearman"))
        top.append(d["r"][q == 4].mean())
    if not seg_ls:
        return None
    return dict(ann_ls=float(np.mean(seg_ls) * 12), ic=float(np.nanmean(seg_ic)),
                ann_top=float(np.mean(top) * 12), n=len(seg_ls))


def main():
    uni = load_etf_universe(data_dir=DEFAULT_DATA_DIR)
    px = load_nav_prices(uni["fund_code"].tolist(), data_dir=DEFAULT_DATA_DIR,
                         start="2017-01-01", end="2026-06-05").dropna(axis=1, thresh=280)
    panels, px = factor_panels(px)
    # 现有三因子合成分(用引擎)
    fac_long = compute_factor_panel(px)
    sc = pd.Series(0.0, index=fac_long.index)
    for f, w in {"combo_eff_accel": 0.45, "momentum_12_1": 0.35, "fund_hit_rate_20": 0.20}.items():
        sc = sc.add(_datewise_z(fac_long, f).fillna(0.0) * w, fill_value=0.0)
    fac_long["score3"] = sc
    score3_panel = fac_long.pivot_table(index="date", columns="fund_code", values="score3")
    score3_panel.index = pd.to_datetime(score3_panel.index)
    panels["现有三因子"] = score3_panel

    mret = px.resample("ME").last().pct_change(fill_method=None)
    me = month_ends(px.index)

    print(f"{'因子':20}{'IC(19-24)':>11}{'多空年化19-24':>14}{'多空年化23-24':>14}{'多空年化2025':>13}{'多头年化23-24':>14}")
    for name, panel in panels.items():
        a = eval_factor(panel, mret, me, "2019-01-01", "2024-12-31")
        b = eval_factor(panel, mret, me, "2023-01-01", "2024-12-31")
        c = eval_factor(panel, mret, me, "2025-01-01", "2026-06-05")
        if a is None:
            print(f"{name:20} 数据不足"); continue
        print(f"{name:20}{a['ic']:>11.3f}{a['ann_ls']:>14.2%}{(b['ann_ls'] if b else float('nan')):>14.2%}"
              f"{(c['ann_ls'] if c else float('nan')):>13.2%}{(b['ann_top'] if b else float('nan')):>14.2%}")

    # 因子间相关(月末因子值的截面 rank 相关, 全期平均)
    print("\n因子相关(与现有三因子合成分):")
    s3 = panels["现有三因子"]
    for name, panel in panels.items():
        if name == "现有三因子": continue
        cs = []
        for t in me:
            x = panel.loc[:t].iloc[-1] if t not in panel.index else panel.loc[t]
            y = s3.loc[:t].iloc[-1] if t not in s3.index else s3.loc[t]
            d = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(d) > 25: cs.append(d["x"].corr(d["y"], method="spearman"))
        print(f"  {name:20} {np.nanmean(cs):+.2f}")


if __name__ == "__main__":
    main()
