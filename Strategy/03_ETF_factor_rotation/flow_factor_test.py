"""拉全市场 ETF 场内流通份额(ths_inner_float_shares_fund)→ 构造资金流因子 → 测 Rank IC + 相关性。
先验证资金流是否有横截面预测力(值不值得进一步做 walk-forward)。"""
from __future__ import annotations
import os
import sqlite3, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from etf_factor_strategy.engine import compute_factor_panel, _datewise_z
import hfq_common as H

RT = os.environ.get("IFIND_REFRESH_TOKEN", "")  # 从环境变量读取,勿硬编码 refresh_token
BASE = "https://quantapi.51ifind.com/api/v1"
DB = H.DEFAULT_DATA_DIR / "etf_share_ifind.db"
IND = "ths_inner_float_shares_fund"


def token():
    return (requests.post(f"{BASE}/get_access_token",
            headers={"Content-Type": "application/json", "refresh_token": RT}, timeout=60).json()
            .get("data") or {}).get("access_token")


def pull_shares(codes, tok):
    """date_sequence 拉份额, 落地 DB, 返回 date×fund_code 面板。断点续跑。"""
    done = set()
    try:
        con = sqlite3.connect(DB)
        done = set(pd.read_sql_query("SELECT DISTINCT fund_code FROM share", con)["fund_code"]); con.close()
    except Exception:
        pass
    todo = [c for c in codes if c.split(".")[0] not in done]
    H_ = {"Content-Type": "application/json", "access_token": tok}
    con = sqlite3.connect(DB)
    try:
        for i in range(0, len(todo), 12):
            batch = todo[i:i + 12]
            p = {"codes": ",".join(batch), "startdate": "2017-01-01", "enddate": "2026-06-12",
                 "functionpara": {"Interval": "D", "Fill": "Blank"},
                 "indipara": [{"indicator": IND, "indicatorparams": [""]}]}
            time.sleep(0.13)
            j = requests.post(f"{BASE}/date_sequence", json=p, headers=H_, timeout=120).json()
            rows = []
            for t in (j.get("tables") or []):
                code = t.get("thscode", "").split(".")[0]; tm = t.get("time") or []
                vals = (t.get("table") or {}).get(IND) or []
                for d, v in zip(tm, vals):
                    if v is not None:
                        rows.append((code, d, float(v)))
            if rows:
                pd.DataFrame(rows, columns=["fund_code", "date", "share"]).to_sql("share", con, if_exists="append", index=False)
            if i % 120 == 0:
                print(f"  pulled {i+len(batch)}/{len(todo)}")
    finally:
        con.close()


def load_share_panel():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql_query("SELECT fund_code,date,share FROM share", con, parse_dates=["date"]); con.close()
    df["fund_code"] = df["fund_code"].astype(str).str.zfill(6)
    return df.pivot_table(index="date", columns="fund_code", values="share").sort_index()


def rank_ic(fac_me, ret_me):
    out = {}
    for d in fac_me.index:
        f, r = fac_me.loc[d], ret_me.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= 10:
            out[d] = f[m].rank().corr(r[m].rank())
    return pd.Series(out).sort_index()


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, _, _ = H.load_hfq(); px = px.loc[:H.END]
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    codes = [c + (".SH" if c.startswith("5") else ".SZ") for c in px.columns]
    print(f"拉 {len(codes)} 只 ETF 场内份额…")
    pull_shares(codes, token())

    share = load_share_panel().reindex(index=px.index).ffill(limit=5)
    share = share.reindex(columns=px.columns)
    print(f"份额面板: {share.shape}, 覆盖 ETF {share.notna().any().sum()} 只")

    # 资金流因子: 份额增长率(scale-free), 多窗口
    flow = {f"share_growth_{w}": share / share.shift(w) - 1.0 for w in (20, 60, 120)}

    # 月末 + 下期收益
    me = [d for d in px.index.to_series().groupby(px.index.to_period("M")).max()
          if pd.Timestamp(H.START) <= d <= pd.Timestamp(H.END)]
    me_idx = pd.DatetimeIndex(me)
    fwd = px.reindex(me_idx).shift(-1) / px.reindex(me_idx) - 1.0

    print("\n=== 资金流因子 Rank IC (月度, 2018-2026) ===")
    for name, fpanel in flow.items():
        ic = rank_ic(fpanel.reindex(me_idx), fwd)
        n = len(ic); mean = ic.mean(); std = ic.std(ddof=1)
        print(f"  {name:18} IC均值{mean:+.3f} ICIR{mean/std:+.3f} IC>0占比{(ic>0).mean():.0%} t{mean/std*np.sqrt(n):+.2f} n={n}")

    # 与现有因子的相关性(确认正交)
    fac = compute_factor_panel(px); fac["date"] = pd.to_datetime(fac["date"])
    fac_me = fac[fac["date"].isin(me_idx)]
    base = {f: fac_me.pivot_table(index="date", columns="fund_code", values=f).reindex(me_idx)
            for f in ["combo_eff_accel", "momentum_12_1"]}
    sg = flow["share_growth_60"].reindex(me_idx)
    print("\n=== share_growth_60 与现有因子 截面相关(均值) ===")
    for f, p in base.items():
        cs = []
        for d in me_idx:
            x, y = sg.loc[d], p.loc[d]; m = x.notna() & y.notna()
            if m.sum() > 20:
                cs.append(x[m].rank().corr(y[m].rank()))
        print(f"  vs {f:18} {np.nanmean(cs):+.2f}")


if __name__ == "__main__":
    main()
