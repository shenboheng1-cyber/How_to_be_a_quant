"""拉价值/基本面数据:ETF→跟踪指数映射 + 指数 PE/PB/股息率历史 → 建价值因子 → 测 IC/相关。"""
from __future__ import annotations
import os
import sqlite3, time
import numpy as np
import pandas as pd
import requests

from etf_factor_strategy.engine import compute_factor_panel, _datewise_z
import hfq_common as H

RT = os.environ.get("IFIND_REFRESH_TOKEN", "")  # 从环境变量读取,勿硬编码 refresh_token
BASE = "https://quantapi.51ifind.com/api/v1"
DB = H.DEFAULT_DATA_DIR / "etf_value_ifind.db"
TOK = {"t": None}


def tok():
    if TOK["t"] is None:
        TOK["t"] = (requests.post(f"{BASE}/get_access_token", headers={"Content-Type": "application/json", "refresh_token": RT}, timeout=60).json().get("data") or {}).get("access_token")
    return TOK["t"]


def H_():
    return {"Content-Type": "application/json", "access_token": tok()}


def safe_post(endpoint, payload, tries=5):
    for k in range(tries):
        try:
            r = requests.post(f"{BASE}/{endpoint}", json=payload, headers=H_(), timeout=120)
            j = r.json()
            ec = j.get("errorcode", j.get("errcode", 0))
            if ec in (-1300, -1302):
                TOK["t"] = None; time.sleep(2); continue
            return j
        except Exception:
            time.sleep(5 * (k + 1))
    return {"tables": []}


def idx_suffix(code):
    c = str(code)
    if c[:1] in "0" and c[:3] not in ("399",):
        return ".SH"
    if c[:3] == "399" or c[:1] == "1" or c[:2] in ("15",):
        return ".SZ"
    if c[:1] in "9" or c[:1].lower() == "h":
        return ".CSI"
    if c[:1] == "6":
        return ".SH"
    return ".CSI"


def get_mapping(etfs):
    """ETF -> 跟踪指数代码 (basic_data)。"""
    out = {}
    for i in range(0, len(etfs), 20):
        b = etfs[i:i + 20]
        time.sleep(0.12)
        j = requests.post(f"{BASE}/basic_data_service", json={"codes": ",".join(b),
            "indipara": [{"indicator": "ths_tracking_index_code_fund", "indicatorparams": [""]}]}, headers=H_(), timeout=60).json()
        for t in (j.get("tables") or []):
            v = (t.get("table") or {}).get("ths_tracking_index_code_fund")
            if v and v[0]:
                out[t["thscode"].split(".")[0]] = str(v[0]).split(".")[0]
    return out


def suffix_order(code):
    c = str(code); base = [".SH", ".SZ", ".CSI"]
    if c[:3] == "399" or c[:2] == "15": return [".SZ", ".SH", ".CSI"]
    if c[:1].lower() == "h" or c[:1] == "9": return [".CSI", ".SH", ".SZ"]
    return base


def _pe_ok(j):
    t = (j.get("tables") or [None])[0]
    return t is not None and any(x is not None for x in (t.get("table") or {}).get("pe_ttm_index", []))


def pull_index_val(index_codes):
    """增量+可断点续跑:逐代码试后缀,拉到就立刻写库;已处理的(含无PE)跳过。"""
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS index_val (index_code TEXT, date TEXT, pe REAL, pb REAL, divyield REAL)")
    con.execute("CREATE TABLE IF NOT EXISTS tried (index_code TEXT PRIMARY KEY)")
    con.commit()
    done = set(pd.read_sql_query("SELECT index_code FROM tried", con)["index_code"].astype(str))
    todo = [c for c in index_codes if c not in done]
    print(f"  待拉 {len(todo)} / {len(index_codes)}（已处理 {len(done)}）")
    for i, code in enumerate(todo):
        jp = None; suf = None
        for s in suffix_order(code):
            time.sleep(0.13)
            cand = safe_post("cmd_history_quotation", {"codes": code + s,
                "indicators": "pe_ttm_index,pb_mrq", "startdate": "2016-01-01", "enddate": "2026-06-12",
                "functionpara": {"Fill": "Blank"}})
            if _pe_ok(cand):
                jp, suf = cand, s; break
        rows = []
        if jp is not None:
            time.sleep(0.13)
            jd = safe_post("date_sequence", {"codes": code + suf,
                "startdate": "2016-01-01", "enddate": "2026-06-12", "functionpara": {"Interval": "D", "Fill": "Blank"},
                "indipara": [{"indicator": "ths_dividend_rate_index", "indicatorparams": [""]}]})
            t0 = jp["tables"][0]; tm = t0.get("time") or []; tab = t0.get("table") or {}
            pe = tab.get("pe_ttm_index") or []; pb = tab.get("pb_mrq") or []
            td = (jd.get("tables") or [{}])[0]
            dvm = dict(zip(td.get("time") or [], (td.get("table") or {}).get("ths_dividend_rate_index") or []))
            rows = [(code, d, pe[k] if k < len(pe) else None, pb[k] if k < len(pb) else None, dvm.get(d))
                    for k, d in enumerate(tm)]
        if rows:
            con.executemany("INSERT INTO index_val VALUES (?,?,?,?,?)", rows)
        con.execute("INSERT OR IGNORE INTO tried VALUES (?)", (code,))
        con.commit()
        if i % 25 == 0:
            n = con.execute("SELECT COUNT(DISTINCT index_code) FROM index_val").fetchone()[0]
            print(f"  指数 {i}/{len(todo)} (有PE指数 {n})")
    df = pd.read_sql_query("SELECT * FROM index_val", con); con.close()
    return df


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, _, _ = H.load_hfq(); px = px.loc[:H.END]
    try:
        mp = dict(pd.read_sql_query("SELECT fund_code,index_code FROM mapping", sqlite3.connect(DB)).astype(str).values)
        print(f"复用已存映射: {len(mp)} 只 ETF→指数")
    except Exception:
        etfs = [c + (".SH" if c.startswith("5") else ".SZ") for c in px.columns]
        print(f"拉 {len(etfs)} 只 ETF 的跟踪指数映射…")
        mp = get_mapping(etfs)
        pd.DataFrame([(k, v) for k, v in mp.items()], columns=["fund_code", "index_code"]).to_sql(
            "mapping", sqlite3.connect(DB), if_exists="replace", index=False)
    print(f"  唯一指数 {len(set(mp.values()))} 个")
    print("拉指数 PE/PB/股息率历史…")
    iv = pull_index_val(sorted(set(mp.values())))
    iv["date"] = pd.to_datetime(iv["date"])
    cov = iv.groupby("index_code")["pe"].apply(lambda s: s.notna().sum())
    print(f"  指数估值落地: {len(iv)} 行, 有PE的指数 {int((cov>100).sum())} 个")

    # 构造价值因子: 估值相对自身历史(便宜=高分) + 股息率
    inv2etf = {}
    for fc, ic in mp.items():
        inv2etf.setdefault(ic, []).append(fc)
    pe_p = iv.pivot_table(index="date", columns="index_code", values="pe").reindex(px.index).ffill(limit=10)
    div_p = iv.pivot_table(index="date", columns="index_code", values="divyield").reindex(px.index).ffill(limit=10)
    # 每指数: PE 相对自身2年历史的 z (低=便宜=高分) → 映到 ETF
    pe_z = -(pe_p - pe_p.rolling(504, min_periods=250).mean()) / pe_p.rolling(504, min_periods=250).std()
    val_etf = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    dvy_etf = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
    for ic, fcs in inv2etf.items():
        if ic in pe_z.columns:
            for fc in fcs:
                if fc in val_etf.columns:
                    val_etf[fc] = pe_z[ic]; dvy_etf[fc] = div_p[ic] if ic in div_p.columns else np.nan

    # 月末 IC vs 下期收益
    me = [d for d in px.index.to_series().groupby(px.index.to_period("M")).max()
          if pd.Timestamp(H.START) <= d <= pd.Timestamp(H.END)]
    me_idx = pd.DatetimeIndex(me)
    fwd = px.reindex(me_idx).shift(-1) / px.reindex(me_idx) - 1.0

    def rank_ic(fp):
        fp = fp.reindex(me_idx); out = {}
        for d in me_idx:
            f, r = fp.loc[d], fwd.loc[d]; m = f.notna() & r.notna()
            if m.sum() >= 10: out[d] = f[m].rank().corr(r[m].rank())
        return pd.Series(out)
    print("\n=== 价值因子 Rank IC (月度, 2018-2026) ===")
    for name, fp in [("估值便宜(PE vs 自身历史)", val_etf), ("股息率", dvy_etf)]:
        ic = rank_ic(fp); n = len(ic); mn, sd = ic.mean(), ic.std(ddof=1)
        print(f"  {name:24} IC{mn:+.3f} ICIR{mn/sd:+.2f} IC>0占比{(ic>0).mean():.0%} t{mn/sd*np.sqrt(n):+.2f} 覆盖ETF{int(fp.notna().any().sum())}")

    # 与动量相关
    fac = compute_factor_panel(px); fac["date"] = pd.to_datetime(fac["date"])
    mom = fac[fac["date"].isin(me_idx)].pivot_table(index="date", columns="fund_code", values="momentum_12_1").reindex(me_idx)
    vz = val_etf.reindex(me_idx)
    cs = [vz.loc[d][vz.loc[d].notna() & mom.loc[d].notna()].rank().corr(mom.loc[d][vz.loc[d].notna() & mom.loc[d].notna()].rank()) for d in me_idx]
    print(f"\n价值 vs 动量 截面相关(均值): {np.nanmean(cs):+.2f}  (越负越好,说明互补)")


if __name__ == "__main__":
    main()
