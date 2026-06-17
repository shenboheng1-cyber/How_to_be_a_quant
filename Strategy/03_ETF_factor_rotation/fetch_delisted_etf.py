"""拉取已摘牌/清盘 ETF 的名单 + 净值(补幸存者偏差用)。
在【你自己的终端】运行(本环境 DNS 黑洞了 Tushare):

    export TUSHARE_TOKEN=你的token   # 或已存在 /tmp/ts_token
    python3 fetch_delisted_etf.py

产出:
    outputs_survivorship/delisted_etf_basic.csv   已摘牌ETF名单
    outputs_survivorship/delisted_etf_nav.parquet [fund_code(6位), date, cum_nav]  供回测合并
"""
from __future__ import annotations
import os, time, sys
from pathlib import Path
import pandas as pd
import tushare as ts

OUT = Path(__file__).resolve().parent / "outputs_survivorship"
OUT.mkdir(exist_ok=True)


def get_token() -> str:
    t = os.getenv("TUSHARE_TOKEN", "").strip()
    if not t and Path("/tmp/ts_token").exists():
        t = Path("/tmp/ts_token").read_text().strip()
    if not t:
        sys.exit("未找到 token: 请 export TUSHARE_TOKEN=... 或写入 /tmp/ts_token")
    return t


def call(fn, _tries=5, _wait=3, **kw):
    last = None
    for a in range(_tries):
        try:
            return fn(**kw)
        except Exception as e:                       # noqa: BLE001
            last = e; time.sleep(_wait * (a + 1))
    raise RuntimeError(f"failed: {last}")


def main():
    ts.set_token(get_token())
    pro = ts.pro_api()

    # 1) 全部交易所基金(含已摘牌 D)
    parts = []
    for st in ("L", "D"):
        df = call(pro.fund_basic, market="E", status=st)
        df["status"] = st; parts.append(df)
        print(f"fund_basic status={st}: {len(df)}")
    allf = pd.concat(parts, ignore_index=True)
    etf = allf[allf["name"].astype(str).str.contains("ETF", case=False, na=False)].copy()
    delisted = etf[etf["status"] == "D"].copy()
    delisted.to_csv(OUT / "delisted_etf_basic.csv", index=False)
    print(f"ETF 总 {len(etf)} | 已摘牌 {len(delisted)}")

    # 2) 已摘牌 ETF 的净值(用 accum_nav 累计净值, 对应 nav_store 的 cum_nav)
    rows = []
    codes = delisted["ts_code"].dropna().tolist()
    for i, ts_code in enumerate(codes):
        try:
            nav = call(pro.fund_nav, ts_code=ts_code, _tries=4, _wait=2)
        except Exception as e:                       # noqa: BLE001
            print(f"  skip {ts_code}: {str(e)[:40]}"); continue
        if nav is None or nav.empty:
            continue
        col = "accum_nav" if "accum_nav" in nav.columns else ("adj_nav" if "adj_nav" in nav.columns else "unit_nav")
        sub = nav[["nav_date", col]].rename(columns={"nav_date": "date", col: "cum_nav"})
        sub["fund_code"] = ts_code.split(".")[0]
        rows.append(sub)
        if (i + 1) % 25 == 0:
            print(f"  nav {i+1}/{len(codes)}")
        time.sleep(0.35)                             # 限速
    if rows:
        navdf = pd.concat(rows, ignore_index=True)
        navdf["date"] = pd.to_datetime(navdf["date"], format="%Y%m%d", errors="coerce")
        navdf = navdf.dropna(subset=["date", "cum_nav"]).sort_values(["fund_code", "date"])
        navdf.to_parquet(OUT / "delisted_etf_nav.parquet", index=False)
        print(f"已存 delisted_etf_nav.parquet: {navdf.shape} | ETF {navdf['fund_code'].nunique()} | "
              f"{navdf['date'].min().date()}~{navdf['date'].max().date()}")
    else:
        print("无已摘牌 ETF 净值(可能该账号无 fund_nav 权限或无数据)")


if __name__ == "__main__":
    main()
