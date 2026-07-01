# -*- coding: utf-8 -*-
"""
iFinD HTTP 一次性下载：中证1000(000852) 历史成分股 + 权重
================================================================
指标 ths_index_weight_stock，参数 [日期, '000852.SH']：成员返回权重(%)，非成员返回 None。
中证1000 半年调样(6/12月)，故只在每年 6 月底、12 月底各取一个快照(成分在两次调样间不变)。
每个快照只查市值域候选(总市值 rank 300-2200，必含中证1000的801-1800)，500股/批。

⚠️ 一次性，不进 CSMAR 日更。token 走环境变量 IFIND_RT，不写盘。
产出：<CSMAR>/raw/IFIND_CSI1000_Weights.parquet (trddt, stkcd, weight)

用法：IFIND_RT='你的refresh_token' /opt/anaconda3/bin/python data_fetch/fetch_csi1000_weights.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests, numpy as np, pandas as pd
from quantlib import data

B = "https://quantapi.51ifind.com/api/v1/"
IDX = "000852.SH"


def ifind_code(c):
    c = str(c).zfill(6)
    return c + (".SH" if c[0] == "6" else ".BJ" if c[0] in "489" else ".SZ")


def main():
    rt = os.environ["IFIND_RT"]
    acc = json.loads(requests.post(B + "get_access_token",
                                   headers={"Content-Type": "application/json", "refresh_token": rt}, timeout=20).content)["data"]["access_token"]
    H = {"Content-Type": "application/json", "access_token": acc}
    print("access_token 取得")

    panel = data.load_research_panel("M", "2015-01-01", "2025-12-31")
    panel["trddt"] = pd.to_datetime(panel["trddt"])
    # 快照日：每年 6、12 月最后一个交易日(2018起)
    dts = sorted([d for d in panel["trddt"].unique()
                  if pd.Timestamp(d).month in (6, 12) and pd.Timestamp(d).year >= 2018])
    print(f"快照日 {len(dts)} 个: {pd.Timestamp(dts[0]).date()} ~ {pd.Timestamp(dts[-1]).date()}")

    def query(batch, ds):                                       # 带重试
        for _ in range(4):
            try:
                return json.loads(requests.post(B + "basic_data_service", headers=H,
                    json={"codes": batch, "indipara": [{"indicator": "ths_index_weight_stock", "indiparams": [ds, IDX]}]}, timeout=90).content)
            except Exception:
                time.sleep(2)
        return {"tables": []}

    rows = []
    for dt in dts:
        g = panel[panel["trddt"] == dt]                          # 查全市场,确保不漏成员
        codes = [ifind_code(c) for c in g["stkcd"].unique()]
        ds = pd.Timestamp(dt).strftime("%Y-%m-%d")
        got = 0
        for i in range(0, len(codes), 500):
            j = query(",".join(codes[i:i + 500]), ds)
            for tb in j.get("tables", []):
                w = tb.get("table", {}).get("ths_index_weight_stock", [None])[0]
                if w is not None:
                    rows.append({"trddt": dt, "stkcd": str(tb["thscode"])[:6], "weight": float(w)})
                    got += 1
            time.sleep(0.25)
        print(f"  {ds}: 成分 {got} 只 (候选{len(codes)})", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(data.DATA_ROOT, "raw", "IFIND_CSI1000_Weights.parquet")
    df.to_parquet(out)
    print(f"\n完成：{len(df)} 行 → {out}")
    print(df.groupby("trddt").size().to_string())


if __name__ == "__main__":
    main()
