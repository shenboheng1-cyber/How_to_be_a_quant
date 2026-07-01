# -*- coding: utf-8 -*-
"""
quantlib.fundamentals —— 基本面因子层（年报口径，PIT 安全）
================================================================
从 CSMAR 三大报表(raw/FS_*.parquet)取年报数据，按【法定披露截止日】做 PIT 对齐，
as-of 合并到月度面板，构造 价值/质量/成长/投资/应计 因子。

== PIT 关键(本库的坑)==
- FS_* 的 DeclareDate 只在重述行(IfCorrect=1)有值=更正日，不是首次公告日 → 不可用。
- 用法定披露截止日保守对齐：年报(12-31)→次年4-30。company早披露也当作截止日才可用(绝不前视)。
- IfCorrect=0 取首次披露口径；Typrep=A 用合并报表。

科目编码(已用茅台核对)：营收 B001101000 / 归母净利 B002000101 /
资产 A001000000 / 归母权益 A003100000 / 经营现金流 C001000000。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from . import data

RAW = lambda t: os.path.join(data.DATA_ROOT, "raw", f"{t}.parquet")


def load_annual(start_year: int = 2010) -> pd.DataFrame:
    """取年报三表，算 YoY 与可用日(avail_date)。返回每股每年一行。"""
    con = data.connect()
    sql = f"""
        SELECT i.Stkcd AS stkcd, i.Accper AS accper,
               TRY_CAST(i.B001101000 AS DOUBLE) AS revenue,
               TRY_CAST(i.B001201000 AS DOUBLE) AS opcost,
               TRY_CAST(i.B002000101 AS DOUBLE) AS net_profit,
               TRY_CAST(b.A001000000 AS DOUBLE) AS assets,
               TRY_CAST(b.A002000000 AS DOUBLE) AS liabilities,
               TRY_CAST(b.A003100000 AS DOUBLE) AS equity,
               TRY_CAST(c.C001000000 AS DOUBLE) AS cfo
        FROM '{RAW('FS_Comins')}' i
        JOIN '{RAW('FS_Combas')}' b
          ON i.Stkcd=b.Stkcd AND i.Accper=b.Accper AND i.Typrep=b.Typrep AND b.IfCorrect=0
        JOIN '{RAW('FS_Comscfd')}' c
          ON i.Stkcd=c.Stkcd AND i.Accper=c.Accper AND i.Typrep=c.Typrep AND c.IfCorrect=0
        WHERE i.Typrep='A' AND i.IfCorrect=0 AND i.Accper LIKE '%-12-31'
          AND CAST(i.Accper[1:4] AS INT) >= {start_year}
    """
    df = con.sql(sql).df()
    con.close()

    df["fyear"] = df["accper"].str[:4].astype(int)
    # 法定 PIT：年报次年 4-30 截止 → 用 5-01 保守可用
    df["avail_date"] = pd.to_datetime((df["fyear"] + 1).astype(str) + "-05-01")
    df = df.sort_values(["stkcd", "fyear"]).reset_index(drop=True)

    # 上一年(算 YoY 成长 / 资产增长)
    g = df.groupby("stkcd")
    for col in ["revenue", "net_profit", "assets", "equity"]:
        df[col + "_prev"] = g[col].shift(1)
    return df


def attach(panel: pd.DataFrame, fund: pd.DataFrame | None = None) -> pd.DataFrame:
    """把 PIT 基本面 as-of 合并到月度面板(每股取 avail_date<=调仓日 的最近一期年报)。"""
    if fund is None:
        fund = load_annual()
    p = panel.copy()
    p["trddt"] = p["trddt"].astype("datetime64[ns]")
    p = p.sort_values("trddt").reset_index(drop=True)
    f = fund.copy()
    f["avail_date"] = f["avail_date"].astype("datetime64[ns]")
    f = f.sort_values("avail_date").reset_index(drop=True)
    cols = ["stkcd", "avail_date", "fyear", "revenue", "opcost", "net_profit", "assets",
            "liabilities", "equity", "cfo", "revenue_prev", "net_profit_prev", "assets_prev"]
    out = pd.merge_asof(p, f[cols], left_on="trddt", right_on="avail_date",
                        by="stkcd", direction="backward")
    return out


# ---------- 因子（值越大=预期收益越高；panel 需含 total_mktcap 与上面合并的基本面列）----------
def ep(panel):    return panel["net_profit"] / panel["total_mktcap"]          # 盈利收益率(价值)
def bp(panel):    return panel["equity"] / panel["total_mktcap"]              # 账面市值比(价值)
def sp(panel):    return panel["revenue"] / panel["total_mktcap"]            # 销售收益率(价值)
def cfp(panel):   return panel["cfo"] / panel["total_mktcap"]               # 现金流市值比(价值)
def roe(panel):   return panel["net_profit"] / panel["equity"]              # 质量
def roa(panel):   return panel["net_profit"] / panel["assets"]              # 质量
def gross_cfo(panel): return panel["cfo"] / panel["net_profit"].abs().clip(lower=1) * np.sign(panel["net_profit"])  # 盈余含金量
def rev_growth(panel): return panel["revenue"] / panel["revenue_prev"] - 1   # 成长
def profit_growth(panel): return panel["net_profit"] / panel["net_profit_prev"].abs().clip(lower=1) - 1  # 成长
def asset_growth(panel): return -(panel["assets"] / panel["assets_prev"] - 1)  # 投资异象(资产扩张→负向)
def accruals(panel):  return -((panel["net_profit"] - panel["cfo"]) / panel["assets"])  # 应计(Sloan,负向)
# 质量(大盘最稳，2026-06新增)
def gross_prof(panel):   return (panel["revenue"] - panel["opcost"]) / panel["assets"]      # Novy-Marx 毛利资产比(最强质量)
def gross_margin(panel): return (panel["revenue"] - panel["opcost"]) / panel["revenue"].abs().clip(lower=1)  # 毛利率
def low_lev(panel):      return -(panel["liabilities"] / panel["assets"])                   # 低杠杆(负债率负向)
def asset_turn(panel):   return panel["revenue"] / panel["assets"]                          # 资产周转率(运营效率)

REGISTRY = {
    "f_ep": (ep, "盈利收益率EP"), "f_bp": (bp, "账面市值比BP"), "f_sp": (sp, "销售收益率SP"),
    "f_cfp": (cfp, "现金流市值比"), "f_roe": (roe, "ROE质量"), "f_roa": (roa, "ROA质量"),
    "f_gross_cfo": (gross_cfo, "盈余含金量"), "f_rev_growth": (rev_growth, "营收增速"),
    "f_profit_growth": (profit_growth, "利润增速"), "f_asset_growth": (asset_growth, "资产增长(负)"),
    "f_accruals": (accruals, "应计(负)"),
    "f_gross_prof": (gross_prof, "毛利资产比GP/A"), "f_gross_margin": (gross_margin, "毛利率"),
    "f_low_lev": (low_lev, "低杠杆"), "f_asset_turn": (asset_turn, "资产周转率"),
}
