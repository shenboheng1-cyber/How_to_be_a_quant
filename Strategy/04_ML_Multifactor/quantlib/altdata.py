# -*- coding: utf-8 -*-
"""
quantlib.altdata —— 另类数据：行业分类 + 机构调研 + 专利（PIT 安全）
================================================================
- 行业分类(STK_INDUSTRYCLASS, 申万2021)：用 ImplementDate as-of 对齐 → 终于能做行业中性。
- 机构调研强度(IRM)：过去90天被机构调研家次(DeclareDate 公告日 PIT) → 关注度/情绪, 正向。
- 专利动量(PT_LCDETAIL)：过去365天授权专利数(GrantDate 授权日 PIT) → 创新产出, 正向。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from . import data

RAW = lambda t: os.path.join(data.DATA_ROOT, "raw", f"{t}.parquet")
# 证监会2012：全历史(2010起)、84个大类(制造业细分)，做行业中性最合适。
# 申万2021只到2021起→早期缺;申万细分级有220-337太碎。
STD = "证监会行业分类2012年版"


def load_industry(standard: str = STD) -> pd.DataFrame:
    con = data.connect()
    df = con.sql(f"""
        SELECT Symbol AS stkcd, IndustryName AS industry, CAST(ImplementDate AS DATE) AS impl
        FROM '{RAW('STK_INDUSTRYCLASS')}'
        WHERE IndustryClassificationName='{standard}' AND IndustryName IS NOT NULL
    """).df()
    con.close()
    df["impl"] = df["impl"].astype("datetime64[ns]")
    return df.sort_values("impl")


def attach_industry(panel: pd.DataFrame, standard: str = STD) -> pd.DataFrame:
    """as-of 把 (生效日<=调仓日) 的最近行业分类并入 panel['industry']。"""
    ind = load_industry(standard)
    p = panel.copy()
    p["trddt"] = p["trddt"].astype("datetime64[ns]")
    p = p.sort_values("trddt")
    out = pd.merge_asof(p, ind, left_on="trddt", right_on="impl",
                        by="stkcd", direction="backward")
    out["industry"] = out["industry"].fillna("UNKNOWN")   # 缺分类→单独一组,中性化时不丢行
    return out


def _trailing_count(panel, sub_sql, days, name):
    """每 (stkcd,trddt) 统计 sub_sql(产出 stkcd,dd) 在 (trddt-days, trddt] 的条数。"""
    k = panel[["stkcd", "trddt"]].drop_duplicates().copy()
    k["trddt"] = k["trddt"].astype("datetime64[ns]")
    con = data.connect()
    con.register("k", k)
    df = con.sql(f"""
        SELECT k.stkcd, k.trddt, count(*) AS {name}
        FROM k JOIN ({sub_sql}) e ON k.stkcd = e.stkcd
          AND e.dd > k.trddt - INTERVAL {days} DAY AND e.dd <= k.trddt
        GROUP BY 1, 2
    """).df()
    con.close()
    return df


def attach_altfactors(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["trddt"] = out["trddt"].astype("datetime64[ns]")
    research = f"""SELECT i.Symbol AS stkcd, CAST(rr.DeclareDate AS DATE) AS dd
                   FROM '{RAW('IRM_INSTITUTION')}' i
                   JOIN '{RAW('IRM_RESEARCHINFO')}' rr ON i.ReportID=rr.ReportID
                   WHERE rr.DeclareDate IS NOT NULL"""
    patent = f"""SELECT Symbol AS stkcd, CAST(GrantDate AS DATE) AS dd
                 FROM '{RAW('PT_LCDETAIL')}' WHERE GrantDate IS NOT NULL"""
    r = _trailing_count(out, research, 90, "research90")
    p = _trailing_count(out, patent, 365, "patent365")
    out = out.merge(r, on=["stkcd", "trddt"], how="left").merge(p, on=["stkcd", "trddt"], how="left")
    out["research90"] = out["research90"].fillna(0)
    out["patent365"] = out["patent365"].fillna(0)
    return out


# 因子（值大=预期收益高）
def research_intensity(panel): return np.log1p(panel["research90"])   # 机构关注度(正向)
def patent_mom(panel):        return np.log1p(panel["patent365"])    # 创新产出动量(正向)

REGISTRY = {"a_research": (research_intensity, "机构调研强度"), "a_patent": (patent_mom, "专利动量")}
