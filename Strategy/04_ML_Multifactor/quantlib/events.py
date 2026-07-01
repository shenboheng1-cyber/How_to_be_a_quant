# -*- coding: utf-8 -*-
"""
quantlib.events —— 避雷 / 事件因子（PIT 安全）
================================================================
4 个事件型因子，全部按真实时间对齐，绝不前视：
  viol      违规处罚滚动12月计数（DeclareDate 公告日，100%覆盖）→ 负向避雷
  lockup    未来60日限售解禁占比（CirculationDate 未来日，天然前瞻安全）→ 负向
  fcst_sue  业绩预告净利增速（PubliDate 公告日，过去180日内最新一条）→ 正向
  earn_mgmt 可操控应计 DisAcc（AIQ 年频，法定披露截止日PIT）→ 负向避雷

构造方式：违规/解禁=区间聚合(DuckDB range-join)；预告/盈余管理=as-of。
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from . import data

RAW = lambda t: os.path.join(data.DATA_ROOT, "raw", f"{t}.parquet")


def _keys(panel):
    k = panel[["stkcd", "trddt"]].drop_duplicates().copy()
    k["trddt"] = k["trddt"].astype("datetime64[ns]")
    return k


def _off(days):
    if days == 0: return "k.trddt"
    return f"k.trddt - INTERVAL {abs(days)} DAY" if days < 0 else f"k.trddt + INTERVAL {days} DAY"


def _range_agg(keys, table, datecol, expr, lo, hi):
    """对每个 (stkcd,trddt)，聚合 table 中 datecol 落在 (trddt+lo, trddt+hi] 的记录。"""
    con = data.connect()
    con.register("k", keys)
    df = con.sql(f"""
        SELECT k.stkcd, k.trddt, {expr} AS v
        FROM k JOIN '{RAW(table)}' t
          ON k.stkcd = t.Symbol
         AND CAST(t.{datecol} AS DATE) >  {_off(lo)}
         AND CAST(t.{datecol} AS DATE) <= {_off(hi)}
        GROUP BY 1, 2
    """).df()
    con.close()
    return df


def attach_events(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["trddt"] = out["trddt"].astype("datetime64[ns]")
    k = _keys(out)

    # 违规：过去365天处罚次数（无事件=0）
    v = _range_agg(k, "STK_Violation_Main", "DeclareDate", "count(*)", -365, 0)
    out = out.merge(v.rename(columns={"v": "viol_count"}), on=["stkcd", "trddt"], how="left")
    out["viol_count"] = out["viol_count"].fillna(0)

    # 解禁：未来60天解禁占比之和（无=0）
    lk = _range_agg(k, "HLD_LockShares_Detail", "CirculationDate",
                    "sum(TRY_CAST(t.Proportion AS DOUBLE))", 0, 60)
    out = out.merge(lk.rename(columns={"v": "lockup_prop"}), on=["stkcd", "trddt"], how="left")
    out["lockup_prop"] = out["lockup_prop"].fillna(0)

    # 业绩预告 SUE：过去180天最新一条的预告净利增速中值（as-of + tolerance）
    con = data.connect()
    fc = con.sql(f"""
        SELECT StockCode AS stkcd, CAST(PubliDate AS DATE) AS pub,
               (TRY_CAST(RatNetProf_Low AS DOUBLE) + TRY_CAST(RatNetProf_Hig AS DOUBLE))/2 AS g
        FROM '{RAW('FIN_F_ForecFin')}'
        WHERE PubliDate IS NOT NULL
          AND (RatNetProf_Low IS NOT NULL OR RatNetProf_Hig IS NOT NULL)
    """).df()
    # 盈余管理 DisAcc：取最全样本口径，年频
    em = con.sql(f"""
        SELECT Symbol AS stkcd, CAST(EndDate AS DATE) AS enddt, TRY_CAST(DisAcc AS DOUBLE) AS disacc
        FROM '{RAW('AIQ_AccEarManIndexMJonesY')}'
        WHERE ISBSE=0 AND ST=0 AND IsNewOrSuspend=0
    """).df()
    con.close()

    fc["pub"] = fc["pub"].astype("datetime64[ns]")
    fc = fc.dropna(subset=["g"]).sort_values("pub")
    out = pd.merge_asof(out.sort_values("trddt"), fc.sort_values("pub"),
                        left_on="trddt", right_on="pub", by="stkcd",
                        direction="backward", tolerance=pd.Timedelta("180D"))
    out = out.rename(columns={"g": "fcst_growth"})

    em["avail"] = (em["enddt"].dt.year + 1).astype(str) + "-05-01"
    em["avail"] = em["avail"].astype("datetime64[ns]")
    em = em.dropna(subset=["disacc"]).sort_values("avail")
    out = pd.merge_asof(out.sort_values("trddt"), em[["stkcd", "avail", "disacc"]].sort_values("avail"),
                        left_on="trddt", right_on="avail", by="stkcd", direction="backward")
    return out.reset_index(drop=True)


# ---------- 因子（值越大=预期收益越高）----------
def viol(panel):      return -panel["viol_count"]              # 违规越多越差
def lockup(panel):    return -panel["lockup_prop"]             # 临近解禁卖压
def fcst_sue(panel):  return panel["fcst_growth"]              # 预告增速越高越好
def earn_mgmt(panel): return -panel["disacc"]                 # 可操控应计越高越差(避雷)

REGISTRY = {
    "e_viol": (viol, "违规处罚(负)"), "e_lockup": (lockup, "临近解禁(负)"),
    "e_fcst_sue": (fcst_sue, "预告盈余惊喜"), "e_earn_mgmt": (earn_mgmt, "盈余操纵(负)"),
}
