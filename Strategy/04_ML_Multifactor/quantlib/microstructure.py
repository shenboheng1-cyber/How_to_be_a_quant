# -*- coding: utf-8 -*-
"""
quantlib.microstructure —— 高频微观结构因子（订单流/价差/知情交易/跳跃）
================================================================
从 raw/HF_*.parquet（日内算好的日频指标）构建与价量正交的因子。
携带日频 OHLCV 没有的信息：订单流方向、真实价差、知情交易、已实现偏度/跳跃。

⚠️ 复合主键坑（见 csmar-data-gotchas）：
  HF_VPIN 主键 (Stkcd,Trddt,N) —— 取每股每日最细分桶 N；
  HF_StockJump 主键 (Stkcd,Trddt,Alpha) —— 取显著性档 Alpha='A'。

因子在调仓日采样，多数取过去 20 个交易日均值（跳跃频率取 60 日）。无前视。
"""
from __future__ import annotations
import os
import pandas as pd
from . import data

RAW = lambda t: os.path.join(data.DATA_ROOT, "raw", f"{t}.parquet")


def _reb_clause(freq, start, end):
    reb = data.rebalance_dates(freq, start, end)
    return ", ".join(f"DATE '{d}'" for d in reb)


def load_micro_features(freq: str = "M", start: str | None = None,
                        end: str | None = None) -> pd.DataFrame:
    """构建微观结构因子面板，返回 (stkcd, trddt) + 因子列。"""
    reb = _reb_clause(freq, start, end)
    con = data.connect()
    D = lambda c: f"TRY_CAST({c} AS DOUBLE)"     # HF 表数值列以字符串存储，需显式转换

    def win(expr, w=20):
        return f"avg({expr}) OVER (PARTITION BY Stkcd ORDER BY Trddt ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW)"

    # ---- 1) 订单流不平衡（BSImbalance）----
    bs = con.sql(f"""
        WITH d AS (
            SELECT Stkcd, CAST(Trddt AS DATE) AS Trddt,
                {D('B_Amount')} AS ba, {D('S_Amount')} AS sa,
                {D('B_Amount_L')} AS bal, {D('S_Amount_L')} AS sal,
                {D('B_Amount_B')} AS bab, {D('S_Amount_B')} AS sab,
                {D('B_Amount_S')} AS bas, {D('S_Amount_S')} AS sas
            FROM '{RAW('HF_BSImbalance')}'
        ),
        e AS (
            SELECT Stkcd, Trddt,
                (ba - sa) / nullif(ba + sa, 0)                              AS ofi,
                (bal + bab - sal - sab) / nullif(bal + bab + sal + sab, 0)  AS ofi_big,
                (bas - sas) / nullif(bas + sas, 0)                          AS ofi_small
            FROM d
        )
        SELECT Stkcd AS stkcd, Trddt AS trddt,
               {win('ofi')} AS ofi, {win('ofi_big')} AS ofi_big, {win('ofi_small')} AS ofi_small
        FROM e QUALIFY Trddt IN ({reb})
    """).df()

    # ---- 2) 有效价差（Spread，相对、额加权）----
    sp = con.sql(f"""
        WITH d AS (
            SELECT Stkcd, CAST(Trddt AS DATE) AS Trddt, {D('AEsp_Amount')} AS spread
            FROM '{RAW('HF_Spread')}'
        )
        SELECT Stkcd AS stkcd, Trddt AS trddt, {win('spread')} AS spread
        FROM d QUALIFY Trddt IN ({reb})
    """).df()

    # ---- 3) 知情交易概率（VPIN，取每股每日最细分桶 N）----
    vp = con.sql(f"""
        WITH d AS (
            SELECT Stkcd, CAST(Trddt AS DATE) AS Trddt, {D('VPIN')} AS VPIN
            FROM '{RAW('HF_VPIN')}'
            QUALIFY row_number() OVER (PARTITION BY Stkcd, Trddt ORDER BY {D('N')} DESC) = 1
        )
        SELECT Stkcd AS stkcd, Trddt AS trddt, {win('VPIN')} AS vpin
        FROM d QUALIFY Trddt IN ({reb})
    """).df()

    # ---- 4) 跳跃/已实现半方差（StockJump，Alpha='A'）----
    jp = con.sql(f"""
        WITH d AS (
            SELECT Stkcd, CAST(Trddt AS DATE) AS Trddt,
                ({D('RS_P')} - {D('RS_N')}) / nullif({D('RV')}, 0)          AS rskew,
                {D('RS_N')} / nullif({D('RS_N')} + {D('RS_P')}, 0)          AS downside,
                TRY_CAST(ISJump AS INT)                                     AS isjump,
                {D('SJV')}                                                  AS sjv
            FROM '{RAW('HF_StockJump')}'
            WHERE Alpha = 'A'
        )
        SELECT Stkcd AS stkcd, Trddt AS trddt,
               {win('rskew')} AS rskew, {win('downside')} AS downside,
               {win('isjump', 60)} AS jump_freq, {win('sjv')} AS sjv
        FROM d QUALIFY Trddt IN ({reb})
    """).df()

    con.close()
    out = bs.merge(sp, on=["stkcd", "trddt"], how="outer") \
            .merge(vp, on=["stkcd", "trddt"], how="outer") \
            .merge(jp, on=["stkcd", "trddt"], how="outer")
    return out


# 因子定义：(原始列, 方向, 中文名)。direction=-1 表示取负后"值大=预期收益高"。
MICRO_SPECS = {
    "ofi":       ("ofi",       +1, "订单流不平衡(净买)"),
    "ofi_big":   ("ofi_big",   +1, "大单不平衡(主力方向)"),
    "ofi_small": ("ofi_small", -1, "小单不平衡(散户,反向)"),
    "spread":    ("spread",    +1, "有效价差(流动性溢价)"),
    "vpin":      ("vpin",      +1, "知情交易概率VPIN"),
    "rskew":     ("rskew",     -1, "已实现偏度(彩票,反向)"),
    "downside":  ("downside",  +1, "下行半方差占比"),
    "jump_freq": ("jump_freq", +1, "跳跃频率"),
    "sjv":       ("sjv",       -1, "符号跳跃(反向)"),
}


def attach_factors(panel: pd.DataFrame) -> dict:
    """把 MICRO_SPECS 转成 {name: (raw_series, cn)}，供统一评估/合成。panel 需已并入微观列。"""
    return {k: (panel[col] * sign, cn) for k, (col, sign, cn) in MICRO_SPECS.items()}


# ===== spec 驱动的批量加载（给定因子目录即可，无需手写每个查询）=====
_BASE = {
    "HF_BSImbalance": "SELECT * FROM '{p}'",
    "HF_Spread":      "SELECT * FROM '{p}'",
    "HF_VPIN":        "SELECT * FROM '{p}' QUALIFY row_number() OVER (PARTITION BY Stkcd,Trddt ORDER BY TRY_CAST(N AS INT) DESC)=1",
    "HF_StockJump":   "SELECT * FROM '{p}' WHERE Alpha='A'",
}


def _col_expr(spec: dict) -> str:
    f = f"({spec['daily_formula']})"
    w = int(spec["window"]); agg = spec["agg"]; nm = spec["name"]
    frame = f"PARTITION BY Stkcd ORDER BY CAST(Trddt AS DATE) ROWS BETWEEN {w-1} PRECEDING AND CURRENT ROW"
    if agg == "mean":    e = f"avg({f}) OVER ({frame})"
    elif agg == "std":   e = f"stddev_samp({f}) OVER ({frame})"
    elif agg == "sum":   e = f"sum({f}) OVER ({frame})"
    elif agg == "delta": e = f"{f} - lag({f}, {w}) OVER (PARTITION BY Stkcd ORDER BY CAST(Trddt AS DATE))"
    else:                e = f                      # last
    return f"{e} AS {nm}"


def load_specs(specs: list, freq: str = "M", start: str | None = None,
               end: str | None = None) -> pd.DataFrame:
    """按因子目录(每个 spec: table/daily_formula/window/agg/name)批量构建因子面板。
    返回 (stkcd, trddt) + 每个 spec 一列(原始值，不含 sign)。"""
    reb = _reb_clause(freq, start, end)
    con = data.connect()
    by_table = {}
    for s in specs:
        by_table.setdefault(s["table"], []).append(s)

    def run(table, slist):
        cols = ",\n               ".join(_col_expr(s) for s in slist)
        base = _BASE[table].format(p=RAW(table))
        return con.sql(f"""
            SELECT * FROM (
                SELECT Stkcd AS stkcd, CAST(Trddt AS DATE) AS trddt, {cols}
                FROM ({base})
            ) WHERE trddt IN ({reb})
        """).df()

    merged, failed = None, []
    for table, slist in by_table.items():
        try:                                   # 先整表批量
            df = run(table, slist)
        except Exception:                      # 有坏公式 → 逐个回退，跳过坏的
            good = []
            for s in slist:
                try:
                    run(table, [s]); good.append(s)
                except Exception as e:
                    failed.append((s["name"], str(e)[:50]))
            df = run(table, good) if good else None
        if df is not None:
            merged = df if merged is None else merged.merge(df, on=["stkcd", "trddt"], how="outer")
    con.close()
    if failed:
        print(f"[load_specs] 跳过 {len(failed)} 个无法计算的因子: {[f[0] for f in failed]}")
    return merged
