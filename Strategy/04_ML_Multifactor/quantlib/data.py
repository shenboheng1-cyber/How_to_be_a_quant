# -*- coding: utf-8 -*-
"""
quantlib.data —— 数据面板加载层
================================================================
职责：把 mart/fact_stock_daily.parquet 这张 1365 万行的日频大表，
按【调仓频率】抽成一个干净的"研究面板(panel)"，每行是：

    (调仓日 trddt, 股票 stkcd, 该日各因子原料, 未来收益 fwd_ret)

其中 fwd_ret = 从本调仓日持有到下一个调仓日的总收益（含分红）。
这是整条流水线唯一接触原始大表的地方，下游模块只看 panel。

== 防前视偏差的三条铁律（本模块负责前两条）==
1. 因子原料只取调仓日 t 当天及以前的数据。
2. fwd_ret 用 t 之后的日频 ret 逐日复利（含分红），绝不含 t 当天及以前。
3. （在 universe.py）调仓日不可交易的票，当期不能进组合。

用法：
    from quantlib import data
    panel = data.load_panel(freq="M", start="2015-01-01")
"""
from __future__ import annotations
import os
import duckdb
import pandas as pd

# ---- 路径 ----
# 代码与数据【分离】：本项目代码可独立放置/上 GitHub，41GB 数据仍留在 CSMAR 目录。
# 数据根默认指向 CSMAR；换机器/换路径时设环境变量 CSMAR_DATA_ROOT 覆盖即可。
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(_PKG_DIR)
DATA_ROOT = os.environ.get("CSMAR_DATA_ROOT", "/Users/shenboheng/CSMAR")
MART_DIR = os.path.join(DATA_ROOT, "mart")
DAILY_PARQUET = os.path.join(MART_DIR, "fact_stock_daily.parquet")

# 默认从大表带出的"因子原料"列。下游因子函数从这些列计算因子值。
DEFAULT_FIELDS = [
    "close", "adj_close", "ret",
    "total_mktcap", "float_mktcap", "turnover", "amount", "volume",
    "pe_ttm", "pb", "ps_ttm",
    "is_st", "limit_status", "market",
]

# 频率 -> DuckDB date_trunc 的单位。调仓日取该周期内的"最后一个交易日"。
_FREQ_TRUNC = {"M": "month", "W": "week", "Q": "quarter"}


def connect() -> duckdb.DuckDBPyConnection:
    """新建一个内存 DuckDB 连接（直接读 parquet，不落库）。"""
    con = duckdb.connect()
    con.sql("PRAGMA memory_limit='6GB'")
    return con


def rebalance_dates(freq: str = "M", start: str | None = None,
                    end: str | None = None) -> list:
    """返回调仓日列表 = 每个周期内最后一个【真实交易日】。

    用真实交易日（而非自然月末 31 号）才不会指向一个没开盘的日期。
    """
    trunc = _FREQ_TRUNC[freq]
    con = connect()
    cond = []
    if start: cond.append(f"trddt >= DATE '{start}'")
    if end:   cond.append(f"trddt <= DATE '{end}'")
    where = ("WHERE " + " AND ".join(cond)) if cond else ""
    sql = f"""
        WITH cal AS (SELECT DISTINCT trddt FROM '{DAILY_PARQUET}' {where})
        SELECT max(trddt) AS reb_dt
        FROM cal
        GROUP BY date_trunc('{trunc}', trddt)
        ORDER BY reb_dt
    """
    df = con.sql(sql).df()
    con.close()
    return df["reb_dt"].tolist()


def load_panel(freq: str = "M", start: str | None = None,
               end: str | None = None, fields: list | None = None) -> pd.DataFrame:
    """构建调仓频率的研究面板。

    返回 DataFrame，列含：trddt, stkcd, name, <fields>, fwd_ret
    —— fwd_ret 是【未来】一期收益（已对齐，无前视）；最后一期因无"下一期"
       而 fwd_ret 为 NaN，下游训练/回测时会自然丢弃。

    实现要点：
    - 只在调仓日那几天的快照上做计算，数据量从 1365 万降到 ~百万行，pandas 拿得动。
    - fwd_ret = 下一调仓日的 adj_close / 本调仓日 adj_close - 1，按 stkcd 分组用 LEAD()。
    - 已退市股：其最后一期 LEAD() 为空 -> fwd_ret=NaN -> 自然丢弃。
      （v1 局限：这样会漏掉"退市当期的最后一段暴跌"，略微低估退市损失；
        L4 再用"区间内最后可得价格"精确化。已在 spec 标注。）
    """
    fields = fields or DEFAULT_FIELDS
    trunc = _FREQ_TRUNC[freq]
    field_sql = ", ".join(fields)
    cond = []
    if start: cond.append(f"trddt >= DATE '{start}'")
    if end:   cond.append(f"trddt <= DATE '{end}'")
    where = ("WHERE " + " AND ".join(cond)) if cond else ""

    con = connect()
    # 思路：
    #   reb  = 每个周期最后一个交易日（调仓日）
    #   snap = 调仓日当天的因子原料快照
    #   fwd  = 未来收益 = 日频 ret 在 (本调仓日, 下一调仓日] 区间内逐日复利
    #          —— 用 ASOF JOIN 把每个交易日归属到"它所属持有期的起始调仓日"
    #          —— 最后一期没有"下一调仓日"，故无 fwd_ret（自然为 NaN）
    sql = f"""
        WITH daily AS (
            SELECT * FROM '{DAILY_PARQUET}' {where}
        ),
        reb AS (
            SELECT max(trddt) AS r
            FROM (SELECT DISTINCT trddt FROM daily)
            GROUP BY date_trunc('{trunc}', trddt)
        ),
        snap AS (
            SELECT d.stkcd, d.trddt, d.name, {", ".join("d."+f for f in fields)}
            FROM daily d
            JOIN reb ON d.trddt = reb.r
        ),
        hold AS (   -- 每个交易日的对数收益，归属到其持有期起始调仓日
            SELECT d.stkcd, r.r AS reb_dt,
                   ln(1 + greatest(d.ret, -0.999)) AS lr
            FROM daily d
            ASOF JOIN reb r ON d.trddt > r.r
        ),
        fwd AS (
            SELECT stkcd, reb_dt, exp(sum(lr)) - 1.0 AS fwd_ret
            FROM hold GROUP BY stkcd, reb_dt
        )
        SELECT s.*, f.fwd_ret
        FROM snap s
        LEFT JOIN fwd f ON s.stkcd = f.stkcd AND s.trddt = f.reb_dt
        ORDER BY s.trddt, s.stkcd
    """
    panel = con.sql(sql).df()
    con.close()
    return panel


def load_trailing_features(freq: str = "M", start: str | None = None,
                           end: str | None = None) -> pd.DataFrame:
    """在日频大表上用窗口函数计算【滚动回看】因子原料，只在调仓日取值。

    全部只用"截至调仓日 t 及以前"的数据 —— 无前视。返回 (stkcd, trddt, 特征...)。
      mom_12_1 : 12-1 月动量 = 过去 252→21 个交易日的累计收益（跳过最近1月避开反转）
      rev_1m   : 1 月反转 = 过去 21 个交易日累计收益（取负即反转因子）
      vol_60   : 60 日收益波动率（特质波动的简化代理）
      amihud   : 60 日 Amihud 非流动性 = mean(|ret|/成交额)，越大越不流动
      max_ret  : 过去 21 日最大单日收益（彩票/博彩因子 Bali 2011）
      turn_1m  : 过去 21 日平均换手率
      w52high  : 距52周高点 = 当前复权价 / 过去252日最高复权价 ∈(0,1]，越接近1越靠近高点
                 （George-Hwang 2004 锚定效应因子；L2 头牌原创因子）
      range_pos: 52周区间位置 = (price-min)/(max-min) ∈[0,1]，0=在最低、1=在最高
    """
    reb = rebalance_dates(freq, start, end)
    reb_list = ", ".join(f"DATE '{d}'" for d in reb)
    # 为保证窗口左侧有足够历史，日频数据从 start 往前推一年开始读
    daily_start = ""
    if start:
        daily_start = f"WHERE trddt >= DATE '{start}' - INTERVAL 400 DAY"

    con = connect()
    sql = f"""
        WITH d AS (
            SELECT stkcd, trddt, ret, amount, turnover, adj_close,
                   ln(1 + greatest(ret, -0.999)) AS lr
            FROM '{DAILY_PARQUET}' {daily_start}
        ),
        feat AS (
            SELECT stkcd, trddt,
                exp(sum(lr) OVER w_12_1) - 1                       AS mom_12_1,
                exp(sum(lr) OVER w_21)  - 1                        AS rev_1m,
                stddev_samp(ret)        OVER w_60                  AS vol_60,
                avg(abs(ret) / nullif(amount, 0)) OVER w_60 * 1e9  AS amihud,
                max(ret)                OVER w_21                  AS max_ret,
                avg(turnover)           OVER w_21                  AS turn_1m,
                adj_close / max(adj_close) OVER w_252              AS w52high,
                (adj_close - min(adj_close) OVER w_252)
                  / nullif(max(adj_close) OVER w_252
                           - min(adj_close) OVER w_252, 0)         AS range_pos
            FROM d
            WINDOW
                w_12_1 AS (PARTITION BY stkcd ORDER BY trddt ROWS BETWEEN 251 PRECEDING AND 21 PRECEDING),
                w_21   AS (PARTITION BY stkcd ORDER BY trddt ROWS BETWEEN 20  PRECEDING AND CURRENT ROW),
                w_60   AS (PARTITION BY stkcd ORDER BY trddt ROWS BETWEEN 59  PRECEDING AND CURRENT ROW),
                w_252  AS (PARTITION BY stkcd ORDER BY trddt ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
        )
        SELECT * FROM feat
        WHERE trddt IN ({reb_list})
        ORDER BY trddt, stkcd
    """
    out = con.sql(sql).df()
    con.close()
    return out


def load_research_panel(freq: str = "M", start: str | None = None,
                        end: str | None = None) -> pd.DataFrame:
    """一键入口：快照面板 + 滚动回看因子，按 (stkcd, trddt) 合并。

    返回的 panel 含：价量估值快照、fwd_ret、以及 mom/rev/vol/amihud/max_ret/turn 等。
    universe 标记请在外层再调 universe.add_universe()。
    """
    panel = load_panel(freq, start, end)
    feat = load_trailing_features(freq, start, end)
    return panel.merge(feat, on=["stkcd", "trddt"], how="left")
