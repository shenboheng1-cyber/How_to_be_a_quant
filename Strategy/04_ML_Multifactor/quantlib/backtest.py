# -*- coding: utf-8 -*-
"""
quantlib.backtest —— 组合回测（L4）
================================================================
把 alpha 信号变成可交易组合，扣【真实交易成本】、对标指数，给出净业绩。
- 多头(top decile, 等权)：贴近真实(A股做空难)，对标中证500算超额/IR。
- 多空(top−bottom)：学术口径参考。
- 换手率驱动的成本：cost_t = 单边换手 × 双边成本率。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import data

IDX = f"{data.DATA_ROOT}/raw/IDX_Idxtrd.parquet"
_PPY = {"M": 12, "W": 52}


def load_benchmark(code: str = "000905", freq: str = "M",
                   start: str | None = None, end: str | None = None) -> pd.Series:
    """指数的【未来一期】收益(对齐 fwd_ret：调仓日 t 的值 = t→t+1 的指数收益)。000905=中证500。"""
    reb = data.rebalance_dates(freq, start, end)
    con = data.connect()
    df = con.sql(f"""
        SELECT CAST(Idxtrd01 AS DATE) AS dt, TRY_CAST(Idxtrd05 AS DOUBLE) AS close
        FROM '{IDX}' WHERE Indexcd='{code}' ORDER BY dt
    """).df()
    con.close()
    s = df.set_index("dt")["close"]
    s.index = pd.to_datetime(s.index)
    lvl = s.reindex(pd.to_datetime(reb), method="ffill")
    fwd = lvl.shift(-1) / lvl - 1.0                    # t→t+1
    fwd.index = reb
    return fwd


def backtest(panel_oos: pd.DataFrame, signal, benchmark_fwd: pd.Series,
             cost: float = 0.003, n_groups: int = 10) -> pd.DataFrame:
    """逐调仓日构建组合，返回每期 毛/净 收益、换手、基准。等权。"""
    df = pd.DataFrame({"dt": panel_oos["trddt"].values, "stk": panel_oos["stkcd"].values,
                       "r": panel_oos["fwd_ret"].values, "s": np.asarray(signal),
                       "cap": panel_oos["total_mktcap"].values}).dropna(subset=["s", "r"])
    dates = sorted(df["dt"].unique())
    prev_l, prev_ls = pd.Series(dtype=float), pd.Series(dtype=float)
    rows = []
    for d in dates:
        g = df[df["dt"] == d]
        q = g["s"].rank(pct=True)
        top, bot = g[q > 1 - 1 / n_groups], g[q < 1 / n_groups]
        # 多头(等权)
        wl = pd.Series(1.0 / len(top), index=top["stk"].values) if len(top) else pd.Series(dtype=float)
        long_r = top["r"].mean() if len(top) else 0.0
        idx = wl.index.union(prev_l.index)
        to_l = 0.5 * (wl.reindex(idx, fill_value=0) - prev_l.reindex(idx, fill_value=0)).abs().sum()
        # 多空
        if len(top) and len(bot):
            wls = pd.concat([pd.Series(1.0 / len(top), index=top["stk"].values),
                             pd.Series(-1.0 / len(bot), index=bot["stk"].values)])
            ls_r = top["r"].mean() - bot["r"].mean()
        else:
            wls, ls_r = pd.Series(dtype=float), 0.0
        idx2 = wls.index.union(prev_ls.index)
        to_ls = 0.5 * (wls.reindex(idx2, fill_value=0) - prev_ls.reindex(idx2, fill_value=0)).abs().sum()
        rows.append({"dt": d, "long_g": long_r, "long_n": long_r - to_l * cost, "to_l": to_l,
                     "ls_g": ls_r, "ls_n": ls_r - to_ls * cost, "to_ls": to_ls,
                     "med_cap": top["cap"].median()})
        prev_l, prev_ls = wl, wls
    out = pd.DataFrame(rows).set_index("dt")
    out["bench"] = pd.Series(benchmark_fwd).reindex(out.index).values
    out["long_excess"] = out["long_n"] - out["bench"]
    return out


def metrics(r: pd.Series, freq: str = "M") -> dict:
    r = pd.Series(r).dropna()
    ppy = _PPY[freq]; n = len(r)
    nav = (1 + r).cumprod()
    ann = (1 + r).prod() ** (ppy / n) - 1
    vol = r.std(ddof=1) * np.sqrt(ppy)
    return {"年化": round(ann, 4), "波动": round(vol, 4),
            "夏普": round(ann / vol, 2) if vol else np.nan,
            "最大回撤": round((nav / nav.cummax() - 1).min(), 3)}


def info_ratio(excess: pd.Series, freq: str = "M") -> float:
    e = pd.Series(excess).dropna(); ppy = _PPY[freq]
    return round(e.mean() / e.std(ddof=1) * np.sqrt(ppy), 2) if e.std(ddof=1) else np.nan
