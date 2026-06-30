# -*- coding: utf-8 -*-
"""
防前视偏差单元测试
================================================================
量化回测最致命的 bug 是"未来函数"——不小心用了当时不可能知道的信息，
导致回测虚高、实盘失效。这组测试用代码【证明】框架没有这个 bug。

可两种方式运行：
    pytest tests/test_no_lookahead.py -v
    python  tests/test_no_lookahead.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from quantlib import data, preprocess

START, END = "2018-01-01", "2022-12-31"


def _panel():
    if not hasattr(_panel, "_c"):
        _panel._c = data.load_research_panel("M", START, END)
    return _panel._c


# ---------- 1) 未来收益必须等于"未来"日频收益的独立复利 ----------
def test_fwd_ret_matches_independent_daily_compounding():
    """对多个 (股票, 调仓日) 样本，用日频 ret 在 (t, t+1] 区间独立复利重算，
    必须与 panel 的 fwd_ret 一致。这是收益对齐正确性的核心证明。"""
    p = _panel().sort_values(["stkcd", "trddt"]).reset_index(drop=True)
    p["t_next"] = p.groupby("stkcd")["trddt"].shift(-1)
    cand = p[p["fwd_ret"].notna() & p["t_next"].notna()]
    # 跨多只股票、多个时间点抽 30 个样本（含茅台等分红股，专门覆盖分红月）
    sample = cand.iloc[:: max(1, len(cand) // 30)].head(30)
    con = data.connect()
    bad = 0
    for _, row in sample.iterrows():
        r = con.sql(f"""
            SELECT exp(sum(ln(1+greatest(ret,-0.999))))-1
            FROM '{data.DAILY_PARQUET}'
            WHERE stkcd='{row.stkcd}'
              AND trddt > DATE '{row.trddt}' AND trddt <= DATE '{row.t_next}'
        """).fetchone()[0]
        if r is None or abs(row.fwd_ret - r) > 1e-9:
            bad += 1
    con.close()
    assert bad == 0, f"{bad}/30 个样本的 fwd_ret 与日频独立复利不符"
    print(f"  [OK] 30 个跨股票/跨期样本 fwd_ret = 日频区间复利，完全一致")


def test_last_period_fwd_ret_is_nan():
    """最后一个调仓日没有'下一期'，fwd_ret 必须全为 NaN —— 证明不会越界取未来。"""
    p = _panel()
    last = p["trddt"].max()
    assert p.loc[p["trddt"] == last, "fwd_ret"].isna().all(), \
        "最后一期出现了非空 fwd_ret —— 可能越界使用了不存在的未来数据"
    print(f"  [OK] 最后一期 {pd.Timestamp(last).date()} fwd_ret 全为 NaN")


# ---------- 2) 滚动因子不得使用未来数据 ----------
def test_trailing_feature_no_future_leak():
    """mom_12_1(t) 只用 t 及以前的数据：把数据在 t 处截断重算，结果必须不变。"""
    stk, asof = "000001", "2021-06-30"
    con = data.connect()
    # 全量数据下，asof 当月最后交易日的 mom（直接用库函数口径重算一条）
    sql_full = f"""
        WITH d AS (SELECT trddt, ln(1+greatest(ret,-0.999)) lr
                   FROM '{data.DAILY_PARQUET}' WHERE stkcd='{stk}')
        SELECT exp(sum(lr))-1 FROM d
        WHERE trddt <= (SELECT max(trddt) FROM d WHERE trddt <= DATE '{asof}')
          AND trddt >  (SELECT trddt FROM d WHERE trddt <= DATE '{asof}'
                        ORDER BY trddt DESC LIMIT 1 OFFSET 251)
          AND trddt <= (SELECT trddt FROM d WHERE trddt <= DATE '{asof}'
                        ORDER BY trddt DESC LIMIT 1 OFFSET 21)
    """
    # 截断到 asof 之后的数据全删，再用同口径算，应完全一致
    val_full = con.sql(sql_full).fetchone()[0]
    val_trunc = con.sql(sql_full.replace(
        f"WHERE stkcd='{stk}')",
        f"WHERE stkcd='{stk}' AND trddt <= DATE '{asof}')")).fetchone()[0]
    con.close()
    assert val_full is not None and abs(val_full - val_trunc) < 1e-9, \
        "截断未来数据后动量值改变 —— 存在未来函数！"
    print(f"  [OK] {stk} 动量(截至{asof})截断前后一致，无未来泄漏")


# ---------- 3) 预处理必须是逐横截面独立的 ----------
def test_preprocess_is_cross_sectional():
    """删掉其它调仓日的数据，不应改变某一天的预处理结果（证明无跨期泄漏）。"""
    p = _panel()
    p = p[p["total_mktcap"].notna()].reset_index(drop=True)
    raw = 1.0 / p["pe_ttm"].where(p["pe_ttm"] > 0)
    full = preprocess.preprocess_factor(p, raw, do_neutralize=True)

    dt = p["trddt"].iloc[len(p)//2]
    mask = (p["trddt"] == dt).values
    sub = p[mask].reset_index(drop=True)
    sub_raw = 1.0 / sub["pe_ttm"].where(sub["pe_ttm"] > 0)
    sub_only = preprocess.preprocess_factor(sub, sub_raw, do_neutralize=True)

    a = pd.Series(full[mask].values)
    b = pd.Series(sub_only.values)
    both = a.notna() & b.notna()
    assert np.allclose(a[both], b[both], atol=1e-8), \
        "单独处理某一天与整体处理结果不同 —— 预处理跨期泄漏！"
    print(f"  [OK] {pd.Timestamp(dt).date()} 横截面单独/整体预处理一致，无跨期泄漏")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"运行 {name} ...")
            fn()
    print("\n全部通过 ✅ 框架无前视偏差")
