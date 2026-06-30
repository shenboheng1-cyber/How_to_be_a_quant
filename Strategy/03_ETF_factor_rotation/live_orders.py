"""模拟盘月度下单助手 —— 把目标权重 + 当前持仓 → 这个月的买卖清单。

用法:
    # 1) 首月建仓(还没有持仓,直接按目标权重一次性铺满):
    python3 live_orders.py --capital 1000000 --first

    # 2) 此后每月调仓(先把上月成交后的真实持仓填进 holdings.csv):
    python3 live_orders.py --holdings holdings.csv

holdings.csv 格式(从券商App抄市值或份额,二选一):
    fund_code,shares
    511880,100000
    511190,50000
    ...
    # 现金(未投部分)写成一行: fund_code=CASH, shares=现金金额
    CASH,30000

核心算法(与回测 backtest_monthly_strategy 的 rebalance_lambda 完全一致):
    新持仓权重 = 当前权重 + lam * (目标权重 - 当前权重)
首月(--first)直接令 新持仓 = 目标(一次铺满),之后每月用 lam=0.4 部分再平衡。
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path("/Users/shenboheng/Documents/ClaudeCode/dataset/基金深度分析")
DB = DATA_DIR / "etf_market_ifind.db"
WEIGHTS_CSV = Path("outputs_v2_final/rebalance_weights.csv")
CASH_CODE = "511880"           # 现金腿 ETF(货币)
LOT = 100                       # ETF 最小申报单位 100 份
AMOUNT_FLOOR = 3000 * 1e4       # 近20日均成交额下限 3000万(报告 §7)
PREMIUM_CAP = 2.0               # |折溢价率| 上限 2%(报告 §7); premiumRatio 单位已是百分数


def load_market(codes: list[str]) -> pd.DataFrame:
    """每只 ETF 最新原始市价 close、折溢价 premiumRatio、近20日均成交额 amount。"""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        q = pd.read_sql_query(
            "SELECT fund_code, date, close, premiumRatio, amount FROM etf_quote "
            "WHERE close IS NOT NULL ORDER BY date", con)
    finally:
        con.close()
    rows = []
    for code, g in q.groupby("fund_code"):
        g = g.tail(20)
        last = g.iloc[-1]
        rows.append({
            "fund_code": code,
            "price": float(last["close"]),
            "premium": float(last["premiumRatio"]) if pd.notna(last["premiumRatio"]) else 0.0,
            "adv20": float(g["amount"].mean()),
            "quote_date": last["date"],
        })
    return pd.DataFrame(rows).set_index("fund_code")


def main() -> None:
    ap = argparse.ArgumentParser(description="模拟盘月度下单助手(V2)")
    ap.add_argument("--weights", default=str(WEIGHTS_CSV), help="目标权重CSV(默认取最新调仓日)")
    ap.add_argument("--date", default=None, help="指定调仓日(默认最新)")
    ap.add_argument("--holdings", default=None, help="当前持仓CSV(fund_code,shares; 现金行 CASH,金额)")
    ap.add_argument("--capital", type=float, default=None, help="首月总资金(无holdings时必填)")
    ap.add_argument("--lam", type=float, default=0.4, help="部分再平衡系数")
    ap.add_argument("--first", action="store_true", help="首月建仓:直接铺到目标(忽略lam)")
    ap.add_argument("--out", default="trade_orders.csv")
    args = ap.parse_args()

    # ---- 目标权重 ----
    w = pd.read_csv(args.weights)
    w.columns = [c.lstrip("﻿") for c in w.columns]
    target_date = args.date or w["date"].max()
    tgt = w[w["date"] == target_date].copy()
    tgt["fund_code"] = tgt["fund_code"].astype(str)
    name_map = dict(zip(tgt["fund_code"], tgt.get("fund_name", tgt["fund_code"])))
    target_w = tgt.set_index("fund_code")["weight"]

    # ---- 当前持仓 + 总资产 ----
    if args.holdings and Path(args.holdings).exists():
        h = pd.read_csv(args.holdings, dtype={"fund_code": str})
        h["fund_code"] = h["fund_code"].astype(str)
        cash = float(h.loc[h["fund_code"] == "CASH", "shares"].sum())
        h = h[h["fund_code"] != "CASH"]
        codes = sorted(set(target_w.index) | set(h["fund_code"]))
        mkt = load_market(codes)
        cur_shares = h.set_index("fund_code")["shares"].astype(float)
        cur_mv = (cur_shares * mkt["price"]).reindex(codes).fillna(0.0)
        total = float(cur_mv.sum() + cash)
    else:
        if args.capital is None:
            ap.error("无 holdings 时必须用 --capital 指定首月总资金,并加 --first")
        codes = sorted(target_w.index)
        mkt = load_market(codes)
        cur_shares = pd.Series(0.0, index=codes)
        cur_mv = pd.Series(0.0, index=codes)
        cash = float(args.capital)
        total = float(args.capital)
        args.first = True

    cur_w = (cur_mv / total).reindex(codes).fillna(0.0)
    cur_w[CASH_CODE] = cur_w.get(CASH_CODE, 0.0) + cash / total  # 现金计入现金腿
    tgt_full = target_w.reindex(codes).fillna(0.0)

    # ---- 核心:执行权重 ----
    if args.first:
        exec_w = tgt_full.copy()
        mode = "首月建仓(铺满目标)"
    else:
        exec_w = cur_w + args.lam * (tgt_full - cur_w)
        mode = f"部分再平衡 lam={args.lam}"

    # ---- 权重 → 目标手数 → 买卖 ----
    price = mkt["price"].reindex(codes)
    tgt_mv = exec_w * total
    tgt_shares = ((tgt_mv / price / LOT).round() * LOT).fillna(0.0)
    cur_sh = cur_shares.reindex(codes).fillna(0.0)
    delta = tgt_shares - cur_sh

    out = pd.DataFrame({
        "代码": codes,
        "名称": [name_map.get(c, "") for c in codes],
        "最新市价": price.round(3).values,
        "目标权重%": (exec_w * 100).round(2).values,
        "目标手数": (tgt_shares / LOT).astype(int).values,
        "当前手数": (cur_sh / LOT).astype(int).values,
        "买卖手数": (delta / LOT).astype(int).values,
        "约需金额": (delta * price).round(0).values,
        "折溢价%": mkt["premium"].reindex(codes).round(2).values,
        "近20日均额(万)": (mkt["adv20"].reindex(codes) / 1e4).round(0).values,
    })
    out["方向"] = out["买卖手数"].apply(lambda x: "买入" if x > 0 else ("卖出" if x < 0 else "持有"))
    # 红线提示
    warn = []
    for _, r in out.iterrows():
        f = []
        if r["买卖手数"] != 0 and r["近20日均额(万)"] < AMOUNT_FLOOR / 1e4:
            f.append("流动性低")
        if abs(r["折溢价%"]) > PREMIUM_CAP:
            f.append("折溢价大")
        warn.append("⚠ " + "/".join(f) if f else "")
    out["提示"] = warn

    out = out[out["买卖手数"] != 0].sort_values("约需金额").reset_index(drop=True)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n调仓日(信号): {target_date} | 模式: {mode} | 总资产: {total:,.0f}")
    print(f"行情基准日: {mkt['quote_date'].iloc[0]}  (下单请用 T+1 尾盘实时价/限价贴 IOPV)\n")
    cols = ["代码", "名称", "方向", "买卖手数", "约需金额", "目标权重%", "折溢价%", "近20日均额(万)", "提示"]
    print(out[cols].to_string(index=False))
    print(f"\n净买入金额: {out['约需金额'].sum():,.0f}  (≈ 应保留现金缓冲 1-2%)")
    flagged = out[out["提示"] != ""]
    if len(flagged):
        print(f"\n⚠ {len(flagged)} 只触发流动性/折溢价红线,建议人工核对或剔除:")
        print(flagged[["代码", "名称", "提示"]].to_string(index=False))
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n已写出: {args.out}")


if __name__ == "__main__":
    main()
