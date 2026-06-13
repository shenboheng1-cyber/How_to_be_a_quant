"""Step 5: 周度调仓回测与报告 (复现 表7/图12)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.config import load_yaml
from src.utils.io import load_parquet
from src.utils.calendar import load_trading_days
from src.data import loaders
from src.backtest.engine import run_backtest
from src.backtest.report import make_report

if __name__ == "__main__":
    cfg = load_yaml("strategy")["backtest"]
    tw = load_parquet("processed", "target_weights")   # [date, code, weight]
    target_weights = {d: g.set_index("code")["weight"]
                      for d, g in tw.groupby("date")}

    etf = loaders.etf_daily().pivot(index="date", columns="code", values="nav")
    days = load_trading_days()
    res = run_backtest(etf, target_weights, days,
                       cost_rate=cfg["cost_rate"],
                       initial_capital=cfg["initial_capital"])

    bench = loaders.index_daily().query("code == @cfg['benchmark_index']") \
                                 .set_index("date")["close"]
    make_report(res["nav"], bench, res["turnover"], "outputs")
