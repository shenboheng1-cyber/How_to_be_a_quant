"""Step 2: 熵权法计算5类宏观综合得分。输出: data/processed/macro_scores_weekly.parquet
自检: 复现报告 图3。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.config import load_yaml
from src.utils.io import save_parquet
from src.data import loaders
from src.macro.entropy import to_weekly, rolling_category_scores

if __name__ == "__main__":
    scfg = load_yaml("strategy")["macro"]
    mcfg = load_yaml("macro_indicators")

    weekly_cols, directions, categories = {}, {}, {}
    for cat, spec in mcfg["categories"].items():
        categories[cat] = []
        for ind in spec["indicators"]:
            key = ind["name"]
            s = loaders.macro_indicator(key)
            weekly_cols[key] = to_weekly(s) if ind["freq"] == "d" else s.resample("W-SUN").mean()
            directions[key] = ind["direction"]
            categories[cat].append(key)

    weekly = pd.DataFrame(weekly_cols)
    scores = rolling_category_scores(weekly, directions, categories,
                                     window=scfg["entropy_window"],
                                     invalid_ratio=scfg["invalid_ratio_threshold"])
    save_parquet(scores, "processed", "macro_scores_weekly")
    print(scores.tail())
