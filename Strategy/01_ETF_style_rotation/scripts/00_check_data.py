"""Step 0: 检查 notebook 取数落地的数据完整性。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import exists
from src.utils.config import load_yaml

REQUIRED = ["trade_calendar", "stock_universe", "stock_daily", "stock_industry",
            "index_daily", "index_constituents", "etf_info", "etf_daily"]

if __name__ == "__main__":
    missing = [n for n in REQUIRED if not exists("raw", n)]
    macro_cfg = load_yaml("macro_indicators")
    for cat, spec in macro_cfg["categories"].items():
        for ind in spec["indicators"]:
            if not ind.get("enabled", True):
                continue
            key = ind["name"]
            if not exists("raw", f"macro_{key}"):
                missing.append(f"macro_{key}")
    if missing:
        print("缺少以下数据, 请运行 notebooks/01_choice_data_fetch.ipynb:")
        for m in missing:
            print("  -", m)
    else:
        print("数据完整 ✓")
