"""Step 1: 计算周度风格因子收益 (Phase1=分位多空 / Phase2=Barra截面回归)。
输出: data/processed/factor_returns_weekly.parquet
TODO: 数据字段确认后, 在此组装 描述变量->预处理->合成->因子收益 的完整流水线。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_yaml

if __name__ == "__main__":
    cfg = load_yaml("strategy")
    phase = cfg["factor_model"]["phase"]
    print(f"Phase {phase} 因子收益计算 — 待数据落地后实现主流程")
    # 伪代码:
    # 1. 加载 stock_daily / stock_industry / stock_financials
    # 2. 对每个周度调仓日 t: 计算23个描述变量 -> standardize_descriptor -> synthesize
    # 3. Phase1: factor_returns_quantile.build_factor_returns
    #    Phase2: factor_returns_barra.build_factor_returns
    # 4. 落地 processed/factor_returns_weekly.parquet
    # 自检: 图1(年化收益率标准差≈5%)、图2(volatility最优占比≈29.28%)
