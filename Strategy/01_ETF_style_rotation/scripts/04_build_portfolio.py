"""Step 4: 信号->指数打分->贪心选择8指数->ETF映射与权重。
输出: data/processed/target_weights.parquet (信号日, etf, weight)
TODO: 需要 指数月度暴露(成分股2年回归) 流水线, 数据落地后组装。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    print("待数据落地后组装: index_scoring + greedy_select + etf_mapping")
    # 伪代码 (每个信号日 t, 全部仅用 <=t 信息):
    # strength = map_strength(composite.loc[t], strength_map)
    # expo_stock = stock_style_exposures(monthly_ret, monthly_factor_ret, asof=最近月末)
    # E = {idx: index_exposure(成分股@最近月末, expo_stock) for idx in 指数池}
    # score = style_score(E, strength)
    # selected, norm = greedy_select(score, E, z=8, w_d=0.5)
    # etf = {i: select_etf_for_index(i, t, ...) for i in selected}
    # w = build_target_weights(selected, norm, etf)
