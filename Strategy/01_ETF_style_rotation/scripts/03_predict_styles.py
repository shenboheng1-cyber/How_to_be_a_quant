"""Step 3: 三标签 Logistic 滚动预测, 输出 Composite 得分。
输出: data/processed/composite_scores.parquet
自检: 准确率表 vs 报告表6 (Trend ≈70%)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_yaml
from src.utils.io import load_parquet, save_parquet
from src.prediction.labels import build_all_labels
from src.prediction.logistic import predict_all, composite_score, accuracy_table

if __name__ == "__main__":
    cfg = load_yaml("strategy")["prediction"]
    fr = load_parquet("processed", "factor_returns_weekly")
    ms = load_parquet("processed", "macro_scores_weekly")

    labels = build_all_labels(fr, cfg["label_window"])
    preds = predict_all(fr, ms, labels,
                        window=cfg["logistic_window"], ar_lags=cfg["ar_lags"])
    comp = composite_score(preds)
    save_parquet(comp, "processed", "composite_scores")

    acc = accuracy_table(preds)
    print("样本外准确率 (对账表6, Trend应≈70%):")
    print(acc.round(4))
    acc.to_csv("outputs/label_accuracy.csv", encoding="utf-8-sig")
