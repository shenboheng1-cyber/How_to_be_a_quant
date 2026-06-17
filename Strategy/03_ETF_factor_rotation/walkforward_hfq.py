"""在后复权市价(close_hfq)口径上重跑 walk-forward，对照 NAV 口径是否仍退化为同一候选。"""
from __future__ import annotations
import json
import pandas as pd

from etf_factor_strategy.engine import compute_factor_panel
from etf_factor_strategy.walk_forward_validate import (
    FOLDS, build_candidates, run_candidate, period_metrics, selection_score,
    prefix_metrics, month_end_dates, slice_and_normalize, stitch_fold_curves,
    period_metrics_from_nav)
import hfq_common as H

OUT = H.ROOT / "outputs_walk_forward_hfq"
OUT.mkdir(exist_ok=True)
COST_BPS = 5.0


def main():
    uni = H.load_etf_universe(data_dir=H.DEFAULT_DATA_DIR)
    px, _, _ = H.load_hfq()
    uni = uni[uni["fund_code"].isin(px.columns)].copy()
    market = H.hs300()
    print(f"HFQ池 {px.shape[1]} 只；计算因子面板…")
    fac = compute_factor_panel(px)
    fac = fac[fac["date"].isin(month_end_dates(fac["date"]))].copy()
    candidates = build_candidates()
    print(f"候选 {len(candidates)} × {len(FOLDS)} 折…")

    all_rows, chosen, curves = [], [], []
    for fold_name, ts, te, vs, ve in FOLDS:
        fold_rows = []
        for cand in candidates:
            eq = run_candidate(cand, fac, px, uni, market, COST_BPS)
            row = {"fold": fold_name, **cand["flat"],
                   **prefix_metrics("train", period_metrics(eq, ts, te)),
                   **prefix_metrics("test", period_metrics(eq, vs, ve))}
            row["selection_score"] = selection_score(row)
            fold_rows.append(row); all_rows.append(row)
        fr = pd.DataFrame(fold_rows).sort_values(
            ["selection_score", "train_calmar", "train_sharpe_rf0"], ascending=False)
        best = fr.iloc[0].to_dict()
        chosen.append(best)
        bc = next(c for c in candidates if c["flat"]["candidate_id"] == best["candidate_id"])
        tc = slice_and_normalize(run_candidate(bc, fac, px, uni, market, COST_BPS), vs, ve)
        tc["fold"] = fold_name
        curves.append(tc)
        print(f"  {fold_name} -> {best['candidate_id']} | test_sharpe={best['test_sharpe_rf0']:.2f}")

    stitched = stitch_fold_curves(curves)
    st = period_metrics_from_nav(stitched["date"], stitched["stitched_nav"])
    print("\n=== 拼接 OOS (2023-2026, 市价口径) ===")
    print(f"年化 {st['annual_return']:+.1%} | Sharpe {st['sharpe_rf0']:.2f} | "
          f"MDD {st['max_drawdown']:.1%} | Calmar {st['calmar']:.2f}")
    distinct = set(c["candidate_id"] for c in chosen)
    print(f"各折选中候选: {[c['candidate_id'] for c in chosen]}")
    print(f"=> {'仍退化为同一候选(选择过程无判别力)' if len(distinct)==1 else f'选中 {len(distinct)} 种不同候选'}")

    pd.DataFrame(all_rows).to_csv(OUT / "wf_all.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(chosen).to_csv(OUT / "wf_chosen.csv", index=False, encoding="utf-8-sig")
    (OUT / "wf_summary.json").write_text(json.dumps(
        {"basis": "close_hfq", "cost_bps": COST_BPS, "stitched_test": st,
         "chosen_ids": [c["candidate_id"] for c in chosen]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出 -> {OUT}")


if __name__ == "__main__":
    main()
