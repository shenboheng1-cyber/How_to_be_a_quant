import unittest

import pandas as pd

from etf_factor_strategy.engine import (
    backtest_monthly_strategy,
    compute_factor_panel,
    make_monthly_weights,
    make_robust_monthly_weights,
    tradable_codes_at_date,
    score_factors_with_weights,
    score_factors,
    zscore_cross_section,
)


class FactorEngineTest(unittest.TestCase):
    def test_zscore_cross_section_uses_population_std_and_ignores_missing(self):
        row = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": None})

        scored = zscore_cross_section(row)

        self.assertAlmostEqual(scored["A"], -1.2247448714)
        self.assertAlmostEqual(scored["B"], 0.0)
        self.assertAlmostEqual(scored["C"], 1.2247448714)
        self.assertTrue(pd.isna(scored["D"]))

    def test_score_factors_applies_requested_three_factor_weights(self):
        factors = pd.DataFrame(
            {
                "date": ["2024-01-31", "2024-01-31", "2024-01-31"],
                "fund_code": ["A", "B", "C"],
                "combo_eff_accel": [1.0, 2.0, 3.0],
                "momentum_12_1": [3.0, 2.0, 1.0],
                "fund_hit_rate_20": [1.0, 1.0, 3.0],
            }
        )

        scored = score_factors(factors)
        by_code = scored.set_index("fund_code")

        self.assertAlmostEqual(by_code.loc["A", "score"], -0.2638958434)
        self.assertAlmostEqual(by_code.loc["B", "score"], -0.1414213562)
        self.assertAlmostEqual(by_code.loc["C", "score"], 0.4053171996)

    def test_score_factors_with_weights_supports_risk_penalties(self):
        factors = pd.DataFrame(
            {
                "date": ["2024-01-31", "2024-01-31", "2024-01-31"],
                "fund_code": ["A", "B", "C"],
                "combo_eff_accel": [1.0, 2.0, 3.0],
                "momentum_12_1": [1.0, 2.0, 3.0],
                "fund_hit_rate_20": [1.0, 2.0, 3.0],
                "vol_60d": [3.0, 2.0, 1.0],
                "max_drawdown_60d": [-0.30, -0.20, -0.10],
            }
        )

        scored = score_factors_with_weights(
            factors,
            {
                "combo_eff_accel": 0.45,
                "momentum_12_1": 0.35,
                "fund_hit_rate_20": 0.20,
                "vol_60d": -0.15,
                "max_drawdown_60d": 0.10,
            },
            score_column="risk_adjusted_score",
        )
        by_code = scored.set_index("fund_code")

        self.assertGreater(by_code.loc["C", "risk_adjusted_score"], by_code.loc["B", "risk_adjusted_score"])
        self.assertGreater(by_code.loc["B", "risk_adjusted_score"], by_code.loc["A", "risk_adjusted_score"])

    def test_compute_factor_panel_matches_factor_definitions(self):
        dates = pd.bdate_range("2023-01-02", periods=280)
        prices = pd.DataFrame(
            {
                "A": [100.0 + i for i in range(280)],
                "B": [100.0 + 0.5 * i for i in range(280)],
            },
            index=dates,
        )

        factors = compute_factor_panel(prices)
        latest = factors[factors["date"] == dates[-1].strftime("%Y-%m-%d")].set_index("fund_code")

        ret20_a = prices["A"].iloc[-1] / prices["A"].iloc[-21] - 1.0
        ret60_a = prices["A"].iloc[-1] / prices["A"].iloc[-61] - 1.0
        daily_a = prices["A"].pct_change()
        expected_eff_a = ret20_a / daily_a.abs().rolling(20).sum().iloc[-1]
        expected_accel_a = ret20_a - ret60_a / 3.0
        expected_mom_a = prices["A"].iloc[-22] / prices["A"].iloc[-253] - 1.0

        self.assertAlmostEqual(latest.loc["A", "efficiency_20d"], expected_eff_a)
        self.assertAlmostEqual(latest.loc["A", "fund_ret_accel_20_60"], expected_accel_a)
        self.assertAlmostEqual(latest.loc["A", "momentum_12_1"], expected_mom_a)
        self.assertAlmostEqual(latest.loc["A", "fund_hit_rate_20"], 1.0)
        self.assertAlmostEqual(latest.loc["A", "downside_vol_60d"], 0.0)
        self.assertAlmostEqual(latest.loc["A", "max_drawdown_60d"], 0.0)

    def test_make_monthly_weights_selects_top_n_equal_weight_on_month_end(self):
        scored = pd.DataFrame(
            {
                "date": ["2024-01-31", "2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"],
                "fund_code": ["A", "B", "C", "A", "B"],
                "score": [3.0, 2.0, 1.0, 1.0, 4.0],
            }
        )

        weights = make_monthly_weights(scored, top_n=2)

        jan = weights[weights["date"] == "2024-01-31"].set_index("fund_code")["weight"]
        feb = weights[weights["date"] == "2024-02-29"].set_index("fund_code")["weight"]
        self.assertEqual(set(jan.index), {"A", "B"})
        self.assertEqual(set(feb.index), {"A", "B"})
        self.assertAlmostEqual(jan.loc["A"], 0.5)
        self.assertAlmostEqual(jan.loc["B"], 0.5)
        self.assertAlmostEqual(feb.loc["A"], 0.5)
        self.assertAlmostEqual(feb.loc["B"], 0.5)

    def test_backtest_rebalance_replaces_old_positions_instead_of_accumulating(self):
        dates = pd.bdate_range("2024-01-31", periods=25)
        prices = pd.DataFrame(
            {
                "A": [100.0 + i for i in range(25)],
                "B": [100.0 for _ in range(25)],
                "C": [100.0 + 2 * i for i in range(25)],
            },
            index=dates,
        )
        weights = pd.DataFrame(
            {
                "date": ["2024-01-31", "2024-01-31", "2024-02-15"],
                "fund_code": ["A", "B", "C"],
                "weight": [0.5, 0.5, 1.0],
            }
        )

        _, effective_weights = backtest_monthly_strategy(prices, weights)

        after_second_rebalance = effective_weights.loc[pd.Timestamp("2024-02-16")]
        self.assertAlmostEqual(after_second_rebalance["A"], 0.0)
        self.assertAlmostEqual(after_second_rebalance["B"], 0.0)
        self.assertAlmostEqual(after_second_rebalance["C"], 1.0)
        self.assertAlmostEqual(after_second_rebalance.sum(), 1.0)

    def test_backtest_partial_rebalancing_moves_fraction_toward_target(self):
        dates = pd.bdate_range("2024-01-31", periods=6)
        prices = pd.DataFrame({"A": [100.0] * 6, "B": [100.0] * 6}, index=dates)
        weights = pd.DataFrame({"date": ["2024-01-31"], "fund_code": ["A"], "weight": [1.0]})

        _, eff_full = backtest_monthly_strategy(prices, weights, rebalance_lambda=1.0)
        _, eff_half = backtest_monthly_strategy(prices, weights, rebalance_lambda=0.5)

        first = pd.Timestamp("2024-02-01")
        self.assertAlmostEqual(eff_full.loc[first, "A"], 1.0)
        self.assertAlmostEqual(eff_half.loc[first, "A"], 0.5)  # 从 0 只移动 50% 朝目标
        self.assertAlmostEqual(eff_half.loc[first, "B"], 0.0)

    def test_backtest_aggregates_duplicate_weight_rows(self):
        dates = pd.bdate_range("2024-01-31", periods=5)
        prices = pd.DataFrame({"A": [100, 101, 102, 103, 104], "B": [100, 100, 100, 100, 100]}, index=dates)
        weights = pd.DataFrame(
            {
                "date": ["2024-01-31", "2024-01-31", "2024-01-31"],
                "fund_code": ["A", "A", "B"],
                "weight": [0.2, 0.3, 0.5],
            }
        )

        _, effective_weights = backtest_monthly_strategy(prices, weights)

        first_effective = effective_weights.loc[pd.Timestamp("2024-02-01")]
        self.assertAlmostEqual(first_effective["A"], 0.5)
        self.assertAlmostEqual(first_effective["B"], 0.5)

    def test_make_robust_monthly_weights_caps_theme_and_single_name_weight(self):
        scored = pd.DataFrame(
            {
                "date": ["2024-01-31"] * 6,
                "fund_code": ["T1", "T2", "T3", "T4", "B1", "G1"],
                "score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                "vol_60d": [0.10, 0.10, 0.10, 0.10, 0.05, 0.20],
            }
        )
        meta = pd.DataFrame(
            {
                "fund_code": ["T1", "T2", "T3", "T4", "B1", "G1"],
                "fund_name": ["通信ETF一", "通信ETF二", "通信ETF三", "通信ETF四", "信用债ETF", "黄金ETF"],
                "fund_type": ["指数型-股票", "指数型-股票", "指数型-股票", "指数型-股票", "指数型-固收", "指数型-其他"],
            }
        )
        prices = pd.DataFrame(
            {
                code: [100.0 + i for i in range(260)]
                for code in ["T1", "T2", "T3", "T4", "B1", "G1", "511880"]
            },
            index=pd.bdate_range("2023-01-02", periods=260),
        )

        weights = make_robust_monthly_weights(
            scored,
            prices,
            meta,
            top_n=6,
            max_per_theme=3,
            max_weight=0.12,
            cash_code="511880",
            volatility_target=1.0,
        )

        selected = set(weights["fund_code"])
        self.assertNotIn("T4", selected)
        risky = weights[weights["fund_code"] != "511880"]
        self.assertLessEqual(risky["weight"].max(), 0.12)
        self.assertAlmostEqual(weights["weight"].sum(), 1.0)

    def test_tradable_codes_require_recent_nav_and_missing_data_limits(self):
        dates = pd.bdate_range("2023-01-02", periods=260)
        prices = pd.DataFrame(
            {
                "GOOD": [100.0 + i for i in range(260)],
                "NEW": [None] * 230 + [100.0 + i for i in range(30)],
                "STALE": [100.0 + i for i in range(254)] + [None] * 6,
                "GAPPY": [100.0 + i if i % 3 else None for i in range(260)],
            },
            index=dates,
        )

        tradable = tradable_codes_at_date(
            prices,
            dates[-1],
            recent_days=5,
            max_missing_60=0.10,
            max_missing_252=0.20,
        )

        self.assertIn("GOOD", tradable)
        self.assertNotIn("NEW", tradable)
        self.assertNotIn("STALE", tradable)
        self.assertNotIn("GAPPY", tradable)


if __name__ == "__main__":
    unittest.main()
