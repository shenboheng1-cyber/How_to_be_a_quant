# 幸存者偏差量化 (Tushare 清盘 ETF 补全后)

日期 2026-06-16 · 数据: Tushare fund_basic+fund_nav 取已停更 ETF; 真停更判定=净值末日<2026-04

## 数据
- Tushare 交易所基金 status='D' 标签不可靠(含新上市 ETF, delist_date 多为空)。
- 改用**净值末日**判定: 123 只补到净值的 ETF 中, 122 只真停更(2018:8/2019:5/2020:9/2021:14/2022:27/2023:22/2024:20/2025:7)。
- 并入回测 116 只(满足 280 日历史)。

## 结果 (固定参数 base+risk_light+balanced, 10bps)
| 区间 | survivor-only(有偏) | 含清盘ETF(修正) | Sharpe偏差 |
|------|------:|------:|------:|
| 样本外 2023-2026 | 年化12.0% Sharpe**1.23** MDD-7.6% | 年化12.8% Sharpe**1.20** MDD-7.5% | **-0.03** |
| 全样本 2020-2026 | 年化12.3% Sharpe1.36 MDD-7.6% | 年化12.6% Sharpe1.25 MDD-7.5% | -0.11 |

## 关键发现
- **幸存者偏差很小**(样本外 Sharpe -0.03, 全样本 -0.11)。策略 edge 不是幸存者假象。
- 修正版**曾持有 68 只清盘 ETF**(73 调仓月, 均 5% 权重)→ 它们清盘前在涨被选中。
- 但清盘=返还NAV(非归零)+动量反转卖出+tradable过滤退出 → 净效应仅略增波动, MDD 未变差。
- 全样本偏差更大因 2020-2022 死的 ETF 更多(71 只在 2023 前清盘)。

## 结论
动量策略真实样本外 Sharpe ≈ **1.20**(扣幸存者+10bps成本)。与 multi_asset 组合结论不变(~30%动量, 主要收益平滑回撤)。
剩余待查: 收益对 2025 单年的集中度(与幸存者无关)。

## 复现
本地终端 `python3 fetch_delisted_etf.py`(需 Tushare token, 本沙箱 DNS 黑洞 Tushare) → 产出 delisted_etf_nav.parquet;
再 `python3 survivorship_check.py`(本地 nav_store + 清盘净值)。
