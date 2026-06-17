# 03 · ETF 多因子月度轮动策略

全市场 ETF 的月度多因子轮动。**后复权市价（close_hfq）口径、含 5bps 成本**，
回测区间 2018-01-02 ~ 2026-06-05。经无未来函数 / walk-forward 真样本外 / 消融 /
成本 / 容量 / 幸存者偏差全套稳健性检验。

> 📄 **完整研究报告**：[`报告_最终版.md`](报告_最终版.md)（含因子审查、稳健性、参数敏感性、实盘清单、全部图表）。
> 本页为摘要。研究纪要（早期版本）见 [`docs/research_notes.md`](docs/research_notes.md)。

---

## 1. 一句话定位

**低回撤、防御型的小—中资金策略**：全周期年化 7.6%、Sharpe ~1.0、最大回撤仅 −7.6%。
强项是**控回撤**而非高收益——下跌/震荡年（2018/2022/2023）顶得住，强牛年（2019/2020）
会明显跑输。适合 0.5–3 亿资金做规则化低波动配置或前向验证。

## 2. 关键业绩（后复权市价，含成本，2018–2026）

| 指标 | 策略 V2 | 沪深300 | 全市场等权 ETF 篮子 |
|------|---:|---:|---:|
| 累计收益 | **+80.6%** | +17.8% | +67.3% |
| 年化收益 | **+7.6%** | +2.0% | +6.6% |
| 年化波动 | 7.4% | 19.2% | 16.2% |
| **Sharpe(rf=0)** | **1.02** | 0.11 | 0.40 |
| 最大回撤 | **−7.6%** | −45.6% | −33.3% |
| Calmar | **1.00** | 0.04 | 0.20 |

> 来源：[`outputs_v2_final/summary.json`](outputs_v2_final/summary.json)。
> **诚实警告**：纳入 2018-2019 后 Sharpe 统计支撑仅勉强为正（bootstrap 95% CI [0.39, 1.64]、
> P(Sharpe>1)≈50%）。卖点是低回撤/防御，不是高 Sharpe。详见报告 §5.5。

![净值曲线](figures/01_nav.png)

## 3. 策略构造（最终版 V2）

合成打分 `score = Σ wᵢ·z(factorᵢ)`（每日横截面 z-score）：

| 因子 | 权重 | 类型 |
|------|---:|------|
| `combo_eff_accel`（路径效率+收益加速度） | +0.45 | 选股 · 收益引擎 |
| `momentum_12_1`（12月动量跳过最近1月） | +0.35 | 选股 · 正交分散 |
| `fund_hit_rate_20`（近20日上涨日占比） | +0.20 | 选股 · 弱（建议降权，见报告 §3） |
| `vol_60d`（60日波动） | −0.15 | 风险 · 惩罚高波动 |
| `max_drawdown_60d`（60日回撤） | +0.10 | 风险 · 偏好浅回撤 |

组合：Top-20、同主题 ≤3、单票 ≤12%、逆波动加权 + 上限再分配、波动目标 0.18、
残仓进货币 ETF（511880）。**已去弱市择时**。换手控制：名次滞后带 `buffer_rank=35`
+ 部分再平衡 `lambda=0.4`（年化双边换手 13.2×→5.8×）。信号 T 日收盘算、**T+1 执行**（严格 PIT）。

## 4. 稳健性一览

| 检验 | 结论 | 产物 |
|------|------|------|
| 无未来函数 | ✅ T+1 执行、单测覆盖 | 代码 + `tests` |
| Walk-forward 真样本外 | ✅ 2023–2026 拼接 Sharpe 1.07 | `outputs_walk_forward_hfq/` |
| 消融（模块归因） | ✅ 跨资产分散+逆波动+风险因子三层 | `outputs_robustness_hfq/ablation.csv` |
| 成本敏感性 | ✅ 30bps 下 Sharpe 仍 0.81 | `outputs_robustness_v2/cost_sweep.csv` |
| 容量/冲击 | ✅ ≤3 亿甜区，5 亿软上限 | 报告 §5.3 |
| Bootstrap / 滚动 Sharpe | ⚠️ 支撑勉强为正 | `outputs_robustness_v2/` |
| 幸存者偏差 | ✅ 并入 122 只清盘 ETF，样本外偏差 +0.03 | `outputs_survivorship/` |
| 五因子扩展 | ⚠️ walk-forward 选不出，不上 | 报告 §8 |

## 5. 目录结构

```
03_ETF_factor_rotation/
├── 报告_最终版.md              # ★ 完整研究报告（含图）
├── etf_factor_strategy/        # 核心引擎包 (data/engine/cli/walk_forward_validate/oos_optimize)
├── hfq_common.py               # 后复权市价口径公共层
├── strategy_v2.py              # V2 构建 + V1/V2 对照 + 容量曲线
├── robustness_v2.py            # V2 分年/滚动/bootstrap/成本
├── robustness_hfq.py           # 消融 + 基准
├── walkforward_hfq.py          # 市价 walk-forward
├── factor_diagnostics.py       # 因子审查 (IC/分层/相关/衰减, 最终5因子)
├── param_sweep.py              # 单参数敏感性
├── make_charts.py              # 生成 figures/
├── build_report_docs.py        # 报告 -> docx/pdf (需 pandoc + xelatex)
├── survivorship_check.py / fetch_delisted_etf.py  # 幸存者偏差
├── notebooks/ifind_etf_history.ipynb  # 抓取后复权市价 DB
├── figures/                    # 报告图 01–07
├── outputs_*/                  # 最终结果 (小体量 CSV/JSON)
└── docs/research_notes.md      # 早期研究纪要
```

## 6. 复现

> **数据依赖（不入库）**：行情/净值存于本地 SQLite（数百 MB–GB）。请在
> [`etf_factor_strategy/data.py`](etf_factor_strategy/data.py) 的 `DEFAULT_DATA_DIR` 改为你本地数据目录
> （需含 `etf_market_ifind.db`、`nav_store.db`、`idx_store.db`、`bulk_universe.json`）。
> 后复权市价 DB 由 [`notebooks/ifind_etf_history.ipynb`](notebooks/ifind_etf_history.ipynb) 抓取生成。

```bash
pip install -r requirements.txt

python3 -m etf_factor_strategy.cli   # 最终版 V2 一键回测 -> outputs_v2_final/
python3 factor_diagnostics.py        # 因子 IC/分层/相关/衰减 -> outputs_factor_diag/
python3 param_sweep.py               # 参数敏感性 -> outputs_param_sweep/
python3 make_charts.py               # 报告图 -> figures/
python3 robustness_v2.py             # V2 分年/滚动/bootstrap/成本 -> outputs_robustness_v2/
python3 robustness_hfq.py            # 消融/基准 -> outputs_robustness_hfq/
python3 walkforward_hfq.py           # 市价 walk-forward -> outputs_walk_forward_hfq/
python3 fetch_delisted_etf.py        # 拉清盘 ETF (需 Tushare token)
python3 survivorship_check.py        # 幸存者偏差 -> outputs_survivorship/
python3 build_report_docs.py         # 报告 -> docx/pdf (需 pandoc + xelatex)
```

脚本以自身所在目录为根，请在本文件夹根目录下运行。区间起点由 `hfq_common.py` 的 `START`
与 `cli.py` 的 `--start` 统一控制（默认 2018-01-02）。
