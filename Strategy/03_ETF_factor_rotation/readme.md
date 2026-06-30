# 全市场 ETF 多因子月度轮动策略

后复权市价口径、月度调仓、严格 PIT、含成本、经 walk-forward 验证的中国 ETF 多因子轮动策略。
完整研究报告见 [`报告_最终版.md`](报告_最终版.md)（含图表与全部稳健性检验）。

> **口径**：后复权市价（iFinD `close_hfq`）、含 5bps 成本、区间 2018-01-02 ~ 2026-06-05。
> 所有结论以 **后复权 + walk-forward + 含成本** 口径为准。

---

## 终版策略

**V3 = 后复权市价 · 月度 · 最小方差加权**

| 指标 | V3（终版） | 沪深300 | 等权 ETF 篮子 |
|---|---:|---:|---:|
| 年化收益 | **10.3%** | 2.0% | 6.6% |
| Sharpe(rf=0) | **1.02** | 0.11 | 0.40 |
| 最大回撤 | **-12.4%** | -45.6% | -33.3% |
| 累计(2018-2026) | **+120.8%** | +17.8% | +67.3% |

**低回撤备选 = 多资产分层组合**（类内因子选股 + 类间趋势/风险平价）：年化 7.5% / **Sharpe 1.24** / 回撤 **-9%**，弱市更稳。

### 构造
- **因子**：`0.45·z(combo_eff_accel) + 0.35·z(momentum_12_1) + 0.20·z(fund_hit_rate_20) − 0.15·z(vol_60d) + 0.10·z(max_drawdown_60d)`
  （`combo_eff_accel = z(路径效率20d) + z(收益加速度20/60d)`）
- **组合**：全市场场内 ETF（≥280 日历史，约 1100 只）→ Top-20、同主题≤3、单票≤12% → **最小方差加权**（收缩协方差、PIT）→ 波动目标 18% → 残余进货币 ETF 511880。
- **降换手**：名次滞后带 `buffer_rank=35` + **部分再平衡 `lam=0.4`**（每月只朝目标移动 40%）→ 年化双边换手约 6.7x。
- **执行**：信号 T 日月末收盘后生成，`searchsorted(right)+shift(1)` 确保 T+1 后才计收益，无未来函数。

---

## 一句话诚实结论

在一个八年只涨 6.6%/年（等权篮子）、2%/年（沪深300）的市场里，本策略做到**跑赢大盘 3–4%/年、回撤砍到大盘 1/4**。
**不加杠杆，这段行情的长仓收益天花板约 10–12% / Sharpe ~1.0；绝对收益受限于市场，而非策略缺陷。**

---

## 稳健性（全部在 `报告_最终版.md` 有证据）

- ✅ 无未来函数、测试集选参隔离、walk-forward 真样本外拼接
- ✅ 成本敏感性（5–50bps）、容量/平方根冲击（甜区 ≤1–3 亿）
- ✅ 消融归因（各风控/分散模块贡献）、bootstrap Sharpe 置信区间、滚动 Sharpe、分年度
- ✅ 幸存者偏差（并入清盘 ETF，样本外偏差 +0.03）
- ⚠️ Sharpe ~1.0 统计支撑勉强（bootstrap 95% CI 偏宽）；收益集中于趋势年（2020/2025）

## 提升上限的探索（均做真 OOS，诚实留档）

| 方向 | 结论 |
|---|---|
| 删/拆因子、按 IC 重配 | ❌ OOS 不成立（样本内幻觉） |
| 资金流因子（场内份额，反向）| IC 显著(t-2.34)但加进组合 walk-forward 仅胜 3/8 年，边际无用 |
| 主动基金入池 | ❌ 理想化仅 Sharpe 0.91，计赎回费即转负 |
| 日内数据 | ❌ 快照无历史、对月度无增量、数据量不现实 |
| **跨资产 GTAA（黄金/债/港/美/商品）** | ✅ 真有效：单独 Sharpe 1.22、与 V3 合并 1.24/回撤-9%（即"低回撤备选"）|

唯一能同时提收益与 Sharpe 的是 **杠杆 / 市场中性（股指期货对冲，纯 alpha Sharpe 1.16–1.48）**，均需衍生品。

---

## 复现

```bash
pip install -r requirements.txt
python3 -m etf_factor_strategy.cli                      # 终版 V3(最小方差) -> outputs_v3_final/
python3 -m etf_factor_strategy.cli --weighting inv_vol  # 低回撤版(逆波动)
python3 factor_diagnostics.py    # 因子 IC/分层/相关/衰减
python3 robustness_v2.py         # 分年度/滚动/bootstrap/成本
python3 multi_asset_v2.py        # 跨资产分层组合
python3 make_charts.py           # 报告所有图 -> figures/
python3 walkforward_hfq.py       # 市价 walk-forward
```

**数据**：行情/份额/估值数据由 iFinD HTTP API 拉取（见 `notebooks/` 与 `*_test.py`），体量大、未入库。
拉数据前设环境变量（**切勿把 token 硬编码提交**）：

```bash
export IFIND_REFRESH_TOKEN="你的_refresh_token"
```

> 数据落地于本地 `dataset/` 目录（`etf_market_ifind.db` 等），不在本仓库。

---

## 关键文件

| 文件 | 内容 |
|---|---|
| `报告_最终版.md` | 完整研究报告（含图表、稳健性、实盘清单）|
| `etf_factor_strategy/engine.py` | 因子/打分/V3权重(最小方差)/回测(含 lam) |
| `etf_factor_strategy/cli.py` | 一键复现入口（`--weighting` 切 V2/V3）|
| `hfq_common.py` | 后复权市价回测工具 |
| `factor_diagnostics.py` / `param_sweep.py` / `make_charts.py` | 因子审查 / 参数敏感性 / 图表 |
| `robustness_v2.py` / `robustness_hfq.py` / `walkforward_hfq.py` | 稳健性 / 消融 / walk-forward |
| `multi_asset_v2.py` / `explore_weighting.py` | 跨资产分层 / 加权方案对比 |
| `flow_*.py` / `active_fund_test.py` / `value_data_pull.py` | 新信号探索（资金流/主动基金/价值）|
| `notebooks/` | iFinD 数据抓取 notebook |
| `figures/` | 报告所有图 |
