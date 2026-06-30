# A股多因子 Alpha 研究框架（L0–L4）

> 一个从 **无前视回测引擎** → **多路线因子生成** → **新数据（订单流）正交 alpha** → **ML 合成** → **扣成本对标中证500** 的端到端量化研究项目。
> 数据：CSMAR 日频全 A 股 + 高频微观结构指标（2015–2025）。强调**研究规范、防泄漏、与诚实的结论**。

---

## TL;DR — 样本外结果（防泄漏 walk-forward，扣双边千3）

| 指标 | 多头(可交易) | 多空(学术口径) | 中证500 |
|---|---|---|---|
| 净年化 | 19.8% | 40.7% | 2.4% |
| 净夏普 | 0.83 | **2.96** | 0.11 |
| 对中证500 **信息比率 IR** | **1.08** | — | — |
| 年化净超额 | **16.4%** | — | — |

ML 合成把 ~230 个因子合成后样本外多空夏普 **2.96**；扣真实成本后多头对中证500 **IR≈1.08、净超额16%**，且对成本稳健（千3→千8 仍正）。

---

## 项目分层（L0–L4）

| 层 | 内容 | 一句话亮点 |
|---|---|---|
| **L0** 引擎 | 无前视的单因子测试引擎 + 防泄漏单元测试 | 用代码证明回测不作弊 |
| **L1** 经典因子 | 10 个异象因子复现 | **发现 12-1 动量在 A 股失效**（散户市、反转主导） |
| **L2** 原创因子 | 52 周高点锚定 + 正交化 + Fama-MacBeth | 朴素看无效→**正交化后证明被反转掩盖、实则显著**（IC t=2.9） |
| **L2.5** 因子工厂 | 算子×信号批量生成 105 个 + 多重检验 | Bonferroni/FDR 校正 + **用相关性自曝冗余** |
| — | 国泰君安 191 因子 | **写了个公式解释器**跑 191 条公式字符串（182/191 可算） |
| — | 50 手写公式 + 遗传规划挖掘器 | GP 自由进化**收敛到成交额波动率**→实验证明价量信息天花板 |
| — | 高频微观结构（新数据） | 订单流/订单拆分因子，与价量**正交**（`order_density` 等，相关仅 0.2） |
| **L3** ML 合成 | LightGBM + 岭回归 + IC加权 + 等权对比，purged CV | **LightGBM≈岭回归**→价值在组合+加权不在非线性；等权失败；微观因子进重要性 Top30 |
| **L4** 组合回测 | 扣成本、对标中证500、成本敏感性、容量 | 把毛夏普 2.95 做成可信的净 IR 1.08 |

每一层都有一个**诚实、反直觉的发现**——这正是项目的核心。

---

## 仓库结构

```
quantlib/                     # 可复用引擎（项目的"产品"）
  data.py                     # 面板加载 + 滚动因子（DuckDB，无前视 fwd_ret）
  universe.py                 # 股票池（剔 ST/次新/涨跌停；停牌靠快照天然剔除）
  preprocess.py               # 去极值 → 市值中性化 → 标准化 → 正交化
  evaluate.py                 # IC/RankIC/ICIR、分层回测、Fama-MacBeth
  ml.py                       # L3：purged&embargoed walk-forward + LightGBM/岭回归
  backtest.py                 # L4：组合构建 + 换手成本 + 对标中证500 + IR/回撤
  plotting.py
  factors/                    # 经典 + 原创(52周高点)因子
  alpha/                      # 因子工厂：算子库 + 宽矩阵 + 批量alpha + 多重检验
    gtja_ops.py / gtja191.py  #   国泰君安191：公式解释器 + 191条公式
    gp_miner.py               #   遗传规划因子挖掘器
  microstructure.py           # 高频订单流/已实现指标因子（spec 驱动）
research/                     # 01–09 研究脚本 + 叙事式 notebook（含内嵌图）
tests/test_no_lookahead.py    # 防前视单元测试（4 项全过）
results/                      # 汇总 CSV + 图（png）+ 因子目录 json
docs/                         # 设计文档 + 数据下载清单
```

## 关键诚实结论（面试脚本）

1. **动量在 A 股失效**（RankIC≈−0.016 不显著）——自己用数据算出并解释成因。
2. **52 周高点因子被反转掩盖**——正交化后纯锚定信号显著（t=2.9），FM 中边际。
3. **450+ 价量公式全坍缩到流动性/反转/低波几个 premia**——遗传规划自由搜索也收敛于此，证明 OHLCV 信息天花板。
4. **订单流是真·正交新 alpha**：`order_density`/`avg_order_size_imb` ICIR≈0.7、与价量相关仅 0.2，日频价量算不出。
5. **ML 的真相**：LightGBM 几乎没赢过正则化岭回归——价值在"组合+加权"，不在非线性魔法；等权合成则彻底失败。
6. **成本与容量**：扣千3后多头 IR 仍 1.08，但持仓中位市值仅 ~26 亿，**偏小盘、容量受限**。

## 局限（诚实声明）

- 数据库缺行业分类表 → 仅做了**市值中性化**，未做行业中性化。
- 多空夏普 2.96 是**学术口径**（A 股做空难/贵/受限）；可交易的是多头 IR 1.08。
- 策略**偏小盘**、容量受限；上规模冲击成本会显著上升。

---

## 复跑

```bash
# 数据不在仓库内（CSMAR 日频，41GB）。用环境变量指向本地数据根：
export CSMAR_DATA_ROOT=/path/to/CSMAR        # 含 mart/fact_stock_daily.parquet
# macOS 上 LightGBM 需 OpenMP：
export DYLD_FALLBACK_LIBRARY_PATH=/opt/anaconda3/lib

python tests/test_no_lookahead.py            # 防前视测试
python research/01_single_factor_test.py     # L1 经典因子
python research/08_ml_synthesis.py           # L3 ML 合成
python research/09_backtest.py               # L4 带成本回测
```

依赖：`python>=3.8, pandas, numpy, duckdb, matplotlib, lightgbm`。
数据原始层（CSMAR parquet/duckdb）与含账号的配置**不入库**（见 `.gitignore`），用脚本可复现。
