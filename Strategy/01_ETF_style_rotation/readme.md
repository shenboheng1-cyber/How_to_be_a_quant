# 宏观信息驱动的宽基 ETF 风格轮动策略 · 复现

复现中国银河证券研报《宏观信息驱动的宽基ETF风格轮动策略 —— ETF策略系列》（2025-04-30，马普凡）。
**注意：这是宽基ETF的 Barra 风格因子轮动，不是行业轮动。**

## 流水线（周度，每周最后交易日发信号，T+1 执行）

```
16类宏观EDB ──熵权法(52周窗)──> 5类宏观综合得分 ─┐
                                              ├─ Logistic(104周滚动) ──> 3标签预测
23个Barra三级描述变量 ──合成──> 10风格因子周收益 ─┘        │
                                              Composite(0~3) -> 风格强度(-1~1)
                                                      │
            指数风格暴露(成分股2年月度收益回归) × 强度 ──> 指数打分
                                                      │
                          贪心算法选8指数(w_d=0.5) ──> 流动性最优ETF ──> 权重∝Norm_score
                                                      │
                              周度回测(成本0.03%, 初始1亿, 基准中证全指)
```

## 目录结构

```
config/                 全部超参/指标代码/指数池 (yaml, 禁止硬编码)
  strategy.yaml         回测参数、窗口、贪心参数、复现目标
  macro_indicators.yaml 16个宏观指标 (EDB代码待确认回填)
  index_universe.yaml   宽基指数池 (static/dynamic 两模式)
  barra_factors.yaml    23描述变量 + 10合成因子公式
data/{raw,interim,processed}/   parquet 缓存 (raw 由 notebook 产生)
notebooks/01_choice_data_fetch.ipynb   Choice取数 (分批+缓存+重试)
docs/DATA_FIELDS.md     ⭐ Choice字段/EDB代码待确认清单
src/
  utils/    config / parquet io / 交易日历(周末调仓日、T+1)
  data/     loaders (只读缓存, 不碰API)
  factors/  preprocess(3MAD+标准化+中性化) / composite(合成) /
            factor_returns_quantile(Phase1) / factor_returns_barra(Phase2 WLS+行业约束) /
            descriptors(23描述变量, 财务类待字段确认)
  macro/    entropy (熵权法, 已实现+单测)
  prediction/ labels(3标签) + logistic(滚动104周, 防未来) (已实现+单测)
  strategy/ index_scoring / greedy(已实现+单测) / etf_mapping
  backtest/ engine(份额记账, 已实现+单测) / metrics / report
scripts/00~05           分步执行入口
tests/                  17个单测 (熵权/标签/贪心/回测会计/Logistic防未来) ✅ 全部通过
```

## 运行顺序

```bash
pip install -r requirements.txt        # EmQuantAPI 需从Choice官网单独安装
# 1) 确认 docs/DATA_FIELDS.md 清单, 回填 notebook 字段与 yaml EDB 代码
# 2) jupyter: notebooks/01_choice_data_fetch.ipynb  (填账号密码, 自上而下运行)
python scripts/00_check_data.py        # 数据完整性校验
python scripts/01_factor_returns.py    # 因子收益 (Phase1: 分位多空)
python scripts/02_macro_scores.py      # 熵权宏观得分 (自检图3)
python scripts/03_predict_styles.py    # Logistic预测 (自检表6: Trend≈70%)
python scripts/04_build_portfolio.py   # 贪心选指数 + ETF映射
python scripts/05_backtest.py          # 回测 (对账表7/图12)
pytest                                 # 单元测试
```

## 交付阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| 骨架 | 目录/配置/核心算法/单测/取数notebook/字段清单 | ✅ 本次交付 |
| Phase1 | 行情类描述变量 + 分位多空因子收益 + 全管线跑通 | 待字段确认后实现 |
| Phase1.5 | 财务类描述变量 (PIT对齐) | 同上 |
| Phase2 | 完整Barra截面WLS回归(行业市值加权约束) + 对账 | 框架已写(`factor_returns_barra.py`) |
| Phase3 | 回测起点前推至2018年前 (数据已按2014起拉取) | 配置已预留 |

## 复现对照（报告表7，2020-01-02 ~ 2026-04-17）

| 指标 | 目标值 |
|---|---|
| 年化收益率 | 19.05% |
| Sharpe | 1.0370 |
| Calmar | 1.0980 |
| 最大回撤 | -17.35% |
| 累计收益 | 286.85% |
| 总换手率 | 335.63% |

中间自检点：图1（因子年化收益标准差≈5%）、图2（volatility最优占比29.28%）、图3（5类宏观得分走势）、图4（规模因子AR1≈0.21）、表6（Trend准确率≈70%）。

## 防未来函数原则（全局强制）

- 每个调仓日 t 只用 ≤t 的数据：因子暴露用 t 时点截面；宏观熵权用过去52周；Logistic 训练集只含 (X_s, y_{s+1}), s+1≤t
- 财务数据按"截至 t 已公告"的 PIT 口径（见 DATA_FIELDS.md 表B 注意事项）
- 成分股权重用"距调仓日最近一个月末"快照
- 单测中显式校验：窗口不足处必须为 NaN、首条预测必须落在第106周之后

## 已知实现决策（与报告模糊处的取舍，对账时可调）

1. 贪心算法中欧氏距离 D_i 默认每轮 min-max 归一到 [0,1]（报告未说明，量纲对齐需要），`greedy_select(normalize_distance=False)` 可关闭
2. 行业口径用中信一级近似 CNE6 的32行业
3. ETF 执行价用复权单位净值，缺失时回退复权收盘价
4. Norm_score 全为0的极端情形（被选指数含截面最低分）ETF 退化为等权
