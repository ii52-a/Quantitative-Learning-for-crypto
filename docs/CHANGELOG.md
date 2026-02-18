# 更新说明 (CHANGELOG)

## [2.0.0] - 2026-02-18

### 重大更新

本次更新对整个项目进行了全面重构，提升了代码质量和用户体验。

### 新增功能

#### 核心模块 (core/)
- **config.py**: 统一配置管理，支持回测配置和交易配置
- **constants.py**: 常量定义，包括K线周期、订单类型、信号类型等
- **exceptions.py**: 自定义异常体系，便于错误处理

#### 策略模块 (strategy/)
- **base.py**: 策略基类，定义标准接口
  - `BaseStrategy`: 所有策略的基类
  - `Signal`: 交易信号数据结构
  - `Bar/Tick`: K线/Tick数据结构
  - `StrategyParameter`: 策略参数定义

- **indicators.py**: 技术指标库
  - `MACD`: MACD指标计算
  - `RSI`: RSI指标计算
  - `BollingerBands`: 布林带指标计算
  - `MovingAverage`: 移动平均线
  - `ATR`: 平均真实波幅

- **templates/**: 策略模板库
  - `MACDStrategy`: MACD趋势跟踪策略
  - `TrendFollowingStrategy`: 趋势跟踪策略
  - `MeanReversionStrategy`: 均值回归策略
  - `BollingerBandsStrategy`: 布林带策略

#### 回测模块 (backtest/)
- **engine.py**: 回测引擎
  - 支持多策略回测
  - 自动计算绩效指标
  - 支持手续费和滑点设置

- **metrics.py**: 绩效指标计算
  - 总收益率、年化收益率
  - 最大回撤、夏普比率、索提诺比率
  - 胜率、盈亏比、期望值

- **report.py**: 回测报告生成
  - 文本格式报告
  - 策略对比报告
  - 交易记录导出

#### 用户界面 (ui/)
- **user_friendly.py**: 用户友好界面
  - 引导式操作流程
  - 策略卡片选择
  - 参数说明提示
  - 帮助文档集成

### 改进

#### 数据模块
- 支持分批获取大数据量（最大50000条）
- 自动初始化数据库功能
- 代理支持（解决地区限制）
- K线周期别名支持（30m → 30min）

#### 回测功能
- 数据量范围扩大到50000条
- 支持自定义初始资金
- 完善的绩效指标计算
- 专业的回测报告

#### UI界面
- 引导式操作流程
- 策略卡片展示
- 参数悬停提示
- 帮助文档集成

### 修复

- 修复回测数据量固定850条的问题
- 修复实盘交易UI卡住的问题
- 修复K线周期"不支持的周期: 30m"错误
- 修复数据更新导致界面卡死的问题
- 修复API地区限制问题

### 文档

- 新增用户手册 (docs/user_guide.md)
- 新增重构规划方案 (docs/refactoring_plan.md)
- 新增更新说明 (docs/CHANGELOG.md)

---

## [1.0.0] - 初始版本

### 功能
- 基础回测功能
- MACD策略
- 实盘交易
- 基础UI界面
