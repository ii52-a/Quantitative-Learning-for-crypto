# 量化交易系统重构规划方案

## 一、现有架构分析

### 1.1 当前目录结构
```
Quantitative Learning/
├── Data/                    # 数据模块
│   ├── api.py              # API接口
│   ├── binance_client.py   # 币安客户端
│   ├── data_service.py     # 数据服务
│   ├── database.py         # 数据库操作
│   └── ...
├── Strategy/               # 策略模块
│   ├── CTA/               # CTA策略
│   ├── PositionContral/   # 仓位管理
│   └── live_trading.py    # 实盘交易
├── UI/                     # 用户界面
│   └── main_ui.py         # 主界面
├── app_logger/            # 日志模块
└── Config.py              # 配置文件
```

### 1.2 存在的问题
1. **模块职责不清晰**：Data模块和Strategy模块存在交叉依赖
2. **代码重复**：多个地方存在重复的数据获取逻辑
3. **缺乏抽象**：策略模板没有统一接口
4. **回测功能不完善**：缺少专业的回测报告和可视化
5. **UI不够友好**：缺乏引导和帮助说明

---

## 二、重构目标架构

### 2.1 新目录结构
```
Quantitative Learning/
├── core/                          # 核心模块
│   ├── __init__.py
│   ├── config.py                  # 统一配置
│   ├── exceptions.py              # 自定义异常
│   └── constants.py               # 常量定义
│
├── data/                          # 数据层
│   ├── __init__.py
│   ├── sources/                   # 数据源
│   │   ├── __init__.py
│   │   ├── base.py               # 数据源基类
│   │   ├── binance.py            # 币安数据源
│   │   └── database.py           # 数据库数据源
│   ├── storage/                   # 数据存储
│   │   ├── __init__.py
│   │   └── sqlite.py             # SQLite存储
│   └── service.py                 # 数据服务统一接口
│
├── strategy/                      # 策略层
│   ├── __init__.py
│   ├── base.py                    # 策略基类
│   ├── templates/                 # 策略模板
│   │   ├── __init__.py
│   │   ├── trend_following.py    # 趋势跟踪策略
│   │   ├── mean_reversion.py     # 均值回归策略
│   │   └── arbitrage.py          # 套利策略
│   ├── indicators/                # 技术指标
│   │   ├── __init__.py
│   │   ├── macd.py
│   │   ├── rsi.py
│   │   └── bollinger.py
│   └── position/                  # 仓位管理
│       ├── __init__.py
│       └── manager.py
│
├── backtest/                      # 回测模块
│   ├── __init__.py
│   ├── engine.py                  # 回测引擎
│   ├── report/                    # 回测报告
│   │   ├── __init__.py
│   │   ├── metrics.py            # 绩效指标
│   │   └── visualization.py      # 可视化
│   └── results.py                 # 结果存储
│
├── trading/                       # 交易模块
│   ├── __init__.py
│   ├── broker.py                  # 券商接口
│   ├── order.py                   # 订单管理
│   └── live.py                    # 实盘交易
│
├── ui/                            # 用户界面
│   ├── __init__.py
│   ├── main_window.py            # 主窗口
│   ├── panels/                    # 功能面板
│   │   ├── __init__.py
│   │   ├── backtest_panel.py     # 回测面板
│   │   ├── strategy_panel.py     # 策略面板
│   │   └── data_panel.py         # 数据面板
│   ├── widgets/                   # 自定义控件
│   │   ├── __init__.py
│   │   ├── chart_widget.py       # 图表控件
│   │   └── table_widget.py       # 表格控件
│   └── styles/                    # 样式
│       └── style.qss
│
├── utils/                         # 工具模块
│   ├── __init__.py
│   ├── logger.py                  # 日志工具
│   ├── validator.py               # 数据验证
│   └── helpers.py                 # 辅助函数
│
├── tests/                         # 测试模块
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_strategy.py
│   └── test_backtest.py
│
├── docs/                          # 文档
│   ├── user_guide.md             # 用户手册
│   ├── api_reference.md          # API文档
│   └── strategy_development.md   # 策略开发指南
│
├── main.py                        # 程序入口
└── requirements.txt               # 依赖清单
```

---

## 三、核心模块设计

### 3.1 策略基类设计
```python
class BaseStrategy(ABC):
    """策略基类"""
    
    @abstractmethod
    def initialize(self, context: StrategyContext) -> None:
        """初始化策略"""
        pass
    
    @abstractmethod
    def on_bar(self, bar: Bar) -> Signal | None:
        """K线事件处理"""
        pass
    
    @abstractmethod
    def on_tick(self, tick: Tick) -> Signal | None:
        """Tick事件处理"""
        pass
    
    def get_parameters(self) -> dict:
        """获取策略参数"""
        pass
    
    def set_parameters(self, params: dict) -> None:
        """设置策略参数"""
        pass
```

### 3.2 回测引擎设计
```python
class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.strategy: BaseStrategy
        self.data_source: DataSource
        self.position_manager: PositionManager
        
    def run(self) -> BacktestResult:
        """运行回测"""
        pass
    
    def generate_report(self) -> BacktestReport:
        """生成报告"""
        pass
```

### 3.3 绩效指标计算
```python
class PerformanceMetrics:
    """绩效指标"""
    
    @staticmethod
    def total_return(equity_curve: pd.Series) -> float:
        """总收益率"""
        pass
    
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """最大回撤"""
        pass
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """夏普比率"""
        pass
    
    @staticmethod
    def win_rate(trades: list[Trade]) -> float:
        """胜率"""
        pass
```

---

## 四、UI设计规范

### 4.1 回测面板设计
```
┌─────────────────────────────────────────────────────────────┐
│                      回测系统                                │
├─────────────────────────────────────────────────────────────┤
│  步骤1: 选择策略                    [帮助]                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ○ MACD趋势策略        适合趋势行情                    │   │
│  │ ○ 均值回归策略        适合震荡行情                    │   │
│  │ ○ 布林带策略          适合波动市场                    │   │
│  │ ○ 自定义策略          上传策略文件                    │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  步骤2: 配置参数                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 交易对: [BTCUSDT  ▼]    K线周期: [30min ▼]           │   │
│  │ 数据量: [1000    ] 条   初始资金: [10000   ] USDT    │   │
│  │                                                      │   │
│  │ 策略参数:                                             │   │
│  │ ┌─────────────────────────────────────────────────┐ │   │
│  │ │ HIST过滤: [0.5    ]  说明: MACD柱状图过滤阈值    │ │   │
│  │ │ 止损比例: [3.0    ]% 说明: 单笔最大亏损比例      │ │   │
│  │ │ 止盈比例: [6.0    ]% 说明: 单笔目标盈利比例      │ │   │
│  │ └─────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  步骤3: 运行回测                                             │
│  [开始回测]    [查看帮助]    [保存配置]                      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 回测结果展示
```
┌─────────────────────────────────────────────────────────────┐
│                      回测结果                                │
├─────────────────────────────────────────────────────────────┤
│  绩效指标                                                    │
│  ┌──────────────┬──────────────┬──────────────┬──────────┐ │
│  │ 总收益率     │ 最大回撤     │ 夏普比率     │ 胜率     │ │
│  │ +45.23%     │ -12.34%     │ 1.85        │ 62.5%   │ │
│  └──────────────┴──────────────┴──────────────┴──────────┘ │
├─────────────────────────────────────────────────────────────┤
│  净值曲线                                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    📈 净值曲线图                      │   │
│  │     ___/\___                                       │   │
│  │    /        \___/\__                               │   │
│  │   /                \                               │   │
│  │  /                  \___                           │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  交易记录  [导出CSV]  [详细分析]                             │
│  ┌──────┬────────┬────────┬────────┬────────┬────────┐    │
│  │ 序号 │ 时间   │ 方向   │ 价格   │ 数量   │ 盈亏   │    │
│  ├──────┼────────┼────────┼────────┼────────┼────────┤    │
│  │  1   │ 01-15  │ 开多   │ 45000  │ 0.1    │ -      │    │
│  │  2   │ 01-16  │ 平多   │ 46000  │ 0.1    │ +100   │    │
│  └──────┴────────┴────────┴────────┴────────┴────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、策略模板库

### 5.1 趋势跟踪策略
- MACD策略
- 均线策略
- 突破策略

### 5.2 均值回归策略
- 布林带策略
- RSI策略
- KDJ策略

### 5.3 套利策略
- 跨期套利
- 跨品种套利
- 统计套利

---

## 六、实施计划

### 第一阶段：架构重构（1-2周）
- [ ] 创建新的目录结构
- [ ] 迁移核心模块
- [ ] 统一配置管理
- [ ] 完善异常处理

### 第二阶段：策略模板库（1周）
- [ ] 设计策略基类
- [ ] 实现MACD策略模板
- [ ] 实现均值回归策略
- [ ] 实现布林带策略

### 第三阶段：回测模块升级（1-2周）
- [ ] 重构回测引擎
- [ ] 实现绩效指标计算
- [ ] 开发可视化图表
- [ ] 完善报告生成

### 第四阶段：UI优化（1周）
- [ ] 设计引导式界面
- [ ] 添加帮助说明
- [ ] 优化交互体验
- [ ] 美化界面样式

### 第五阶段：测试与文档（1周）
- [ ] 编写单元测试
- [ ] 编写集成测试
- [ ] 编写用户手册
- [ ] 编写API文档

---

## 七、代码规范

### 7.1 命名规范
- 类名：PascalCase（如 `BacktestEngine`）
- 函数名：snake_case（如 `calculate_sharpe_ratio`）
- 常量：UPPER_CASE（如 `MAX_POSITION_SIZE`）
- 私有方法：_leading_underscore（如 `_validate_params`）

### 7.2 文档规范
- 所有公共方法必须有docstring
- 使用Google风格的docstring
- 复杂逻辑必须添加注释

### 7.3 测试规范
- 测试覆盖率不低于80%
- 每个模块必须有对应的测试文件
- 使用pytest框架
