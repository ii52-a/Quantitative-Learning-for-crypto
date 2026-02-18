"""
项目目录结构说明

Quantitative Learning/
├── core/                   # 核心模块
│   ├── config.py          # 配置管理
│   ├── constants.py       # 常量定义
│   └── exceptions.py      # 异常定义
│
├── Data/                   # 数据模块
│   ├── data_service.py    # 统一数据服务
│   ├── database.py        # 数据库操作
│   ├── binance_client.py  # Binance客户端
│   └── kline_aggregator.py # K线聚合器
│
├── Strategy/               # 策略模块
│   ├── base.py            # 策略基类
│   ├── indicators.py      # 技术指标
│   ├── templates/         # 策略模板
│   ├── multi_indicator_strategy.py  # 多指标策略
│   └── parameter_optimizer.py       # 参数优化器
│
├── backtest/               # 回测模块
│   ├── engine.py          # 回测引擎
│   ├── metrics.py         # 绩效指标
│   ├── report.py          # 报告生成
│   └── visualization.py   # 可视化
│
├── UI/                     # 用户界面
│   └── main_ui.py         # 主界面
│
├── app_logger/             # 日志模块
│   └── logger_setup.py    # 日志配置
│
├── tests/                  # 测试模块
│   ├── test_config.py
│   ├── test_engine.py
│   └── ...
│
├── docs/                   # 文档
│   ├── CHANGELOG.md
│   └── user_guide.md
│
├── output/                 # 输出目录（自动创建）
│   └── reports/           # 回测报告
│
├── main.py                 # 程序入口
├── run_live.py            # 实盘运行入口
└── README.md              # 项目说明
"""

# 目录常量
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CORE_DIR = PROJECT_ROOT / "core"
DATA_DIR = PROJECT_ROOT / "Data"
STRATEGY_DIR = PROJECT_ROOT / "Strategy"
BACKTEST_DIR = PROJECT_ROOT / "backtest"
UI_DIR = PROJECT_ROOT / "UI"
LOGGER_DIR = PROJECT_ROOT / "app_logger"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"


def ensure_directories():
    """确保必要的目录存在"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)


def get_report_path(filename: str) -> Path:
    """获取报告文件路径"""
    ensure_directories()
    return REPORTS_DIR / filename
