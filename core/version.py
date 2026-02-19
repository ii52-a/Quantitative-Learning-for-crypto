"""版本更新记录模块"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import json
import os

VERSION = "2.0.0"
VERSION_FILE = os.path.join(os.path.dirname(__file__), "..", "version.json")


@dataclass
class UpdateItem:
    """单个更新项"""
    category: str
    description: str
    details: list[str] = field(default_factory=list)


@dataclass
class VersionInfo:
    """版本信息"""
    version: str
    release_date: str
    updates: list[UpdateItem]
    breaking_changes: list[str] = field(default_factory=list)
    deprecations: list[str] = field(default_factory=list)


VERSION_HISTORY: list[VersionInfo] = [
    VersionInfo(
        version="2.0.0",
        release_date="2026-02-19",
        updates=[
            UpdateItem(
                category="🆕 新功能",
                description="实盘交易模块",
                details=[
                    "支持测试网/主网切换",
                    "实时行情监控",
                    "自动策略执行",
                    "仓位管理和风险控制",
                    "交易日志和统计",
                ],
            ),
            UpdateItem(
                category="🆕 新功能",
                description="版本更新界面",
                details=[
                    "版本历史查看",
                    "更新内容详情",
                    "一键检查更新",
                ],
            ),
            UpdateItem(
                category="🔧 优化",
                description="参数优化系统",
                details=[
                    "新增复合参数优化（组合多种算法）",
                    "支持交易模式参数优化（做多/做空/双向）",
                    "参数依赖关系支持",
                    "并行参数评估优化",
                ],
            ),
            UpdateItem(
                category="🔧 优化",
                description="回测引擎",
                details=[
                    "修复盈亏比计算逻辑",
                    "新增仓位比例设置",
                    "导出数据包含回测参数",
                    "HTML报告新增MACD指标图表",
                ],
            ),
            UpdateItem(
                category="🐛 修复",
                description="Bug修复",
                details=[
                    "修复参数表格对齐问题",
                    "修复参数同步不完整问题",
                    "修复数据解析异常处理",
                    "修复优化停止功能",
                ],
            ),
        ],
        breaking_changes=[],
        deprecations=[],
    ),
    VersionInfo(
        version="1.5.0",
        release_date="2026-02-15",
        updates=[
            UpdateItem(
                category="🆕 新功能",
                description="参数探索系统",
                details=[
                    "随机搜索优化",
                    "网格搜索优化",
                    "遗传算法优化",
                    "模拟退火优化",
                    "粒子群优化",
                    "贝叶斯优化",
                    "强化学习优化",
                ],
            ),
            UpdateItem(
                category="🆕 新功能",
                description="策略系统",
                details=[
                    "MACD趋势策略",
                    "布林带策略",
                    "均线交叉策略",
                    "策略参数配置",
                ],
            ),
            UpdateItem(
                category="🆕 新功能",
                description="数据管理",
                details=[
                    "K线数据获取",
                    "数据库存储",
                    "数据缓存",
                ],
            ),
        ],
        breaking_changes=[],
        deprecations=[],
    ),
    VersionInfo(
        version="1.0.0",
        release_date="2026-02-01",
        updates=[
            UpdateItem(
                category="🎉 首发版本",
                description="核心功能",
                details=[
                    "回测引擎",
                    "策略框架",
                    "数据服务",
                    "可视化报告",
                    "主界面UI",
                ],
            ),
        ],
        breaking_changes=[],
        deprecations=[],
    ),
]


def get_current_version() -> str:
    """获取当前版本"""
    return VERSION


def get_version_history() -> list[VersionInfo]:
    """获取版本历史"""
    return VERSION_HISTORY


def get_latest_version_info() -> VersionInfo | None:
    """获取最新版本信息"""
    if VERSION_HISTORY:
        return VERSION_HISTORY[0]
    return None


def format_version_info(info: VersionInfo) -> str:
    """格式化版本信息为文本"""
    lines = [
        f"版本 {info.version} ({info.release_date})",
        "=" * 50,
        "",
    ]
    
    for update in info.updates:
        lines.append(f"{update.category}: {update.description}")
        for detail in update.details:
            lines.append(f"  • {detail}")
        lines.append("")
    
    if info.breaking_changes:
        lines.append("⚠️ 破坏性变更:")
        for change in info.breaking_changes:
            lines.append(f"  • {change}")
        lines.append("")
    
    if info.deprecations:
        lines.append("🗑️ 废弃功能:")
        for dep in info.deprecations:
            lines.append(f"  • {dep}")
        lines.append("")
    
    return "\n".join(lines)


def format_all_versions() -> str:
    """格式化所有版本信息"""
    lines = ["量化交易系统版本历史", "=" * 50, ""]
    
    for info in VERSION_HISTORY:
        lines.append(f"📌 版本 {info.version} ({info.release_date})")
        for update in info.updates:
            lines.append(f"  {update.category}: {update.description}")
        lines.append("")
    
    return "\n".join(lines)
