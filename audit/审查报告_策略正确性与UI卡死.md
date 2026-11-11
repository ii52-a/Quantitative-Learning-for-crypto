# 问题1 审查报告：策略正确性、边界处理与 UI 卡死

> 审查对象：当前工作区 `main` 分支（HEAD `7d46cce`）
> 审查方式：源码走读 + 运行竞态/仓位数学复现 + 日志与 SQLite 实物取证
> 复现脚本：`audit/repro_position_math.py`、`audit/repro_data_order.py`、`audit/repro_ui_freeze.py`
> 运行环境：`.venv` Python 3.12.10 / pandas 3.0.1 / PyQt5 5.15.11（缺 `tqdm`、`TA-Lib`，与能跑通回测的环境不一致）

---

## 0. 结论先行

| # | 结论 | 严重度 |
| :-- | :-- | :-- |
| 1 | `main` 的 UI **根本起不来**：`UI/Ui.py` 导入 `data.Api`（小写），而包里是 `Data/`；`Data/Api.py` 又导入顶层 `kline_process`。两条都会 `ModuleNotFoundError`。 | P0 |
| 2 | 回测结论不成立：746 笔交易、胜率 36.33%、毛利 **+49.95 USDT（+5.0%）**，但按 0.04%/边、5x、全仓计算手续费约 **2875 USDT ≈ 57 倍毛利**，扣费后约 **-282%**。费用参数定义了却从未被引用。 | P0 |
| 3 | 仓位层 4 个致命错误：反手变成 **2 倍多头**、空仓收到平仓信号会 **开多**、`SHORT` 永远开出 **多单**、部分平仓 **盈亏符号反向且名义价值反向增长**。 | P0 |
| 4 | 数据层：写库走 `./LocalData/`，读库走 `./Data/LocalData/`；`start_time` 查询返回**倒序**（MACD 在时间反序序列上计算）；`start+end` 查询无 `ORDER BY` 且忽略 `number`。 | P0 |
| 5 | UI 三类硬崩溃：槽函数未捕获异常 → Qt `abort`（`0xC0000409`）；`QThread` 局部变量被 GC → `QThread: Destroyed while thread is still running` → `abort`；GUI 线程里 `Client()` 默认 `ping()` + `20×3s` 重试 → 最长 60s 假死。 | P0 |
| 6 | `retry` 被 `@catch_and_log` 架空：异常在内层被吞成 `None`，重试逻辑一次都不会触发（实测 1 次 vs 3 次）。 | P1 |
| 7 | 回测框架只统计总资金与胜率，无权益曲线/回撤/夏普/盈亏比，无样本外与参数敏感性；数据侧全表重算、逐行 `iterrows` 写库。 | P1 |

---

## 1. 审查方法与证据

| 证据 | 内容 |
| :-- | :-- |
| 静态走读 | 全部 `.py`（约 20 个文件），逐行核对信号、仓位、资金、数据读写、线程与异常路径 |
| 运行复现 | `audit/repro_position_math.py` 复现 7 个仓位/边界缺陷；`audit/repro_data_order.py` 用只读连接复现排序、聚合对齐与聚合滞后；`audit/repro_ui_freeze.py` 用 offscreen Qt 复现 3 类崩溃并打印退出码 |
| 日志实物 | `Logs/Strategy.CTA.cta_macd_strategy/*.log`（746 笔 / 271 胜 / 36.33%）、`Logs/Strategy.PositionContral.Position/*.log`（3735 条逐笔记录，用于估算手续费） |
| 数据库实物 | `Data/LocalData/ETHUSDT_aggerate.db`（14 张聚合表，30min 48001 根，覆盖 2023-05-09→2026-02-02）；`LocalData/*_base.db` 写入侧 |

---

## 2. 策略与算法正确性

### 2.1 信号层（CTA-MACD）

实现本身是对的：`HIST` 由 0 下方上穿为金叉、上方下穿为死叉，与 `DIF` 上穿/下穿 `DEA` 等价（`talib.MACD` 的 `hist = macd - signal`，符号不受国内 2 倍差值约定影响）。问题在边界与合理性：

| 编号 | 问题 | 位置 | 说明 |
| :-- | :-- | :-- | :-- |
| S-1 | 只看 `HIST` 符号，无趋势/波动过滤 | [macd_base.py:24-49](Strategy/CTA/CTA_MACD_CORE/macd_base.py#L24-L49) | 零轴下方的金叉也做多，震荡市反复被打脸，这是 36% 胜率的直接来源；`CTA_MACD_FILTER` 目录是空的 0 字节文件 |
| S-2 | 最后一根 K 线永远不参与信号 | [macd_base.py:18-22](Strategy/CTA/CTA_MACD_CORE/macd_base.py#L18-L22) + [cta_macd_strategy.py:26](Strategy/CTA/cta_macd_strategy.py#L26) | 循环 `range(2, count)` 且用 `iloc[i-1]`，最大只取到 `iloc[count-2]`；`all_full()` 强平价却是 `iloc[-1]`，口径差一根 |
| S-3 | 同 bar 收盘价成交 | 同上 | 信号由该 bar 收盘推出、又按该收盘成交，等于无滑点、无下一根开盘；实盘不可行 |
| S-4 | warmup 硬编码 300 | [data_service.py:53](Strategy/data_service.py#L53) | 与 `ApiConfig.PADDING_COUNT=300 / GET_COUNT=350` 不一致；`additional_data=300` 又是第三处魔法数 |
| S-5 | 数据不足 300 根时静默产出空表 | [data_service.py:53-56](Strategy/data_service.py#L53-L56) | `iloc[300:]` 后为空 → `data_count==0` → 回测 0 笔 → 汇总行 `win/total*100` **ZeroDivisionError**（已复现） |
| S-6 | 指标依赖 TA-Lib | [data_service.py:5,49](Strategy/data_service.py#L49) | 该 venv 未安装 `TA-Lib`，换环境即 `ModuleNotFoundError`；应提供 talib/纯 pandas 双实现或锁依赖 |
| S-7 | 原地修改入参 DataFrame | [data_service.py:50-53](Strategy/data_service.py#L50-L53) | `macd_data()` 给调用方 df 追加 3 列并切片，隐式副作用 |
| S-8 | 参数写死 12/26/9 | 同上 | 无 `Params` 数据类，无法做参数敏感性/寻优（后续"更多策略"的前置条件） |
| S-9 | `get_data` 越界不校验 | [data_service.py:41](Strategy/data_service.py#L41) | `length+300` 传进 SQL `LIMIT`；`length<=-300` 时 SQLite 视为无限制 → 拉全表 |

### 2.2 仓位/资金层（最严重的一层）

这一层的缺陷会让**任何**策略的回测结果失真，已全部复现（`audit/repro_position_math.py`）。

| 编号 | 现象 | 证据（复现输出） | 根因 |
| :-- | :-- | :-- | :-- |
| P-01 | 反手（`LONG→SHORT`）后名义价值 **5000 → 10000**，方向仍是 +1 | `[2] nominal=10000 dir=+1` | [Position.py:179-189](Strategy/PositionContral/Position.py#L179-L189) 先算 `other = N + ΔN`（ΔN 与 N 同号）再全平再 `_open`，等于把原仓位翻倍；且从未取反 |
| P-02 | 空仓收到 `FULL`（平仓）信号 → **开出 1000 保证金多单** | `[3] margin=1000 nominal=5000` | [PonsitionContral.py:36-38](Strategy/PositionContral/PonsitionContral.py#L36-L38) 第一分支只判断 `avg_price==0`，不看信号类型 |
| P-03 | 空仓收到 `SHORT` → 开**多**；空头永远开不出来 | `[4] nominal=+5000` | 同上，`size*_usdt` 配 `PositionChange.OPEN`，方向由 [Position.py:60-61](Strategy/PositionContral/Position.py#L60-L61) 的 `nominal>=0` 推出，空仓时为 +1 |
| P-04 | 多头 +10% 部分平仓 500 → `d_pnl=-250`（应 +250），`nominal 7500`（应 2500） | `[5] d_pnl=-250 margin=500 nominal=7500` | [Position.py:114-131](Strategy/PositionContral/Position.py#L114-L131)：盈亏多乘了一个 `-`，且 `_update_context` 用 `+changed*L*direction` 把平仓当成加仓 |
| P-05 | `PositionSignal.PARTIAL` 根本走不到 `_close()` | 走读 | [PonsitionContral.py:31-49](Strategy/PositionContral/PonsitionContral.py#L31-L49) 没有 PARTIAL 分支，落到 `ERROR`；`PARTIAL` 是死代码 |
| P-06 | 无手续费/滑点/资金费/强平 | [Config.py:67-68](Config.py#L67-L68) | `OPEN_FEE_RADIO/CLOSE_FEE_RADIO` 全仓库**零引用**（grep 证实）；`all_usdt` 只加已实现盈亏，`true_margin()`/`open_count`/`close_count` 从未使用 |
| P-07 | 资金耗尽时开仓 **0/0 除零** | `[7] => ZeroDivisionError: float division by zero` | `_usdt==0` 时 `changed=0` → [Position.py:161-164](Strategy/PositionContral/Position.py#L161-L164) 的 `cur_persent = 0/0` |
| P-08 | 统计口径：`win` 取 `d_pnl>=0`，0 收益算胜；`total==0` 除零 | [Position.py:145](Strategy/PositionContral/Position.py#L145)、[cta_macd_strategy.py:39-40](Strategy/CTA/cta_macd_strategy.py#L39-L40) | 应区分 win/loss/breakeven，并对 0 笔做保护 |
| P-09 | `size=1.0` + `5x` = 全仓 5 倍名义敞口，无仓位预算/风险预算 | [macd_base.py:32,45](Strategy/CTA/CTA_MACD_CORE/macd_base.py#L32) | 单笔最大亏损 16.79%（保证金计），尾部风险完全未建模 |
| P-10 | 初始资金三处不一致：`StrategyMacd.usdt=1000`、`BackConfig.ORIGIN_USDT=100`、`BackTestSetting.origin_usdt` | [cta_macd_strategy.py:13](Strategy/CTA/cta_macd_strategy.py#L13) | 配置没有单一事实来源 |

#### 手续费量化（用真实成交日志反推）

- 每笔保证金中位数 963.5 USDT，5x → 名义 4817 USDT；
- 单笔往返手续费 `4817 × 0.04% × 2 ≈ 3.85 USDT`；
- 746 笔 ≈ **2875 USDT**；毛利仅 **+49.95 USDT** → 手续费是毛利的 **≈57 倍**；
- 换算成收益率：毛利 +5.0%，扣费后约 **-282%（相对 1000 本金）**。

> 也就是说：**当前回测的正收益完全来自"没有计费"这个假设**。同口径下每笔需要 ~0.4%（按保证金）的毛利才能打平，实际平均只有 ~0.007%。

#### 修复方向（建议签名）

把"方向"从 `int` 改为**带符号名义价值**，平仓量用"减少多少名义价值"表达，彻底消掉 P-01/P-04：

```python
class Position:
    nominal: float      # 带符号：>0 多，<0 空
    avg_price: float

    def apply(self, delta: float, price: float) -> float:
        """delta 为带符号名义价值增量：与持仓同号=开/加仓，反号=减仓/平仓；返回本次已实现盈亏。"""
        if self.nominal == 0:
            self.avg_price, self.nominal = price, delta
            return 0.0

        if self.nominal * delta > 0:                     # 加仓：更新均价
            a, b = abs(self.nominal), abs(delta)
            self.avg_price = (self.avg_price * a + price * b) / (a + b)
            self.nominal += delta
            return 0.0

        closed = min(abs(delta), abs(self.nominal)) * (1 if self.nominal > 0 else -1)
        pnl = (price / self.avg_price - 1.0) * closed    # 平掉的部分仍带原持仓符号
        self.nominal -= closed
        if abs(delta) > abs(closed):                     # 反手：剩余部分开反向
            self.avg_price = price
            self.nominal = delta + closed
        if self.nominal == 0:
            self.avg_price = 0.0
        return pnl
```

对照用例：多单 5000@100 平一半 @110 → `pnl=+250, nominal=2500`；空单 -5000@100 全平 @90 → `pnl=+500`；多单 5000@100 在 110 反手做空（`delta=-10000`）→ `pnl=+500, nominal=-5000`。保证金 = `abs(nominal)/leverage`，手续费 = `abs(delta)*fee_rate` 同层扣除。

信号侧改成"目标仓位"（见 §5.1），仓位层只做 `target - current`，就不会再出现"平仓信号变开仓""反手不翻转"。

### 2.3 回测框架

- 只输出总资金与胜率（[cta_macd_strategy.py:39-40](Strategy/CTA/cta_macd_strategy.py#L39-L40)），无权益曲线、最大回撤、夏普、盈亏比、期望值、持仓时长、逐笔明细导出。
- 无样本外/滚动窗口/参数敏感性；单标的、单参数组（12/26/9）、单周期（30min），无法判断过拟合。
- 执行模型：市价全成交、按 bar 收盘、无滑点、无部分成交、无资金费。
- 回测与实盘不共用代码路径：策略内部直接查库（`Dataservice`），没有 Broker/Clock 抽象，未来接实盘必然二次改写。
- `usdt`/杠杆/费率都不是通过 `BackTestSetting` 注入，UI 传参形同虚设。

### 2.4 数据层边界

| 编号 | 问题 | 位置 | 证据 |
| :-- | :-- | :-- | :-- |
| D-01 | 读写目录分裂：写 `./LocalData/`，读 `PROJECT_ROOT/Data/LocalData/` | [sqlite_oper.py:17](Data/sqlite_oper.py#L17)、[kline_process.py:13](Data/kline_process.py#L13) vs [sqlite_oper.py:273-281](Data/sqlite_oper.py#L273-L281) | 写侧只有 `*_base.db`（mtime 2026-02-19/20），读侧 `ETHUSDT_aggerate.db` 停在 **2026-02-02**，且写侧根本没有 `_aggerate.db`；`with_open_and_func` 里 `1m` 分支被第 281 行无条件覆盖成 `_aggerate.db`，是死代码 |
| D-02 | `start_time` 分支返回**倒序**；`start+end` 分支无 `ORDER BY` | [sqlite_oper.py:216-233](Data/sqlite_oper.py#L216-L233) | 实测前 5 行：`22:30, 22:00, 21:30…`，`monotonic_increasing=False` → MACD 在时间反序序列上计算，结果无意义 |
| D-03 | `start+end` 时 `number` 被忽略 | 同上 | 实测"要 5 行给了 7 行"，无上限 → 可一次拉全表（内存/卡顿） |
| D-04 | 时区混用 | [Api.py:136](Data/Api.py#L136) | `pd.to_datetime(ms, unit='ms')`（无 `utc=True`）得到 naive UTC，再被 `pd.to_datetime(x).timestamp()` 按本地时区解释 → 8 小时偏移、重复请求；初始化路径用的是 naive 本地时间，两条路径口径不同 |
| D-05 | 聚合对齐错位 | [kline_process.py:64](Data/kline_process.py#L64) | `origin='start_day'` 作用在北京时间索引上：`1d/3d/1W/1ME` 全部落在 16:00 UTC（北京 0 点），`6h` 落在 10:00/16:00/22:00 UTC；与 Binance 的 UTC 网格不一致（`4h/8h/12h` 恰好不受影响） |
| D-06 | 聚合不增量，且不丢弃"当前未完成桶" | [kline_process.py:64-90](Data/kline_process.py#L64-L90) | 实测：1min 最新 15:09，聚合库 30min 最新只到 14:30（缺 15:00 桶）→ 聚合结果滞后于源数据；反过来说，若在 15:09 执行聚合，`resample` 会写出 15:00 这个只含 9 分钟的半根 bar 并落库 → 回测读到不完整收盘价 |
| D-07 | 每次聚合都全表重算 + 全量重写 | [kline_process.py:27-30,83-90](Data/kline_process.py#L27-L30) | 一次性 `SELECT *` 拉 144 万行 × 14 个周期 resample，再逐行重写 109MB；分钟级 CPU + 内存峰值，是"接 UI 就卡死"的结构性原因 |
| D-08 | 空表/空 df 边界直接抛异常 | [sqlite_oper.py:83](Data/sqlite_oper.py#L83)、[sqlite_oper.py:112-117](Data/sqlite_oper.py#L112-L117) | `logger.exception(f"{path}…")` 里 `path` 未定义 → `NameError`；`read_time_span_str` 判空后又索引 `time_span[0]` → `TypeError` |
| D-09 | 写库静默丢数据 | [sqlite_oper.py:144-173](Data/sqlite_oper.py#L144-L173) | `iterrows` 逐行转换，任一行异常即 `except` 中断整个列表，但仍继续 `executemany` 并打印"写入成功"；`float(row['close_time'].value)` 遇到字符串列会中断 |
| D-10 | 错误日志本身抛异常 | [sqlite_oper.py:172](Data/sqlite_oper.py#L172)、[sqlite_oper.py:181](Data/sqlite_oper.py#L181) | `logger.Exception`（大写 E）不存在 → `AttributeError` 覆盖真实错误；空表时 `last_entry is None` 必然走到这里 |
| D-11 | 表不存在仍继续查询 | [sqlite_oper.py:246-249](Data/sqlite_oper.py#L246-L249) | 只 `logger.error` 不 `return` → 下一行 `OperationalError` → 装饰器吞掉 → 返回 `None` → 下游 `AttributeError` |
| D-12 | K 线键名不一致 | [Config.py:80-119](Config.py#L80-L119) | `data_time` 有 `'2h'`、`'1W'`、`'1ME'`，`K_LINE_TYPE` 只有 `'1w'` 没有 `'2h'`/`'1W'`；`K_LINE_TYPE['1m']` 映射的却是 1 个月。UI 一旦扩展到这些周期直接 `KeyError` |
| D-13 | 类级可变状态 + 共享 cursor | [kline_process.py:15-18](Data/kline_process.py#L15-L18) | `cursor`/`symbol`/`db_path` 挂在类上，多线程/多品种会串库；`with sqlite3.connect()` 结束后仍可能被外部使用 |

### 2.5 异常与工程细节

| 编号 | 问题 | 位置 | 说明 |
| :-- | :-- | :-- | :-- |
| E-01 | `retry` 被 `catch_and_log` **架空** | [Api.py:46-48](Data/Api.py#L46-L48) + [decorator.py:4-17](decorator.py#L4-L17) | 装饰器顺序是 `retry(catch_and_log(func))`，异常在内层被吞成 `None`，外层永远看不到异常。实测：包在一起的调用只执行 **1 次**，不包的执行 **3 次**。README 里"tenacity 自动重试"实际上是死代码 |
| E-02 | 吞异常导致下游 `None` 崩溃 | 同上 | `catch_and_log` 的 `return_default=None` 让 `get_data`/`read_range_kline` 返回 `None`，调用方 `macd_data(None)` 直接 `TypeError`，且堆栈指向无关位置 |
| E-03 | 日志方法名错误 | [sqlite_oper.py:172,181](Data/sqlite_oper.py#L172) | `logger.Exception`（大写 E）不存在 → `AttributeError` 覆盖真实异常；`logger.exception(f"{path}…")` 中 `path` 未定义 → `NameError` |
| E-04 | 导入期固定的可变默认值 | [type.py:23](type.py#L23) | `StaSetting.end_time=pd.Timestamp.now()` 在 import 时求值一次，之后所有实例共享同一个"现在" |
| E-05 | 包/命名/依赖 | 全仓库 | `Data/`、`Strategy/` 无 `__init__.py`（靠 PEP 420 命名空间包，脆弱）；`type.py` 与 `StrategyTypes.py` 语义冲突；`PonsitionContral` 拼写错误；`backtest/`、`core/` 只剩 `.pyc`；无 `requirements.txt`，venv 缺 `tqdm`/`TA-Lib` |
| E-06 | 双入口 + 废弃路径仍是默认 | [main.py](main.py) 与 [Ui.py:273-277](UI/Ui.py#L273) | 两个 `if __name__ == '__main__'`；CSV 相关代码已废（注释"csv废弃"），但 UI 默认仍走 CSV，且 `Api` 里已无对应方法 |

---

## 3. UI 卡死/崩溃（含复现）

### 3.1 复现矩阵

`audit/repro_ui_freeze.py` 一键跑全部用例：

```
--- case import: exit=1 (0x00000001)      ModuleNotFoundError: No module named 'data'
--- case layout: exit=0                   layout warnings: 2
--- case click:  exit=3221226505 (0xC0000409)  AttributeError: 'str' object has no attribute 'parent'
--- case thread: exit=3221226505 (0xC0000409)  QThread: Destroyed while thread is still running
--- case ping:   exit=0                   Client(ping=True) by default: True / worst case 20 x 3s = 60s
```

| 编号 | 触发 | 现象 | 根因 | 证据 |
| :-- | :-- | :-- | :-- | :-- |
| U-01 | 启动 | 窗口不出现 | `from data.Api import Api`（[Ui.py:13](UI/Ui.py#L13)）大小写不符；`Data/Api.py` 又 `from kline_process import`（[Api.py:23](Data/Api.py#L23)），缺顶层模块 | case import |
| U-02 | 点"开始回测"（默认勾选本地 csv） | 进程 **abort**（0xC0000409），看起来像闪退/卡死 | `check_local_path()` 收到 `str` 却用 `path.parent`（[Ui.py:188](UI/Ui.py#L188) + [urlibs.py:53](urlibs.py#L53)）；异常从 Qt 槽抛出 → PyQt5 调 `qFatal` → 进程终止；`button.setDisabled(True)` 之后再无恢复 | case click |
| U-03 | 取消勾选本地 csv（走 API 分支） | 窗口"未响应"数十秒 | `init_api()` 在 GUI 线程执行 `Api()`；`python-binance` 的 `Client.__init__` 默认 `ping=True` 发网络请求，外层 `@retry(20, 3s)` 最坏 60s（[Ui.py:33-41](UI/Ui.py#L33-L41)）。修好 U-02 之后，"本地 csv 为空"的分支也会走到这里 | case ping |
| U-04 | 本地/API 线程 | 永远"数据加载失败"，看不到错误原因 | `LoaclWorker` 调 `self.api.update_local_csv`、`ApiWorker` 调 `self.api.get_backtest_data`，两个方法**全仓库不存在**（[QtheaderWork.py:20,41](UI/QtheaderWork.py#L20)） | grep 证实 |
| U-05 | 本地线程运行中 | 进程 **abort** | `local_worker` 是局部变量，函数返回即被 GC → 正在运行的 `QThread` 被析构（[Ui.py:213-218](UI/Ui.py#L213-L218)） | case thread |
| U-06 | 打开窗口 | 结果区布局异常 + `QLayout` 警告 | `h_layout`/`h_layout1` 又用 `QHBoxLayout(self.results_pandel)` 构造函数挂到已有布局的 widget 上（[Ui.py:72,78](UI/Ui.py#L72)） | case layout |
| U-07 | 未来接上真实回测 | 点一次卡十几秒到几分钟 | `run_backtest()` 在槽里同步执行；CLI 全量 2 万根约 12s，聚合任务分钟级（[Ui.py:247-259](UI/Ui.py#L247-L259)） | 日志时间差 |
| U-08 | 任意失败 | 按钮永久灰掉，只能重启 | 只有 `run_backtest` 的 `finally` 会恢复按钮；`start()` 前半段的异常路径没有 `finally` | case click |
| U-09 | 成功路径 | "回测完成"，但**没有任何回测** | `run_backtest` 只构造 `BackTestSetting` 就打印完成，`data` 参数未使用（[Ui.py:247-259](UI/Ui.py#L247-L259)） | 走读 |
| U-10 | 日志 | 控制台被 DEBUG 刷屏 | 模块级 `logging.basicConfig(level=logging.DEBUG)`（[Ui.py:18](UI/Ui.py#L18)） | 走读 |

### 3.2 修复范式（可直接套用）

1. **槽函数一律有兜底**，避免异常逃逸到 Qt 事件循环（Qt 会直接 `qFatal` 终止进程）：
   ```python
   def guard(fn):
       @wraps(fn)
       def wrapper(self, *a, **k):
           try:
               return fn(self, *a, **k)
           except Exception:
               logger.exception("slot failed")
               QMessageBox.critical(self, "错误", traceback.format_exc())
               self.button.setEnabled(True)      # 任何失败都要恢复按钮
       return wrapper

   @guard
   def start(self): ...
   ```
   另外在 `main.py` 里 `sys.excepthook = _log_and_show`，兜住非槽路径的异常。
2. **所有 I/O / 回测 / 聚合进 Worker**，信号只传轻量对象（`progress(int, str)`、`log(str)`、`finished(dict)`、`error(str)`）；主线程只更新控件。
3. **线程要有归属**：`self._workers.add(worker)`，`worker.finished.connect(lambda: self._workers.discard(worker))` + `worker.wait()`；关闭窗口时统一 `_shutdown_thread()`。
4. **网络初始化不要在主线程**：`Client(..., ping=False)`，首次请求放在 Worker 里；重试用指数退避并且可取消。
5. **进度节流**：回测内每 N 根 bar 发一次 `progress`（或 100ms 定时器聚合），不要每根都 `pct.setValue`。
6. **按钮状态机**：`idle → running → idle`，`finally` 恢复；提供"取消"（`worker.requestInterruption()` + 策略内检查 `isInterruptionRequested()`）。

### 3.3 跨分支提示

`origin/codex/ui` 等分支已有 `UI/main_ui.py`、`live_trading`、`Data/binance_client` 等完整实现（日志里出现的模块名都来自那里）。**当前 `main` 的 `UI/Ui.py` 是对着那套目录写的**，所以"有时能跑、有时卡死"很可能取决于当时在哪个分支/哪个工作目录。建议二选一：把 UI 修到与 `Data/` 一致，或把 `Data/api.py` 那套布局合回 `main`，不要长期并存。上面 U-02/U-03/U-05 三类模式在任何分支都值得复查。

---

## 4. 优化方向

### 4.1 架构（先做这个，后面所有事都省力）

```
Config.DataPaths           单一绝对路径来源（PROJECT_ROOT 派生），消除 D-01
Data.Repository            只负责 K 线读写：UTC ms 存储、升序保证、增量更新、去重、缺口检测
Indicator (纯函数)         macd(df, 12, 26, 9) -> df，不修改入参，talib/pandas 双实现
Strategy (信号)            on_bar(ctx) -> Signal(target ∈ [-1,1], reason)
RiskManager                仓位预算、波动率目标、最大回撤/单笔风险
Broker (回测/实盘同接口)   submit(order) -> fill；费用/滑点/资金费/强平都在这层
Metrics/Report             权益曲线、MDD、Sharpe、盈亏比、逐笔明细
UI (Qt)                    只订阅信号、只渲染
```

关键点是 **Strategy 只表达目标仓位、Broker 才动钱**，这样 P-02/P-03/P-05 从设计上不可能再出现。

### 4.2 数据与性能

- 内部统一 **UTC ms**，只在展示层转 `Asia/Shanghai`（README 已有这条原则，代码未贯彻）。
- `resample(..., origin='epoch')` 或显式 `label`/`closed`，让日线/周线/月线与交易所对齐（修 D-05）；聚合时**丢掉当前未完成的桶**（修 D-06）。
- 聚合改**增量**：只重算"最新 1min 时间戳之后"的桶，写库用 `executemany`（不要 `iterrows`），并加 `PRAGMA journal_mode=WAL`、`synchronous=NORMAL`、`timestamp` 索引。
- 大数据量可考虑 `DuckDB/Parquet` 读侧（列存 + 谓词下推），SQLite 继续做落地。
- 回测循环向量化：MACD 交叉点用 `np.sign(hist).diff()!=0` 一次取出所有信号点，再事件驱动撮合，避免逐 bar `iloc`。

### 4.3 评估体系（决定"策略是否值得上线"）

- 必备指标：总收益、年化、最大回撤、Calmar、Sharpe/Sortino、盈亏比、期望值、平均持仓时长、交易次数、手续费占比。
- 必须做：样本内/样本外切分（如 7:3）、滚动 walk-forward、参数敏感性热图（快线×慢线×信号线）、手续费/滑点敏感性（0.02%/0.04%/0.08% 三档）。
- 基准：买入持有 ETH、以及随机入场的同交易次数对照。
- 上线门槛示例：扣费后年化 > 0、MDD < 20%、样本外不掉一半、单笔风险可控。

### 4.4 工程

- `requirements.txt` / `pyproject.toml` 锁依赖（当前 venv 缺 `tqdm`、`TA-Lib`，代码直接 import）。
- `pytest`：仓位数学用参数化用例（多/空/加仓/部分平/全平/反手/零资金），指标用对拍（talib vs 纯 pandas）。
- CI：`ruff` + `mypy` + 单测；把 `audit/repro_*.py` 收敛成正式测试。
- 命名/结构：`PonsitionContral` → `PositionControl`、`StrategyTypes.py` → `types.py`（避免与根目录 `type.py` 混淆）、补 `Data/__init__.py`、删除只剩 `.pyc` 的 `backtest/`、`core/`。
- 异常策略：`catch_and_log` 只用于"可降级"的地方，禁止包在 `@retry` 内层（修 E-01）；日志方法名统一小写。

---

## 5. 更多策略引入

### 5.1 先定接口，再谈策略

```python
@dataclass(frozen=True)
class Signal:
    ts: pd.Timestamp
    target: float            # -1 空, 0 平, +1 多；可小数表示半仓
    reason: str = ""

class Strategy(Protocol):
    params: dict
    def warmup(self) -> int: ...
    def on_bar(self, ctx: "BarContext") -> Signal: ...   # 只读已收盘 bar
```

配合 §4.1 的 `Broker`，新增策略只需要实现 `on_bar`，不再碰仓位/资金。

### 5.2 候选策略清单（按落地成本排序）

| 优先级 | 策略 | 逻辑 | 默认参数 | 适用市况 | 备注 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | **MACD + 趋势过滤**（补齐空的 `CTA_MACD_FILTER`） | 金叉且 `close > EMA200` 且 `ADX > 20` 才做多 | 12/26/9、EMA200、ADX14>20 | 趋势 | 改动最小，直接验证 S-1 能否把胜率/期望拉正 |
| 2 | EMA 双均线 + ATR 过滤 | `EMA20` 上穿 `EMA60` 做多，反向平；`ATR/close` 过大跳过 | 20/60、ATR14 | 趋势 | 与 MACD 相关性高，可做组合分散 |
| 3 | Donchian 突破 + ATR 跟踪止损（海龟） | 突破 20 日高做多，跌破 10 日低平；`2*ATR` 止损 | 20/10、ATR14×2 | 单边 | 需要 intrabar 高低点，考验回测撮合层 |
| 4 | 布林带均值回归 | 触下轨且 `RSI<30` 做多，回中轨平 | 20,2 / RSI14 | 震荡 | 与趋势策略负相关，适合组合 |
| 5 | RSI 反转 | `RSI<30` 分批做多，`RSI>55` 平 | 14 | 震荡 | 简单，适合验证仓位分批（配合 P-04 修复） |
| 6 | 资金费率/基差 carry | 永续资金费极端时反向 + 现货对冲 | 资金费阈值 | 中性 | 需要现货腿与资金费数据源，工程量中等 |
| 7 | 订单流不平衡（OFI） | 主动买卖量差 / 深度失衡 | 窗口 N 秒 | 高频/日内 | 仓库已有 `codex/add-order-flow-for-hedging` 分支可复用 |
| 8 | 多因子组合 | 趋势+动量+波动+资金费打分加权 | 权重搜索 | 全 | `codex` 分支有 composite optimizer，可移植 |

### 5.3 引入节奏

1. **先修 P0**（仓位数学 + 费用 + 排序 + 目录），否则新策略的回测数字都不可信。
2. 再上 **策略接口 + 指标纯函数 + 向量化回测**，用策略 1/2 做基准。
3. 然后做 **参数寻优 + walk-forward**，给策略 1/3/4 各出一张热图。
4. 最后做 **组合层**（多品种、波动率目标、相关性约束）与实盘同构的 Broker。

---

## 6. 建议的修复顺序（可直接当 TODO）

| 顺序 | 事项 | 涉及文件 | 验收标准 |
| :-- | :-- | :-- | :-- |
| 1 | 修仓位数学（P-01/02/03/04/07）+ 单测 | `Position.py`、`PonsitionContral.py` | `audit/repro_position_math.py` 全部符合"expect" |
| 2 | 接入费用/滑点（P-06） | `Position.py`、`Config.py` | 746 笔回测扣费后收益为负，且逐笔费用可查 |
| 3 | 统一数据目录 + 排序 + UTC（D-01/02/04） | `Config.py`、`sqlite_oper.py`、`Api.py` | `repro_data_order.py` 的 `monotonic_increasing=True`，读写同一路径 |
| 4 | UI: 修导入 + 槽兜底 + Worker 归属 + 线程内初始化 | `UI/Ui.py`、`UI/QtheaderWork.py` | `repro_ui_freeze.py` 五个 case 退出码全为 0，界面不 abort |
| 5 | 回测真正接 UI（Worker 化 + 进度 + 取消） | `UI/*`、`Strategy/*` | 点击可跑、可取消、按钮状态正确 |
| 6 | 指标纯函数 + 向量化 + 评估指标 | 新 `indicators/`、`backtest/` | 单测覆盖 + 报告含 MDD/Sharpe |
| 7 | 策略 1/2 接入 + walk-forward | `Strategy/` | 有样本外结果与热图 |

---

## 附录 A：复现命令

```bash
# 仓位/资金数学（7 个用例，纯计算，不写库）
.venv\Scripts\python.exe audit\repro_position_math.py

# 数据层（只读连接）：路径分裂 / 倒序 / 聚合对齐 / 聚合滞后
.venv\Scripts\python.exe audit\repro_data_order.py

# UI（offscreen，无需显示器）：导入失败 / 布局警告 / 槽异常 abort / QThread GC abort / ping 阻塞
.venv\Scripts\python.exe audit\repro_ui_freeze.py
```

## 附录 B：关键证据

- 逐笔日志：`Logs/Strategy.PositionContral.Position/Strategy.PositionContral.Position.log`（3735 条 = 5 次 746 笔回测），单次汇总见 `Logs/Strategy.CTA.cta_macd_strategy/Strategy.CTA.cta_macd_strategy.log`：`1049.94851301963(104.99%) / total:746 win:271 per:36.33%`。
- 聚合库：`Data/LocalData/ETHUSDT_aggerate.db`，30min 48001 根，`2023-05-09 14:30Z → 2026-02-02 14:30Z`；同目录 1min 库最新 `15:09Z` → 聚合库比源数据少最后一根（14:30 之后没有 15:00 桶），说明聚合不是增量的。
- 写侧目录 `LocalData/` 只有 7 个 `*_base.db`（mtime 2026-02-19/20），**没有任何 `_aggerate.db`** → 聚合结果从未落到策略真正读取的路径。
- 退出码：`0xC0000409`（STATUS_STACK_BUFFER_OVERRUN，Windows 上 `abort()` 的映射）= Qt 对"槽内未捕获异常"和"QThread 运行中被析构"的处理结果。
