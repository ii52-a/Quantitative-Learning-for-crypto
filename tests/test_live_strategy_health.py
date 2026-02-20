from Trading.live_trader import LiveTrader, TradingConfig, TradingMode
from Strategy.templates import STRATEGY_REGISTRY
from Strategy.multi_indicator_strategy import MultiIndicatorStrategy, AdaptiveMultiIndicatorStrategy


def test_all_strategies_pass_live_runtime_health_check():
    trader = LiveTrader(TradingConfig(mode=TradingMode.TEST))

    strategy_classes = list(STRATEGY_REGISTRY.values()) + [MultiIndicatorStrategy, AdaptiveMultiIndicatorStrategy]

    for strategy_class in strategy_classes:
        strategy = strategy_class()
        trader.set_strategy(strategy)
        health = trader.check_strategy_runtime_health()
        assert health["ok"], f"{health['strategy']} health failed: {health['message']}"
