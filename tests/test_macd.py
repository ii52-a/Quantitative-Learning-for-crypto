import numpy as np
import pandas as pd
import pytest

from Strategy.data_service import Dataservice


def test_ema_seed_and_recursion():
    values = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)
    ema = Dataservice._ema(values, 3)
    expected = [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0]
    np.testing.assert_allclose(ema.to_numpy(), expected, equal_nan=True)


def test_macd_hist_is_dif_minus_dea():
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 500)))
    macd, signal, hist = Dataservice.macd(close)
    np.testing.assert_allclose(hist.to_numpy(), (macd - signal).to_numpy(), equal_nan=True)


def test_macd_first_valid_bar():
    rng = np.random.default_rng(11)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    _macd, _signal, hist = Dataservice.macd(close)
    # EMA(26) 从第 25 根有效，MACD_SIGNAL 为 MACD 的 EMA(9)，故从第 33 根开始有值
    assert close.index.get_loc(hist.first_valid_index()) == 26 - 1 + 9 - 1
    assert hist.notna().sum() == len(close) - 33


def test_macd_data_no_nan_left():
    index = pd.date_range("2024-01-01", periods=120, freq="30min", tz="UTC")
    close = pd.Series(np.linspace(100, 140, 120), index=index)
    data = pd.DataFrame({"close": close}, index=index)
    out = Dataservice.macd_data(data)
    assert out[["MACD", "MACD_SIGNAL", "MACD_HIST"]].isna().sum().sum() == 0
    assert "MACD" not in data.columns  # 不得修改入参


def test_macd_uptrend_positive_hist():
    index = pd.date_range("2024-01-01", periods=120, freq="30min", tz="UTC")
    # 先横盘再加速上涨，MACD 柱应为正
    close = pd.Series(np.concatenate([np.full(60, 100.0), 100 * np.exp(np.linspace(0, 0.6, 60))]),
                      index=index)
    _macd, _signal, hist = Dataservice.macd(close)
    assert hist.dropna().iloc[-1] > 0


def test_get_data_rejects_bad_params():
    with pytest.raises(ValueError):
        Dataservice.get_data("ETHUSDT", "7min", 10)
    with pytest.raises(ValueError):
        Dataservice.get_data("ETHUSDT", "30min", 0)
