"""Signal generators. Each returns target exposure in {0, 1} decided at the close."""

import pandas as pd


def sma_crossover(prices: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Long when the fast simple moving average is above the slow one."""
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("Require 0 < fast < slow")
    f = prices.rolling(fast, min_periods=fast).mean()
    s = prices.rolling(slow, min_periods=slow).mean()
    return (f > s).astype(float).where(s.notna(), 0.0)


def momentum(prices: pd.Series, lookback: int = 60) -> pd.Series:
    """Long when the trailing ``lookback``-session return is positive."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    change = prices / prices.shift(lookback) - 1
    return (change > 0).astype(float).where(change.notna(), 0.0)
