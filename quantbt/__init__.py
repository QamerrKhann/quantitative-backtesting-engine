"""Daily-bar quantitative research backtester."""

from quantbt.engine import performance_metrics, run_backtest
from quantbt.strategies import momentum, sma_crossover

__all__ = ["run_backtest", "performance_metrics", "sma_crossover", "momentum"]
__version__ = "1.0.0"
