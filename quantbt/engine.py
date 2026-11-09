"""Vectorised daily-bar backtest engine and performance statistics."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def run_backtest(prices: pd.Series, target_exposure: pd.Series,
                 initial_capital: float = 100_000.0,
                 commission_bps: float = 1.0, slippage_bps: float = 2.0) -> pd.DataFrame:
    """Simulate a long-only strategy.

    ``target_exposure[t]`` is the desired exposure decided at the close of
    session t, in [0, 1]. It is shifted one session so it earns the return from
    close t to close t+1 -- the signal never sees the return it is paid on.
    Costs are charged on |change in exposure| times (commission + slippage).
    """
    if initial_capital <= 0 or commission_bps < 0 or slippage_bps < 0:
        raise ValueError("Capital must be positive and costs non-negative")
    prices = prices.astype(float).sort_index()
    signal = target_exposure.reindex(prices.index).fillna(0.0).clip(0.0, 1.0)
    asset_returns = prices.pct_change().fillna(0.0)
    exposure = signal.shift(1).fillna(0.0)
    turnover = exposure.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(exposure.iloc[0])
    cost = turnover * ((commission_bps + slippage_bps) / 10_000.0)
    strategy_returns = exposure * asset_returns - cost
    equity = initial_capital * (1.0 + strategy_returns).cumprod()
    benchmark = initial_capital * (1.0 + asset_returns).cumprod()
    return pd.DataFrame({"price": prices, "asset_return": asset_returns,
                         "signal": signal, "exposure": exposure, "turnover": turnover,
                         "cost": cost, "strategy_return": strategy_returns,
                         "equity": equity, "benchmark_equity": benchmark})


def _return_stats(r: pd.Series, equity: pd.Series, risk_free_rate: float) -> dict:
    # The first row is the starting point (return 0), so N rows span N-1 periods.
    periods = max(len(r) - 1, 1)
    years = periods / TRADING_DAYS
    growth = float(equity.iloc[-1] / equity.iloc[0])
    std = r.std(ddof=1) if len(r) > 1 else 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = r - daily_rf
    sharpe = float(excess.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    downside = np.minimum(excess.to_numpy(), 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS))
    sortino = float(excess.mean() * TRADING_DAYS / downside_dev) if downside_dev > 0 else 0.0
    max_dd = float((equity / equity.cummax() - 1).min())
    cagr = growth ** (1 / years) - 1
    return {"total_return": growth - 1, "cagr": float(cagr),
            "annualized_volatility": float(std * np.sqrt(TRADING_DAYS)),
            "sharpe_ratio": sharpe, "sortino_ratio": sortino, "max_drawdown": max_dd,
            "calmar_ratio": float(cagr / abs(max_dd)) if max_dd < 0 else 0.0}


def performance_metrics(results: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    """Strategy statistics plus a buy-and-hold benchmark on the same prices."""
    r = results["strategy_return"].dropna()
    if r.empty:
        raise ValueError("No returns available")
    stats = _return_stats(r, results["equity"].dropna(), risk_free_rate)
    stats["average_daily_turnover"] = float(results["turnover"].mean())
    stats["trades"] = int((results["turnover"] > 0).sum())
    stats["time_in_market"] = float((results["exposure"] > 0).mean())
    if "benchmark_equity" in results:
        bench = _return_stats(results["asset_return"], results["benchmark_equity"], risk_free_rate)
        stats["benchmark"] = {k: bench[k] for k in
                              ("total_return", "cagr", "sharpe_ratio", "max_drawdown")}
    return stats
