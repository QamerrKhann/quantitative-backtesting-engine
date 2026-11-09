# Quantitative Backtesting Engine (Python)

A vectorized backtester for long-only strategies on daily price bars. It includes SMA-crossover and momentum strategies, applies each signal one session after it is computed, charges transaction costs, and reports risk statistics against a buy-and-hold benchmark.

## Setup and run
Requires Python 3.10+ (tested on 3.13 with pandas 3.0).
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or: pip install -e ".[dev]"
python scripts/generate_sample_data.py
python -m quantbt.cli --csv data/sample_prices.csv --strategy sma --fast 20 --slow 50 --plot
python -m quantbt.cli --csv data/sample_prices.csv --strategy momentum --lookback 60
pytest -q
```
After `pip install .` the same tool is also available as the `quantbt` command.

Input CSV columns: `Date, Open, High, Low, Close, Volume`. If an `Adj Close` column exists, it is used as the price series.

The output is JSON metrics plus `equity_curve.csv`, and with `--plot`, `equity_curve.png` comparing the strategy to buy and hold.

## How it works
```
prices ──► strategy ──► target exposure (decided at close t)
                              │ shift(1)
                              ▼
         exposure held over (close t → close t+1) × asset return
                       − |Δexposure| × (commission + slippage) bps
                              │ cumprod
                              ▼
                     equity curve ──► metrics
```

- **No lookahead:** a signal computed from data up to close *t* only earns the return from *t* to *t+1*. A test proves this by changing every price from day 200 onward and checking that the signals, exposures and equity up to day 200 are unchanged.
- **Costs:** charged on turnover, meaning the absolute change in exposure. Entering or exiting a full position costs `commission + slippage` basis points.
- **Metrics:** total return, CAGR, annualized volatility, Sharpe, Sortino (computed with downside deviation), max drawdown, Calmar, trade count, time in market, and the same core statistics for buy and hold.

## Module layout
| File | Role |
|---|---|
| `quantbt/data.py` | Loads the CSV and validates the schema, duplicate dates and non-positive prices |
| `quantbt/strategies.py` | Strategies that turn prices into target exposure (0 or 1) |
| `quantbt/engine.py` | `run_backtest` (simulation) and `performance_metrics` |
| `quantbt/cli.py` | Command-line interface |
| `tests/` | 13 pytest tests |

## Tests
These include the lag test, the no-lookahead test, exact cost arithmetic, a known drawdown checked by hand (100 → 120 → 90 gives −25%), always-long matching buy and hold, clipping to long-only, CSV validation, and a full CLI run.

## Limitations (say these before an interviewer does)
- A single asset, with exposure between 0 and 1 (no shorting, no leverage)
- Fills at the close, with a fixed cost in basis points. There is no market-impact model and no volume limit.
- The sample data is a synthetic random walk. Results on it mean nothing about real markets.
- Testing one parameter set on the full history invites overfitting. The fix is walk-forward or out-of-sample testing.

## Interview prep
**What is lookahead bias and how do you prevent it?** It means using information that wasn't available when the trading decision was made. Here the signal is shifted by one session, and a test changes future prices to prove that past results don't move.

**Why vectorized instead of event-driven?** It is fast and simple for daily bars with a single asset. An event-driven design is needed for intraday fills, order types, or portfolios whose sizing depends on the current state.

**Sharpe vs Sortino?** Sharpe penalizes all volatility. Sortino only penalizes downside deviation, so it doesn't punish large gains.

**Why is CAGR computed over N−1 periods?** N price rows contain only N−1 returns. The first row is the starting point.

**What would you add next?** Walk-forward evaluation, a parameter sensitivity grid, volatility targeting, multiple assets with rebalancing, and bootstrapped confidence intervals for Sharpe.
