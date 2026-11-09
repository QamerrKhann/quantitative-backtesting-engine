import numpy as np
import pandas as pd
import pytest
from quantbt.engine import run_backtest, performance_metrics
from quantbt.strategies import sma_crossover, momentum
from quantbt.data import load_prices

def test_signal_lagged_one_session():
    ix = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([100, 110, 100, 120], index=ix)
    signal = pd.Series([1, 1, 0, 0], index=ix)
    out = run_backtest(prices, signal, commission_bps=0, slippage_bps=0)
    assert out.exposure.tolist() == [0, 1, 1, 0]
    assert out.strategy_return.iloc[1] == pytest.approx(0.10)

def test_costs_reduce_equity():
    ix = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([100, 100, 100, 100], index=ix)
    signal = pd.Series([1, 0, 1, 0], index=ix)
    free = run_backtest(prices, signal, commission_bps=0, slippage_bps=0)
    charged = run_backtest(prices, signal, commission_bps=10, slippage_bps=10)
    assert charged.equity.iloc[-1] < free.equity.iloc[-1]

def test_metrics_finite_and_drawdown_bounded():
    ix = pd.date_range("2024-01-01", periods=100)
    prices = pd.Series(np.linspace(100, 130, 100), index=ix)
    metrics = performance_metrics(run_backtest(prices, momentum(prices, 5)))
    flat = [v for v in metrics.values() if not isinstance(v, dict)] + list(metrics["benchmark"].values())
    assert np.isfinite(flat).all()
    assert -1 <= metrics["max_drawdown"] <= 0

def test_invalid_strategy_parameters():
    with pytest.raises(ValueError): sma_crossover(pd.Series(range(100)), 20, 10)
    with pytest.raises(ValueError): momentum(pd.Series(range(100)), 0)

def test_csv_schema_validation(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Date,Close\n2024-01-01,10\n")
    with pytest.raises(ValueError): load_prices(str(path))


def test_no_lookahead_bias():
    """Changing future prices must not change any earlier signal, exposure or equity."""
    rng = np.random.default_rng(0)
    ix = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=ix)
    shocked = prices.copy()
    shocked.iloc[200:] *= 3.0  # huge jump on day 200 onwards
    for strat in (lambda p: sma_crossover(p, 10, 30), lambda p: momentum(p, 20)):
        a = run_backtest(prices, strat(prices))
        b = run_backtest(shocked, strat(shocked))
        # Everything up to and including day 199 is identical...
        pd.testing.assert_frame_equal(a.iloc[:200], b.iloc[:200])
        # ...and the exposure on day 200 was decided before the jump was visible.
        assert a.exposure.iloc[200] == b.exposure.iloc[200]


def test_always_long_matches_buy_and_hold_without_costs():
    ix = pd.date_range("2024-01-01", periods=50)
    prices = pd.Series(np.linspace(100, 150, 50), index=ix)
    out = run_backtest(prices, pd.Series(1.0, index=ix), commission_bps=0, slippage_bps=0)
    # Decided at close 0, so it earns every return from close 0 to the end.
    expected = 100_000 * prices.iloc[-1] / prices.iloc[0]
    assert out.equity.iloc[-1] == pytest.approx(expected)


def test_cost_amount_is_exact():
    ix = pd.date_range("2024-01-01", periods=3)
    prices = pd.Series([100.0, 100.0, 100.0], index=ix)
    out = run_backtest(prices, pd.Series([1.0, 1.0, 1.0], index=ix),
                       commission_bps=5, slippage_bps=5)
    # Enter once (turnover 1.0) at 10 bps.
    assert out.equity.iloc[-1] == pytest.approx(100_000 * (1 - 0.001))
    assert performance_metrics(out)["trades"] == 1


def test_known_drawdown():
    ix = pd.date_range("2024-01-01", periods=4)
    prices = pd.Series([100.0, 120.0, 90.0, 100.0], index=ix)
    out = run_backtest(prices, pd.Series(1.0, index=ix), commission_bps=0, slippage_bps=0)
    m = performance_metrics(out)
    assert m["benchmark"]["max_drawdown"] == pytest.approx(90 / 120 - 1)


def test_signal_clipped_to_long_only():
    ix = pd.date_range("2024-01-01", periods=3)
    prices = pd.Series([100.0, 101.0, 102.0], index=ix)
    out = run_backtest(prices, pd.Series([2.0, -1.0, 0.5], index=ix))
    assert out.signal.tolist() == [1.0, 0.0, 0.5]


def test_load_prices_prefers_adjusted_close_and_sorts(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("Date,Open,High,Low,Close,Volume,Adj Close\n"
                    "2024-01-02,1,1,1,10,5,9\n2024-01-01,1,1,1,20,5,18\n")
    frame = load_prices(str(path))
    assert frame["Price"].tolist() == [18.0, 9.0]


def test_load_prices_rejects_duplicates_and_nonpositive(tmp_path):
    dup = tmp_path / "dup.csv"
    dup.write_text("Date,Open,High,Low,Close,Volume\n2024-01-01,1,1,1,10,5\n2024-01-01,1,1,1,11,5\n")
    with pytest.raises(ValueError): load_prices(str(dup))
    neg = tmp_path / "neg.csv"
    neg.write_text("Date,Open,High,Low,Close,Volume\n2024-01-01,1,1,1,-3,5\n")
    with pytest.raises(ValueError): load_prices(str(neg))


def test_cli_end_to_end(tmp_path, capsys):
    from quantbt.cli import main
    ix = pd.bdate_range("2020-01-01", periods=120)
    close = np.linspace(100, 140, 120)
    pd.DataFrame({"Date": ix, "Open": close, "High": close, "Low": close,
                  "Close": close, "Volume": 1000}).to_csv(tmp_path / "p.csv", index=False)
    out = tmp_path / "eq.csv"
    main(["--csv", str(tmp_path / "p.csv"), "--fast", "5", "--slow", "20",
          "--out", str(out), "--plot"])
    assert out.exists() and (tmp_path / "eq.png").exists()
    assert '"sharpe_ratio"' in capsys.readouterr().out
