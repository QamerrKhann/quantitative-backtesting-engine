"""Command-line entry point: ``python -m quantbt.cli --csv prices.csv``."""

import argparse
import json

from quantbt.data import load_prices
from quantbt.engine import performance_metrics, run_backtest
from quantbt.strategies import momentum, sma_crossover


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a daily-bar strategy backtest")
    ap.add_argument("--csv", required=True, help="OHLCV CSV with a Date column")
    ap.add_argument("--strategy", choices=["sma", "momentum"], default="sma")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=50)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--capital", type=float, default=100000.0)
    ap.add_argument("--out", default="equity_curve.csv", help="where to write the daily results")
    ap.add_argument("--plot", action="store_true", help="also save equity_curve.png")
    a = ap.parse_args(argv)

    prices = load_prices(a.csv)["Price"]
    if a.strategy == "sma":
        signal = sma_crossover(prices, a.fast, a.slow)
    else:
        signal = momentum(prices, a.lookback)
    result = run_backtest(prices, signal, a.capital, a.commission_bps, a.slippage_bps)
    print(json.dumps(performance_metrics(result), indent=2))
    result.to_csv(a.out)
    print(f"Saved {a.out}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")  # works without a display
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4.5))
        result["equity"].plot(ax=ax, label=f"{a.strategy.upper()} strategy")
        result["benchmark_equity"].plot(ax=ax, label="Buy and hold", alpha=0.7)
        ax.set_ylabel("Portfolio value")
        ax.set_title(f"{a.strategy.upper()} strategy vs buy and hold")
        ax.legend()
        fig.tight_layout()
        png = a.out.rsplit(".", 1)[0] + ".png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"Saved {png}")


if __name__ == "__main__":
    main()
