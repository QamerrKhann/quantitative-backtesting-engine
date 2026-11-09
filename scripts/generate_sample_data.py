from pathlib import Path
import numpy as np
import pandas as pd
rng = np.random.default_rng(7)
dates = pd.bdate_range("2018-01-01", periods=1500)
ret = rng.normal(0.00025, 0.012, len(dates))
close = 100 * np.exp(np.cumsum(ret))
open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.006, len(dates)))
low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.006, len(dates)))
volume = rng.integers(500000, 5000000, len(dates))
out = Path(__file__).resolve().parents[1] / "data" / "sample_prices.csv"
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"Date": dates, "Open": open_, "High": high, "Low": low,
              "Close": close, "Volume": volume}).to_csv(out, index=False)
print(f"Wrote {out}")
