import pandas as pd
REQUIRED = {"Date", "Open", "High", "Low", "Close", "Volume"}
def load_prices(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"])
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.sort_values("Date").set_index("Date")
    if frame.index.has_duplicates:
        raise ValueError("Date values must be unique")
    col = "Adj Close" if "Adj Close" in frame.columns else "Close"
    frame["Price"] = pd.to_numeric(frame[col], errors="coerce")
    if frame["Price"].isna().any() or (frame["Price"] <= 0).any():
        raise ValueError("Prices must be finite and positive")
    return frame
