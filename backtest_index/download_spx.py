"""
Download SPX (S&P 500) daily data since 1950 using yfinance.

Saves to backtest_index/data/spx_daily.csv. Polygon does not provide index
history before 2023, so we use Yahoo Finance (^GSPC) for long history.
"""

import time
import pandas as pd
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Install yfinance: pip install yfinance")

DATA_DIR = Path(__file__).resolve().parent / "data"
start_y=1950

MAX_RETRIES = 3


def _download_chunk(start: str, end: str) -> pd.DataFrame | None:
    """Download one chunk; returns None on failure."""
    try:
        df = yf.download(
            "^GSPC",
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            timeout=30,
            threads=False,
        )
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0).str.lower()
    df = df.reset_index()
    df = df.rename(columns={"Date": "date", "Close": "close"})
    return df[["date", "close"]]


def download_spx(end_date: str | None = None) -> pd.DataFrame:
    """Fetch SPX daily bars from 1950 and save to data/spx_daily.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "spx_daily.csv"

    end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
    chunks = []
    # Download in 5-year chunks to reduce Yahoo API issues
    end_ts = pd.Timestamp(end)
    while start_y < end_ts.year:
        chunk_end_y = min(start_y + 5, end_ts.year + 1)
        start_str = f"{start_y}-01-01"
        end_str = f"{chunk_end_y}-01-01"
        for attempt in range(MAX_RETRIES):
            part = _download_chunk(start_str, end_str)
            if part is not None and not part.empty:
                chunks.append(part)
                break
            time.sleep(3)
        start_y = chunk_end_y

    if not chunks:
        raise SystemExit(
            "No SPX data from yfinance (try again later or check network)."
        )

    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").drop_duplicates(subset=["date"])

    if len(df) < 100:
        raise SystemExit("Insufficient SPX data returned.")

    df.to_csv(path, index=False)
    print(f"Saved SPX: {path} ({len(df)} rows, {df['date'].min()} to {df['date'].max()})")
    return df


if __name__ == "__main__":
    download_spx()
