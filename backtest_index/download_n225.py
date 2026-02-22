"""
Download Nikkei 225 (^N225) daily data using yfinance.

Saves to backtest_index/data/n225_daily.csv.

Note: If you get YFTzMissingError or empty data, yfinance may be failing for
this symbol (known for some indices). Try again later, or export from
Yahoo Finance and save as data/n225_daily.csv with columns: date, close.
"""

import time
import pandas as pd
from pathlib import Path
import yfinance as yf
from setup import DATA_DIR, MAX_RETRIES

START_DATE = "1990-01-01"


def _download_chunk(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Download one chunk; returns None on failure."""
    try:
        df = yf.download(
            symbol,
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


def download_n225(end_date: str | None = None) -> pd.DataFrame:
    """Fetch ^N225 daily bars and save to data/n225_daily.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "n225_daily.csv"

    end = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(end)

    chunks = []
    # Download in 5-year chunks to reduce Yahoo API issues
    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end_ts = min(
            chunk_start + pd.DateOffset(years=5),
            end_ts + pd.Timedelta(days=1),
        )
        start_str = chunk_start.strftime("%Y-%m-%d")
        end_str = chunk_end_ts.strftime("%Y-%m-%d")
        for attempt in range(MAX_RETRIES):
            part = _download_chunk("^N225", start_str, end_str)
            if part is not None and not part.empty:
                chunks.append(part)
                break
            time.sleep(3)
        chunk_start = chunk_end_ts

    if not chunks:
        raise SystemExit(
            "No N225 data from yfinance (try again later or check network)."
        )

    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").drop_duplicates(subset=["date"])

    if len(df) < 100:
        raise SystemExit("Insufficient N225 data returned.")

    df.to_csv(path, index=False)
    print(
        f"Saved N225: {path} ({len(df)} rows, {df['date'].min()} to {df['date'].max()})"
    )
    return df


if __name__ == "__main__":
    download_n225()
