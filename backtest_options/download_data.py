"""
Download option chain and price data from Polygon API for local backtesting.

Saves to the `data/` folder. Respects Polygon free-tier rate limit of 5 API calls per minute.
Run this once to populate data, then use main.py for backtesting (no API calls).
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from polygon import RESTClient
import pandas as pd
from tqdm import tqdm
from setup import START_DATE, END_DATE

load_dotenv()

# =========================
# CONFIGURATION
# =========================

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
if not POLYGON_API_KEY:
    raise SystemExit("Set POLYGON_API_KEY in .env")

UNDERLYING = "QQQ"
TARGET_DTE = 30

DATA_DIR = Path("data")
PUT_CHAINS_DIR = DATA_DIR / "put_chains"
RATE_LIMIT_CALLS_PER_MINUTE = 5

# =========================
# RATE LIMITER
# =========================


class RateLimiter:
    """Ensure we do not exceed RATE_LIMIT_CALLS_PER_MINUTE API calls."""

    def __init__(self, calls_per_minute: int = 5):
        self.calls_per_minute = calls_per_minute
        self.call_times: list[float] = []

    def wait_if_needed(self) -> None:
        now = time.monotonic()
        # Drop calls older than 1 minute
        self.call_times = [t for t in self.call_times if now - t < 60]
        if len(self.call_times) >= self.calls_per_minute:
            sleep_until = self.call_times[0] + 60
            sleep_s = max(0, sleep_until - now)
            if sleep_s > 0:
                time.sleep(sleep_s)
            self.call_times = self.call_times[1:]
        self.call_times.append(time.monotonic())

    def __call__(self, fn, *args, **kwargs):
        self.wait_if_needed()
        return fn(*args, **kwargs)


# =========================
# POLYGON CLIENT + WRAPPED CALLS
# =========================

client = RESTClient(POLYGON_API_KEY)
limiter = RateLimiter(calls_per_minute=RATE_LIMIT_CALLS_PER_MINUTE)


def _get_underlying_aggs():
    limiter.wait_if_needed()
    aggs = client.get_aggs(
        ticker=UNDERLYING,
        multiplier=1,
        timespan="day",
        from_=START_DATE,
        to=END_DATE,
        limit=50_000,
    )
    return list(aggs)


def _list_options_contracts(expiry_date, as_of: str | None = None):
    limiter.wait_if_needed()
    kwargs = {
        "underlying_ticker": UNDERLYING,
        "contract_type": "put",
        "expiration_date": expiry_date,
        "limit": 1000,
    }
    if as_of is not None:
        kwargs["as_of"] = as_of
    return list(client.list_options_contracts(**kwargs))


def _get_option_aggs(ticker: str, from_date: str, to_date: str):
    limiter.wait_if_needed()
    aggs = client.get_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=from_date,
        to=to_date,
        limit=50_000,
    )
    return list(aggs)


# =========================
# DOWNLOAD AND SAVE
# =========================


def download_underlying() -> pd.DataFrame:
    """Fetch underlying daily bars and save to data/underlying.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "underlying.csv"

    rows = _get_underlying_aggs()
    if not rows:
        raise SystemExit(f"No underlying data for {UNDERLYING} from {START_DATE} to {END_DATE}")

    df = pd.DataFrame(
        [{"date": pd.Timestamp(r.timestamp, unit="ms").date(), "close": r.close} for r in rows]
    )
    df = df.sort_values("date").drop_duplicates(subset=["date"])
    df.to_csv(path, index=False)
    print(f"Saved underlying: {path} ({len(df)} rows)")
    return df


def get_required_expiry_dates(underlying_df: pd.DataFrame) -> list[str]:
    """Expiry dates we need: entry_date + TARGET_DTE where expiry is also in the calendar."""
    dates = pd.to_datetime(underlying_df["date"])
    date_set = set(dates)
    expiries = set()
    for d in dates:
        exp = d + pd.Timedelta(days=TARGET_DTE)
        if exp in date_set:
            expiries.add(exp.strftime("%Y-%m-%d"))
    return sorted(expiries)


def download_put_chains(expiry_dates: list[str]) -> list[str]:
    """Fetch put chain for each expiry; save to data/put_chains/expiry_YYYY-MM-DD.csv. Returns all option tickers."""
    PUT_CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    all_tickers: list[str] = []

    for expiry in tqdm(expiry_dates, desc="Put chains"):
        contracts = _list_options_contracts(expiry, as_of=START_DATE)
        rows = [
            {
                "ticker": c.ticker,
                "strike": c.strike_price,
                "expiry": pd.to_datetime(c.expiration_date),
            }
            for c in contracts
        ]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        path = PUT_CHAINS_DIR / f"expiry_{expiry}.csv"
        df.to_csv(path, index=False)
        all_tickers.extend(df["ticker"].tolist())

    return list(dict.fromkeys(all_tickers))  # unique order preserved


def download_option_daily(tickers: list[str]) -> None:
    """Fetch daily aggregates for each option ticker and append to data/option_daily.csv."""
    path = DATA_DIR / "option_daily.csv"
    all_rows: list[dict] = []

    for ticker in tqdm(tickers, desc="Option daily"):
        aggs = _get_option_aggs(ticker, START_DATE, END_DATE)
        for r in aggs:
            dt = pd.Timestamp(r.timestamp, unit="ms")
            all_rows.append({"ticker": ticker, "date": dt.date(), "close": r.close})

    if not all_rows:
        print("No option daily data to save.")
        return
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"])
    df.to_csv(path, index=False)
    print(f"Saved option daily: {path} ({len(df)} rows)")


def main():
    print(f"Using Polygon API key: {POLYGON_API_KEY[:4]}...")
    print(f"Rate limit: {RATE_LIMIT_CALLS_PER_MINUTE} calls/minute")
    print(f"Data dir: {DATA_DIR.absolute()}\n")

    underlying_df = download_underlying()
    # underlying_df = pd.read_csv("data/underlying.csv")
    expiry_dates = get_required_expiry_dates(underlying_df)
    if not expiry_dates:
        print("No expiry dates in range; nothing else to download.")
        return

    tickers = download_put_chains(expiry_dates)
    if not tickers:
        print("No put chains; skipping option daily.")
        return

    download_option_daily(tickers)
    print("\nDone. Run main.py for backtesting (uses local data only).")


if __name__ == "__main__":
    main()
