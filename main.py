"""
QQQ Short Put Backtest (local data only).

Strategy:
- Sell cash-secured puts on QQQ
- Target fixed DTE (e.g. 30)
- Select strike closest to target delta
- Hold to expiration
- Daily resolution backtest

Data: Load from local `data/` folder (no API calls).
- Run download_data.py first to fetch option chain and prices from Polygon.
- Greeks: Approximated via Black–Scholes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
from datetime import timedelta
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================

DATA_DIR = Path("data")
UNDERLYING = "QQQ"
START_DATE = "2025-11-01"
END_DATE = "2025-11-30"

TARGET_DTE = 30
TARGET_DELTA = -0.25

RISK_FREE_RATE = 0.04
DEFAULT_IV = 0.25

INITIAL_CAPITAL = 100_000

# =========================
# MATH HELPERS
# =========================


def put_delta(S, K, T, r, iv):
    """Black-Scholes put delta"""
    if T <= 0 or iv <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    return norm.cdf(d1) - 1


def select_strike_by_delta(df, target_delta):
    df = df.copy()
    df["delta_diff"] = abs(df["delta"] - target_delta)
    return df.sort_values("delta_diff").iloc[0]


# =========================
# LOCAL DATA LOADING
# =========================


def load_underlying_prices(start: str, end: str) -> pd.DataFrame:
    """Load underlying daily prices from data/underlying.csv (no API)."""
    path = DATA_DIR / "underlying.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run download_data.py first to fetch data."
        )
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df = df[["close"]]
    # Filter to requested range
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask]


def load_put_chain(expiry_date, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load put chain for expiry from data/put_chains/expiry_YYYY-MM-DD.csv."""
    if hasattr(expiry_date, "date"):
        expiry_date = expiry_date.date()
    date_str = pd.Timestamp(expiry_date).strftime("%Y-%m-%d")
    path = data_dir / "put_chains" / f"expiry_{date_str}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


def load_option_daily(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load option daily (ticker, date, close) from data/option_daily.csv."""
    path = data_dir / "option_daily.csv"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "date", "close"])
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def get_option_close_from_lookup(option_ticker, date, option_lookup: dict):
    """Look up option close from preloaded option_lookup[(ticker, date)] -> close."""
    key = (option_ticker, pd.Timestamp(date).date() if hasattr(date, "year") else date)
    return option_lookup.get(key)


# =========================
# BACKTEST ENGINE
# =========================


def backtest():
    prices = load_underlying_prices(START_DATE, END_DATE)
    if prices.empty:
        return pd.DataFrame()

    option_daily = load_option_daily()
    option_lookup = {}
    if not option_daily.empty:
        for row in option_daily.itertuples(index=False):
            option_lookup[(row.ticker, row.date)] = row.close

    capital = INITIAL_CAPITAL
    trades = []
    dates = prices.index

    for entry_date in tqdm(dates[:-TARGET_DTE]):
        expiry_date = entry_date + timedelta(days=TARGET_DTE)

        if expiry_date not in prices.index:
            continue

        spot = prices.loc[entry_date, "close"]
        T = TARGET_DTE / 365

        chain = load_put_chain(expiry_date)
        if chain.empty:
            continue

        option_rows = []

        for _, row in chain.iterrows():
            strike = row["strike"]
            delta = put_delta(S=spot, K=strike, T=T, r=RISK_FREE_RATE, iv=DEFAULT_IV)

            price = get_option_close_from_lookup(
                row["ticker"], entry_date.date(), option_lookup
            )
            if price is None or price <= 0:
                continue

            option_rows.append(
                {
                    "ticker": row["ticker"],
                    "strike": strike,
                    "delta": delta,
                    "price": price,
                }
            )

        if not option_rows:
            continue

        df_opts = pd.DataFrame(option_rows).dropna()
        if df_opts.empty:
            continue

        selected = select_strike_by_delta(df_opts, TARGET_DELTA)

        credit = selected["price"] * 100
        cash_required = selected["strike"] * 100

        if capital < cash_required:
            continue

        capital -= cash_required

        exit_price = get_option_close_from_lookup(
            selected["ticker"], expiry_date.date(), option_lookup
        )
        if exit_price is None:
            exit_price = 0.0

        pnl = credit - exit_price * 100
        capital += cash_required + pnl

        trades.append(
            {
                "entry_date": entry_date,
                "expiry_date": expiry_date,
                "spot_entry": spot,
                "strike": selected["strike"],
                "delta": selected["delta"],
                "credit": credit,
                "exit_price": exit_price,
                "pnl": pnl,
                "capital": capital,
            }
        )

    return pd.DataFrame(trades)


# =========================
# ANALYSIS
# =========================


def analyze(results):
    if results.empty:
        print("No trades executed.")
        return

    results["cum_pnl"] = results["pnl"].cumsum()

    total_pnl = results["pnl"].sum()
    win_rate = (results["pnl"] > 0).mean()
    max_dd = (results["cum_pnl"].cummax() - results["cum_pnl"]).max()

    print("\n===== BACKTEST RESULTS =====")
    print(f"Trades: {len(results)}")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Max drawdown: ${max_dd:,.2f}")
    print(f"Final capital: ${results.iloc[-1]['capital']:,.2f}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    results = backtest()
    analyze(results)

    if not results.empty:
        results.to_csv("qqq_short_put_results.csv", index=False)
        print("\nTrade log saved to qqq_short_put_results.csv")
