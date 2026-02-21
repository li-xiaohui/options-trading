"""
QQQ Short Put Backtest using Polygon.io

Strategy:
- Sell cash-secured puts on QQQ
- Target fixed DTE (e.g. 30)
- Select strike closest to target delta
- Hold to expiration
- Daily resolution backtest

Data:
- Underlying prices: Polygon daily aggregates
- Option prices: Polygon option aggregates
- Greeks: Approximated via Black–Scholes

IMPORTANT:
- Polygon options data coverage varies by date
- This is research-grade, not execution-grade
"""

from polygon import RESTClient
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import timedelta
from tqdm import tqdm
import os
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIGURATION
# =========================

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
print(f"Using Polygon API Key: {POLYGON_API_KEY[:4]}...")

UNDERLYING = "QQQ"
START_DATE = "2025-11-01"
END_DATE = "2025-11-30"

TARGET_DTE = 30
TARGET_DELTA = -0.25

RISK_FREE_RATE = 0.04
DEFAULT_IV = 0.25

INITIAL_CAPITAL = 100_000

# =========================
# POLYGON CLIENT
# =========================

client = RESTClient(POLYGON_API_KEY)

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
# DATA FETCHING
# =========================


def get_underlying_prices(start, end):
    df = yf.download(UNDERLYING, start=start, end=end, progress=False)
    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index.name = "date"
    return df


def get_put_chain(expiry_date):
    contracts = client.list_options_contracts(
        underlying_ticker=UNDERLYING,
        contract_type="put",
        expiration_date=expiry_date,
        limit=1000,
    )

    rows = []
    for c in contracts:
        rows.append(
            {
                "ticker": c.ticker,
                "strike": c.strike_price,
                "expiry": pd.to_datetime(c.expiration_date),
            }
        )

    return pd.DataFrame(rows)


def get_option_close(option_ticker, date):
    aggs = client.get_aggs(
        ticker=option_ticker, multiplier=1, timespan="day", from_=date, to=date, limit=1
    )

    if len(aggs) == 0:
        return None

    return aggs[0].close


# =========================
# BACKTEST ENGINE
# =========================


def backtest():
    prices = get_underlying_prices(START_DATE, END_DATE)
    capital = INITIAL_CAPITAL
    trades = []

    dates = prices.index

    for entry_date in tqdm(dates[:-TARGET_DTE]):
        expiry_date = entry_date + timedelta(days=TARGET_DTE)

        if expiry_date not in prices.index:
            continue

        spot = prices.loc[entry_date, "close"]
        T = TARGET_DTE / 365

        chain = get_put_chain(expiry_date.date())
        if chain.empty:
            continue

        option_rows = []

        for _, row in chain.iterrows():
            strike = row["strike"]
            delta = put_delta(S=spot, K=strike, T=T, r=RISK_FREE_RATE, iv=DEFAULT_IV)

            price = get_option_close(row["ticker"], entry_date.date())
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

        # Enter trade
        capital -= cash_required

        exit_price = get_option_close(selected["ticker"], expiry_date.date())

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

    # Save results
    results.to_csv("qqq_short_put_results.csv", index=False)
    print("\nTrade log saved to qqq_short_put_results.csv")
