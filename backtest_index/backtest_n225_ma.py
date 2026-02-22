"""
Nikkei 225 (^N225) moving-average crossover backtest.

- Fast MA: 5, 10, 20, 35, 50, 100 days
- Slow MA: 10, 20, 35, 50, 100 days
- For each (fast, slow) pair with fast < slow: compute annualized Sharpe
  using daily returns only on days when fast MA > slow MA.
- Also prints buy-and-hold annualized Sharpe for the full period.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from setup import DATA_DIR, RESULT_DIR, FAST_MA, SLOW_MA, TRADING_DAYS_PER_YEAR

N225_PATH = DATA_DIR / "n225_daily.csv"


def load_n225() -> pd.DataFrame:
    """Load N225 daily (date, close)."""
    if not N225_PATH.exists():
        raise FileNotFoundError(
            f"Missing {N225_PATH}. Run: python -m backtest_index.download_n225"
        )
    df = pd.read_csv(N225_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)


def add_mas(df: pd.DataFrame) -> pd.DataFrame:
    """Add fast and slow MA columns."""
    out = df.copy()
    for w in FAST_MA:
        out[f"ma_{w}"] = out["close"].rolling(window=w, min_periods=w).mean()
    for w in SLOW_MA:
        if f"ma_{w}" not in out.columns:
            out[f"ma_{w}"] = out["close"].rolling(window=w, min_periods=w).mean()
    return out


def daily_returns(df: pd.DataFrame) -> pd.Series:
    """Daily log or simple return. Use simple for consistency with typical Sharpe."""
    return df["close"].pct_change().dropna()


def annualized_sharpe(daily_ret: pd.Series, risk_free: float = 0.0) -> float | None:
    """Annualized Sharpe (excess return / vol * sqrt(252)). Returns None if insufficient or zero vol."""
    if daily_ret.empty or len(daily_ret) < 2:
        return None
    excess = daily_ret - (risk_free / TRADING_DAYS_PER_YEAR)
    std = excess.std()
    if std is None or pd.isna(std) or std == 0:
        return None
    return (excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def _trade_stats_from_mask(df: pd.DataFrame, mask: pd.Series) -> tuple[int, int, int]:
    """From a boolean mask (True = in market), return (n_trades, min_holding_days, max_holding_days)."""
    if not mask.any():
        return 0, 0, 0
    run_id = (mask.ne(mask.shift())).cumsum()
    in_market = df.loc[mask].copy()
    in_market["_run"] = run_id[mask].values
    holding = in_market.groupby("_run").size()
    n_trades = len(holding)
    min_hold = int(holding.min())
    max_hold = int(holding.max())
    return n_trades, min_hold, max_hold


def run_ma_crossover_backtest() -> pd.DataFrame:
    """
    For each (fast, slow) with fast < slow:
    - Mask to rows where fast MA > slow MA and both MAs are valid
    - Collect daily returns on those days only
    - Compute annualized Sharpe of that return series
    """
    df = load_n225()
    df = add_mas(df)
    df["return_1d"] = df["close"].pct_change()
    df = df.dropna(subset=["return_1d"]).reset_index(drop=True)

    rows = []
    for fast in FAST_MA:
        for slow in SLOW_MA:
            if fast >= slow:
                continue
            fast_col = f"ma_{fast}"
            slow_col = f"ma_{slow}"
            if fast_col not in df.columns or slow_col not in df.columns:
                continue
            # in market when fast MA > slow MA
            mask = (df[fast_col] > df[slow_col]) & df[fast_col].notna() & df[slow_col].notna()
            ret_in = df.loc[mask, "return_1d"]
            sharpe = annualized_sharpe(ret_in)
            n_trades, min_hold, max_hold = _trade_stats_from_mask(df, mask)
            rows.append(
                {
                    "fast_ma": fast,
                    "slow_ma": slow,
                    "sharpe": sharpe,
                    "n_trades": n_trades,
                    "min_holding_days": min_hold,
                    "max_holding_days": max_hold,
                }
            )

    return pd.DataFrame(rows)


def buy_and_hold_sharpe() -> float | None:
    """Annualized Sharpe for buy-and-hold over full N225 history."""
    df = load_n225()
    daily_ret = daily_returns(df)
    return annualized_sharpe(daily_ret)


def get_ma_trades(fast: int, slow: int) -> pd.DataFrame:
    """
    For a (fast, slow) MA pair, return a DataFrame of trades: entry_date, exit_date,
    holding_days, return (from entry close to exit close).
    """
    df = load_n225()
    df = add_mas(df)
    fast_col = f"ma_{fast}"
    slow_col = f"ma_{slow}"
    mask = (df[fast_col] > df[slow_col]) & df[fast_col].notna() & df[slow_col].notna()
    if not mask.any():
        return pd.DataFrame(columns=["entry_date", "exit_date", "holding_days", "return"])
    run_id = (mask.ne(mask.shift())).cumsum()
    in_market = df.loc[mask].copy()
    in_market["_run"] = run_id[mask].values
    trades = []
    for _run, grp in in_market.groupby("_run"):
        entry_date = grp["date"].iloc[0]
        exit_date = grp["date"].iloc[-1]
        hold_days = len(grp)
        ret = (grp["close"].iloc[-1] / grp["close"].iloc[0]) - 1
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "holding_days": hold_days,
                "return": ret,
            }
        )
    return pd.DataFrame(trades)


def plot_ma_trades_2020_2025(
    fast: int,
    slow: int,
    filename: str,
    start: str | pd.Timestamp = "2020-01-01",
    end: str | pd.Timestamp = "2025-12-31",
) -> None:
    """Bar plot: x = entry date of each fast>slow trade, y = return. Period [start, end]."""
    trades = get_ma_trades(fast, slow)
    if trades.empty:
        return
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    subset = trades[(trades["entry_date"] >= start) & (trades["entry_date"] <= end)]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = np.where(subset["return"] >= 0, "steelblue", "coral")
    ax.bar(subset["entry_date"], subset["return"] * 100, color=colors, width=15, edgecolor="none")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel(f"Trade entry date (first date MA{fast} > MA{slow})")
    ax.set_ylabel("Return (%)")
    ax.set_title(f"Nikkei 225 MA{fast} vs MA{slow}: return per trade ({start.strftime('%Y-%m')} – {end.strftime('%Y-%m')})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=25)
    plt.tight_layout()

    out = RESULT_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Bar plot saved to {out}")


def plot_holding_days_barchart(
    fast: int,
    slow: int,
    filename: str,
) -> None:
    """Bar chart: x = holding days, y = count of trades with that holding period."""
    trades = get_ma_trades(fast, slow)
    if trades.empty:
        return
    counts = trades["holding_days"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index.astype(int), counts.values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Holding days")
    ax.set_ylabel("Number of trades")
    ax.set_title(f"Nikkei 225 MA{fast} vs MA{slow}: distribution of holding period (total {len(trades)} trades)")
    plt.tight_layout()

    out = RESULT_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Holding-days bar chart saved to {out}")


def main() -> None:
    print("\n===== Nikkei 225 (N225) MA Crossover Backtest =====\n")

    # Buy and hold
    bh_sharpe = buy_and_hold_sharpe()
    if bh_sharpe is not None:
        print(f"Buy & hold (full period) annualized Sharpe: {bh_sharpe:.4f}")
    else:
        print("Buy & hold: insufficient data for Sharpe.")

    # MA pairs
    results = run_ma_crossover_backtest()
    if results.empty:
        print("No MA crossover results.")
        return

    print("\n----- Sharpe when fast MA > slow MA (only those days) -----\n")
    print(f"{'Fast':>6} {'Slow':>6} {'Sharpe':>10} {'Trades':>8} {'MinHold':>8} {'MaxHold':>8}")
    print("-" * 52)
    for _, r in results.iterrows():
        sharpe_str = f"{r['sharpe']:.4f}" if r["sharpe"] is not None and not np.isnan(r["sharpe"]) else "N/A"
        print(f"{r['fast_ma']:>6} {r['slow_ma']:>6} {sharpe_str:>10} {r['n_trades']:>8} {r['min_holding_days']:>8} {r['max_holding_days']:>8}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "n225_ma_backtest_results.csv"
    results.to_csv(out, index=False)
    print(f"\nResults saved to {out}")

    # Bar plots: MA5 vs MA10 and MA5 vs MA35 trades, 2020–2025
    plot_ma_trades_2020_2025(5, 10, "n225_ma5_ma10_trades_2020_2025.png")
    plot_ma_trades_2020_2025(5, 35, "n225_ma5_ma35_trades_2020_2025.png")

    # Holding-days distribution
    plot_holding_days_barchart(5, 10, "n225_ma5_ma10_holding_days.png")
    plot_holding_days_barchart(5, 35, "n225_ma5_ma35_holding_days.png")


if __name__ == "__main__":
    main()
