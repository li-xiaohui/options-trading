"""
SPX January backtest: hit rate and average return from 16 Jan to year end.

Scenarios:
- 2.1 Conditional on return > 1% between 1 Jan and 15 Jan
- 2.2 Conditional on return <= 1% between 1 Jan and 15 Jan
- 2.3 Unconditional (all years)
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SPX_PATH = DATA_DIR / "spx_daily.csv"

# Return threshold for early January (1 Jan to 15 Jan)
JAN_EARLY_THRESHOLD_PCT = 1.0

# Return threshold for first half (1 Jan to 30 Jun) — used for Jul–YE analysis
H1_THRESHOLD_PCT = 5.0


def load_spx() -> pd.DataFrame:
    """Load SPX daily (date, close)."""
    if not SPX_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SPX_PATH}. Run: python -m backtest_index.download_spx"
        )
    df = pd.read_csv(SPX_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].sort_values("date").drop_duplicates(subset=["date"])


def get_price_on_or_before(df: pd.DataFrame, ref_date: pd.Timestamp) -> float | None:
    """Close on the last trading day on or before ref_date."""
    sub = df[df["date"].dt.date <= ref_date.date()]
    if sub.empty:
        return None
    return sub.iloc[-1]["close"]


def get_price_on_or_after(df: pd.DataFrame, ref_date: pd.Timestamp) -> float | None:
    """Close on the first trading day on or after ref_date."""
    sub = df[df["date"].dt.date >= ref_date.date()]
    if sub.empty:
        return None
    return sub.iloc[0]["close"]


def get_last_trading_day_of_year(df: pd.DataFrame, year: int) -> pd.Timestamp | None:
    """Last trading date in the given year."""
    sub = df[df["date"].dt.year == year]
    if sub.empty:
        return None
    return sub.iloc[-1]["date"]


def get_this_year_jan1_15_return() -> float | None:
    """
    Return this year's SPX price return between 1 Jan and 15 Jan.
    (close on or before 15 Jan) / (close on or before 1 Jan) - 1.
    Returns None if data is missing or insufficient.
    """
    df = load_spx()
    year = pd.Timestamp.now().year
    jan1 = pd.Timestamp(year=year, month=1, day=1)
    jan15 = pd.Timestamp(year=year, month=1, day=15)
    p_jan1 = get_price_on_or_before(df, jan1)
    p_jan15 = get_price_on_or_before(df, jan15)
    if p_jan1 is None or p_jan15 is None or p_jan1 <= 0 or p_jan15 <= 0:
        return None
    return (p_jan15 / p_jan1) - 1


def run_backtest() -> pd.DataFrame:
    """
    For each year, compute:
    - return_jan1_15: (close on or before Jan 15) / (close on or before Jan 1) - 1
    - return_jan16_ye: (close at year end) / (close on or after Jan 16) - 1
    """
    df = load_spx()
    rows = []

    for year in range(df["date"].min().year + 1, df["date"].max().year + 1):
        jan1 = pd.Timestamp(year=year, month=1, day=1)
        jan15 = pd.Timestamp(year=year, month=1, day=15)
        jan16 = pd.Timestamp(year=year, month=1, day=16)

        p_jan1 = get_price_on_or_before(df, jan1)
        p_jan15 = get_price_on_or_before(df, jan15)
        p_jan16 = get_price_on_or_after(df, jan16)
        last_dt = get_last_trading_day_of_year(df, year)
        p_ye = get_price_on_or_before(df, last_dt) if last_dt is not None else None

        if p_jan1 is None or p_jan15 is None or p_jan16 is None or p_ye is None:
            continue
        if p_jan1 <= 0 or p_jan15 <= 0 or p_jan16 <= 0 or p_ye <= 0:
            continue

        ret_jan1_15 = (p_jan15 / p_jan1) - 1
        ret_jan16_ye = (p_ye / p_jan16) - 1

        rows.append(
            {
                "year": year,
                "return_jan1_15": ret_jan1_15,
                "return_jan16_ye": ret_jan16_ye,
                "jan1_15_gt_1pct": ret_jan1_15 > (JAN_EARLY_THRESHOLD_PCT / 100),
            }
        )

    return pd.DataFrame(rows)


def run_backtest_jul_ye() -> pd.DataFrame:
    """
    For each year, compute:
    - return_jan1_jun30: (close on or before Jun 30) / (close on or before Jan 1) - 1
    - return_jul1_ye: (close at year end) / (close on or after Jul 1) - 1
    """
    df = load_spx()
    rows = []

    for year in range(df["date"].min().year + 1, df["date"].max().year + 1):
        jan1 = pd.Timestamp(year=year, month=1, day=1)
        jun30 = pd.Timestamp(year=year, month=6, day=30)
        jul1 = pd.Timestamp(year=year, month=7, day=1)

        p_jan1 = get_price_on_or_before(df, jan1)
        p_jun30 = get_price_on_or_before(df, jun30)
        p_jul1 = get_price_on_or_after(df, jul1)
        last_dt = get_last_trading_day_of_year(df, year)
        p_ye = get_price_on_or_before(df, last_dt) if last_dt is not None else None

        if p_jan1 is None or p_jun30 is None or p_jul1 is None or p_ye is None:
            continue
        if p_jan1 <= 0 or p_jun30 <= 0 or p_jul1 <= 0 or p_ye <= 0:
            continue

        ret_jan1_jun30 = (p_jun30 / p_jan1) - 1
        ret_jul1_ye = (p_ye / p_jul1) - 1

        rows.append(
            {
                "year": year,
                "return_jan1_jun30": ret_jan1_jun30,
                "return_jul1_ye": ret_jul1_ye,
                "jan1_jun30_gt_5pct": ret_jan1_jun30 > (H1_THRESHOLD_PCT / 100),
            }
        )

    return pd.DataFrame(rows)


def _jul_ye_stats(returns: pd.Series) -> dict:
    """Hit rate, avg return, std, avg win, avg loss, win/loss ratio for a series of returns."""
    if returns.empty:
        return {
            "n": 0,
            "hit_rate": None,
            "avg_return": None,
            "std": None,
            "avg_win": None,
            "avg_loss": None,
            "win_loss_ratio": None,
        }
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    if avg_loss != 0:
        win_loss_ratio = abs(avg_win / avg_loss)
    else:
        win_loss_ratio = float("inf") if avg_win > 0 else None
    return {
        "n": len(returns),
        "hit_rate": (returns > 0).mean(),
        "avg_return": returns.mean(),
        "std": returns.std(),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
    }


def analyze_jul_ye() -> None:
    """
    SPX return 1 Jul to year end (full history since 1950), compare:
    1. When SPX return > 5% between 1 Jan and 30 Jun
    2. When SPX return <= 5% between 1 Jan and 30 Jun

    Prints: hit rate, avg return, std, avg win, avg loss, win/loss ratio.
    """
    results = run_backtest_jul_ye()
    if results.empty:
        print("No years with valid data for Jul–YE backtest.")
        return

    r = results["return_jul1_ye"]
    gt5 = results[results["jan1_jun30_gt_5pct"]]["return_jul1_ye"]
    le5 = results[~results["jan1_jun30_gt_5pct"]]["return_jul1_ye"]

    def print_stats(label: str, s: dict) -> None:
        if s["n"] == 0:
            print(f"{label}: n=0 (no data)")
            return
        hr = f"{s['hit_rate']:.2%}" if s["hit_rate"] is not None else "N/A"
        ar = f"{s['avg_return']:.2%}" if s["avg_return"] is not None else "N/A"
        std = f"{s['std']:.2%}" if s["std"] is not None and pd.notna(s["std"]) else "N/A"
        aw = f"{s['avg_win']:.2%}" if s["avg_win"] is not None else "N/A"
        al = f"{s['avg_loss']:.2%}" if s["avg_loss"] is not None else "N/A"
        wl = f"{s['win_loss_ratio']:.2f}" if s["win_loss_ratio"] is not None and s["win_loss_ratio"] != float("inf") else ("∞" if s["win_loss_ratio"] == float("inf") else "N/A")
        print(f"{label}:")
        print(f"  n={s['n']}, hit_rate={hr}, avg_return={ar}, std={std}")
        print(f"  avg_win={aw}, avg_loss={al}, win/loss_ratio={wl}")

    print("\n===== SPX: 1 Jul to year end (conditional on 1 Jan–30 Jun return) =====\n")
    print("1. When SPX return > 5% between 1 Jan and 30 Jun:")
    print_stats("  H1 > 5%", _jul_ye_stats(gt5))
    print("\n2. When SPX return <= 5% between 1 Jan and 30 Jun:")
    print_stats("  H1 <= 5%", _jul_ye_stats(le5))
    print("\n3. Unconditional (all years):")
    print_stats("  All", _jul_ye_stats(r))


def analyze(results: pd.DataFrame) -> None:
    """Print hit rate and avg return for each scenario."""
    if results.empty:
        print("No years with valid data.")
        return

    def stats(sub: pd.DataFrame, label: str) -> None:
        if sub.empty:
            print(f"{label}: n=0 (no data)")
            return
        hit_rate = (sub["return_jan16_ye"] > 0).mean()
        avg_ret = sub["return_jan16_ye"].mean()
        print(f"{label}: n={len(sub)}, hit_rate={hit_rate:.2%}, avg_return_jan16_ye={avg_ret:.2%}")

    print("\n===== SPX: 16 Jan to year end (full history) =====\n")

    gt1 = results[results["jan1_15_gt_1pct"]]
    le1 = results[~results["jan1_15_gt_1pct"]]

    print("2.1 If return > 1% between 1 Jan and 15 Jan:")
    stats(gt1, "  Conditional (>1%)")

    print("\n2.2 If return <= 1% between 1 Jan and 15 Jan:")
    stats(le1, "  Conditional (<=1%)")

    print("\n2.3 Unconditional (all years):")
    stats(results, "  Unconditional")


def main() -> None:
    results = run_backtest()
    analyze(results)

    analyze_jul_ye()

    out = DATA_DIR / "spx_jan_backtest_results.csv"
    if not results.empty:
        results.to_csv(out, index=False)
        print(f"\nPer-year results saved to {out}")


if __name__ == "__main__":
    main()
    ret = get_this_year_jan1_15_return()
    pct = f"{ret:.2%}" if ret is not None else "N/A"
    print(f"This year's SPX price return between 1 Jan and 15 Jan: {pct}")

