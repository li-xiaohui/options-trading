"""
Tests for main.py — QQQ short put backtest helpers and engine (local data only).
"""

import unittest.mock as mock
import pandas as pd
import numpy as np
import pytest

import main


# =========================
# put_delta
# =========================


class TestPutDelta:
    """Black-Scholes put delta."""

    def test_returns_nan_when_T_zero(self):
        assert np.isnan(main.put_delta(S=100, K=100, T=0, r=0.04, iv=0.25))

    def test_returns_nan_when_T_negative(self):
        assert np.isnan(main.put_delta(S=100, K=100, T=-0.1, r=0.04, iv=0.25))

    def test_returns_nan_when_iv_zero(self):
        assert np.isnan(main.put_delta(S=100, K=100, T=0.5, r=0.04, iv=0))

    def test_returns_nan_when_iv_negative(self):
        assert np.isnan(main.put_delta(S=100, K=100, T=0.5, r=0.04, iv=-0.1))

    def test_put_delta_between_minus_one_and_zero(self):
        delta = main.put_delta(S=100, K=100, T=1.0, r=0.04, iv=0.25)
        assert -1 <= delta <= 0

    def test_OTM_put_delta_closer_to_zero(self):
        otm = main.put_delta(S=100, K=90, T=0.5, r=0.04, iv=0.25)
        itm = main.put_delta(S=100, K=110, T=0.5, r=0.04, iv=0.25)
        assert otm > itm
        assert -1 <= otm <= 0 and -1 <= itm <= 0


# =========================
# select_strike_by_delta
# =========================


class TestSelectStrikeByDelta:
    def test_returns_row_closest_to_target_delta(self):
        df = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "strike": [100, 105, 110],
            "delta": [-0.30, -0.25, -0.20],
        })
        row = main.select_strike_by_delta(df, target_delta=-0.25)
        assert row["ticker"] == "B"
        assert row["strike"] == 105
        assert row["delta"] == -0.25

    def test_does_not_mutate_input_dataframe(self):
        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "strike": [100, 105],
            "delta": [-0.30, -0.25],
        })
        original_cols = list(df.columns)
        main.select_strike_by_delta(df, target_delta=-0.25)
        assert "delta_diff" not in df.columns
        assert list(df.columns) == original_cols

    def test_single_row_returns_that_row(self):
        df = pd.DataFrame({
            "ticker": ["X"],
            "strike": [100],
            "delta": [-0.20],
        })
        row = main.select_strike_by_delta(df, target_delta=-0.25)
        assert row["ticker"] == "X"


# =========================
# load_underlying_prices
# =========================


class TestLoadUnderlyingPrices:
    def test_returns_dataframe_with_close_and_date_index(self, tmp_path):
        (tmp_path / "underlying.csv").write_text(
            "date,close\n2025-01-02,100.0\n2025-01-03,101.0\n"
        )
        with mock.patch.object(main, "DATA_DIR", tmp_path):
            df = main.load_underlying_prices("2025-01-01", "2025-01-10")
        assert "close" in df.columns
        assert df.index.name == "date"
        assert len(df) == 2

    def test_raises_when_file_missing(self):
        with mock.patch.object(main, "DATA_DIR", __import__("pathlib").Path("/nonexistent_data")):
            with pytest.raises(FileNotFoundError, match="Run download_data.py"):
                main.load_underlying_prices("2025-01-01", "2025-01-10")


# =========================
# load_put_chain
# =========================


class TestLoadPutChain:
    def test_returns_dataframe_from_csv(self, tmp_path):
        (tmp_path / "put_chains").mkdir()
        (tmp_path / "put_chains" / "expiry_2025-01-17.csv").write_text(
            "ticker,strike,expiry\nO:QQQ250117P00500000,500.0,2025-01-17\n"
        )
        with mock.patch.object(main, "DATA_DIR", tmp_path):
            df = main.load_put_chain(pd.Timestamp("2025-01-17").date(), data_dir=tmp_path)
        assert not df.empty
        assert list(df.columns) == ["ticker", "strike", "expiry"]
        assert df.iloc[0]["ticker"] == "O:QQQ250117P00500000"
        assert df.iloc[0]["strike"] == 500.0

    def test_returns_empty_when_file_missing(self, tmp_path):
        (tmp_path / "put_chains").mkdir()
        with mock.patch.object(main, "DATA_DIR", tmp_path):
            df = main.load_put_chain(pd.Timestamp("2025-01-17").date())
        assert df.empty


# =========================
# get_option_close_from_lookup
# =========================


class TestGetOptionCloseFromLookup:
    def test_returns_close_when_key_present(self):
        d = pd.Timestamp("2025-01-02").date()
        lookup = {("O:QQQ250117P00500000", d): 2.50}
        out = main.get_option_close_from_lookup("O:QQQ250117P00500000", d, lookup)
        assert out == 2.50

    def test_returns_None_when_key_missing(self):
        lookup = {}
        out = main.get_option_close_from_lookup(
            "O:QQQ250117P00500000", "2025-01-02", lookup
        )
        assert out is None


# =========================
# analyze
# =========================


class TestAnalyze:
    def test_empty_dataframe_does_not_raise(self, capsys):
        main.analyze(pd.DataFrame())
        out = capsys.readouterr().out
        assert "No trades executed" in out

    def test_computes_and_prints_metrics(self, capsys):
        results = pd.DataFrame([
            {"pnl": 100.0, "capital": 100_100},
            {"pnl": -50.0, "capital": 100_050},
            {"pnl": 75.0, "capital": 100_125},
        ])
        main.analyze(results)
        out = capsys.readouterr().out
        assert "BACKTEST RESULTS" in out
        assert "Trades: 3" in out
        assert "Total PnL" in out
        assert "Win rate" in out
        assert "Max drawdown" in out
        assert "100,125" in out or "100125" in out


# =========================
# backtest
# =========================


class TestBacktest:
    """Backtest engine with mocked local data."""

    @mock.patch.object(main, "load_option_daily")
    @mock.patch.object(main, "load_put_chain")
    @mock.patch.object(main, "load_underlying_prices")
    def test_returns_dataframe_and_no_crash(
        self, mock_prices, mock_chain, mock_option_daily
    ):
        start = pd.Timestamp("2026-11-01")
        dates = pd.date_range(start, periods=35, freq="B")
        mock_prices.return_value = pd.DataFrame(
            {"close": 500.0}, index=dates
        )
        expiry = start + pd.Timedelta(days=main.TARGET_DTE)
        mock_chain.return_value = pd.DataFrame([
            {
                "ticker": "O:QQQ261130P00450000",
                "strike": 450.0,
                "expiry": expiry,
            }
        ])
        mock_option_daily.return_value = pd.DataFrame([
            {"ticker": "O:QQQ261130P00450000", "date": start.date(), "close": 2.0},
            {"ticker": "O:QQQ261130P00450000", "date": expiry.date(), "close": 0.0},
        ])

        result = main.backtest()

        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "entry_date" in result.columns
            assert "pnl" in result.columns
            assert "capital" in result.columns
