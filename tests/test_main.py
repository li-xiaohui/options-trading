"""
Tests for main.py — QQQ short put backtest helpers and engine.
"""

import unittest.mock as mock
import os
import pandas as pd
import numpy as np
import pytest

# Set dummy API key and mock Polygon client before importing main (avoids real API calls)
os.environ.setdefault("POLYGON_API_KEY", "test_key_placeholder")
_client = mock.patch("polygon.RESTClient")
_client.start()

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
        # OTM put (K < S) has delta closer to 0
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
# get_underlying_prices
# =========================


class TestGetUnderlyingPrices:
    @mock.patch("main.yf.download")
    def test_returns_dataframe_with_close_and_date_index(self, mock_download):
        mock_download.return_value = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.DatetimeIndex(["2025-01-02", "2025-01-03"], name="Date"),
        )
        df = main.get_underlying_prices("2025-01-01", "2025-01-10")
        assert "close" in df.columns
        assert df.index.name == "date"
        assert len(df) == 2
        mock_download.assert_called_once_with(
            main.UNDERLYING, start="2025-01-01", end="2025-01-10", progress=False
        )


# =========================
# get_put_chain
# =========================


class TestGetPutChain:
    def test_returns_dataframe_with_ticker_strike_expiry(self):
        mock_contract = mock.MagicMock()
        mock_contract.ticker = "O:QQQ250117C00100000"
        mock_contract.strike_price = 500.0
        mock_contract.expiration_date = "2025-01-17"
        main.client.list_options_contracts = mock.MagicMock(return_value=[mock_contract])

        df = main.get_put_chain(pd.Timestamp("2025-01-17").date())
        assert not df.empty
        assert list(df.columns) == ["ticker", "strike", "expiry"]
        assert df.iloc[0]["ticker"] == "O:QQQ250117C00100000"
        assert df.iloc[0]["strike"] == 500.0

    def test_empty_contracts_returns_empty_dataframe(self):
        main.client.list_options_contracts = mock.MagicMock(return_value=[])
        df = main.get_put_chain(pd.Timestamp("2025-01-17").date())
        assert df.empty


# =========================
# get_option_close
# =========================


class TestGetOptionClose:
    def test_returns_close_when_aggs_non_empty(self):
        mock_agg = mock.MagicMock()
        mock_agg.close = 2.50
        main.client.get_aggs = mock.MagicMock(return_value=[mock_agg])
        out = main.get_option_close("O:QQQ250117P00500000", "2025-01-02")
        assert out == 2.50

    def test_returns_None_when_no_aggs(self):
        main.client.get_aggs = mock.MagicMock(return_value=[])
        out = main.get_option_close("O:QQQ250117P00500000", "2025-01-02")
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
        assert "Final capital" in out
        assert "100,125" in out or "100125" in out


# =========================
# backtest
# =========================


class TestBacktest:
    """Backtest engine with mocked data and API."""

    @mock.patch.object(main, "get_option_close")
    @mock.patch.object(main, "get_put_chain")
    @mock.patch.object(main, "get_underlying_prices")
    def test_returns_dataframe_and_no_crash(
        self, mock_prices, mock_chain, mock_option_close
    ):
        # Need > TARGET_DTE (30) dates so dates[:-30] is non-empty, and expiry in index
        start = pd.Timestamp("2026-11-01")
        dates = pd.date_range(start, periods=35, freq="B")
        mock_prices.return_value = pd.DataFrame(
            {"close": 500.0}, index=dates
        )
        # One put: 450 strike so delta is in range; expiry = first entry + 30
        expiry = start + pd.Timedelta(days=main.TARGET_DTE)
        mock_chain.return_value = pd.DataFrame([
            {
                "ticker": "O:QQQ261130P00450000",
                "strike": 450.0,
                "expiry": expiry,
            }
        ])
        def option_close(ticker, date):
            if date == expiry.date():
                return 0.0
            return 2.0
        mock_option_close.side_effect = option_close

        result = main.backtest()

        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "entry_date" in result.columns
            assert "pnl" in result.columns
            assert "capital" in result.columns
