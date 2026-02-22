from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
RESULT_DIR = Path(__file__).resolve().parent / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FAST_MA = [5, 10, 20, 35, 50, 100]
SLOW_MA = [10, 20, 35, 50, 100]
TRADING_DAYS_PER_YEAR = 252
