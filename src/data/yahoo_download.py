"""Download daily S&P 500 and VIX market data from Yahoo Finance."""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATADIR")
if not DATA_DIR:
    raise RuntimeError("DATADIR must be set in .env or the environment")

START_DATE = date(1994, 1, 1)
DEFAULT_OUTPUT = Path(DATA_DIR) / "sp500_vix_daily.parquet"
SYMBOLS = {
    "sp500": "^GSPC",
    "vix": "^VIX",
}


def download_prices(
    start: date = START_DATE,
    end: date | None = None,
) -> pd.DataFrame:
    """Return daily S&P 500 and VIX prices from *start* through *end*.

    Yahoo Finance treats ``end`` as exclusive, so one day is added when making
    the request. The returned frame uses one row per trading date and prefixes
    every price field with its series name.
    """
    end = end or date.today()
    if start > end:
        raise ValueError(f"start date {start} must not be after end date {end}")

    frames: list[pd.DataFrame] = []
    request_end = end + timedelta(days=1)

    for name, ticker in SYMBOLS.items():
        prices = yf.download(
            ticker,
            start=start.isoformat(),
            end=request_end.isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            multi_level_index=False,
        )
        if prices.empty:
            raise RuntimeError(f"Yahoo Finance returned no data for {ticker}")

        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices.index.name = "date"
        prices.columns = [
            f"{name}_{str(column).lower().replace(' ', '_')}"
            for column in prices.columns
        ]
        frames.append(prices)

    combined = pd.concat(
        frames, axis="columns", join="outer", sort=False
    ).sort_index()
    return combined.loc[
        (combined.index.date >= start) & (combined.index.date <= end)
    ]


def save_prices(prices: pd.DataFrame, output: Path = DEFAULT_OUTPUT) -> Path:
    """Write *prices* to a Parquet file and return its resolved path."""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output, engine="pyarrow", index=True)
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Download daily S&P 500 and VIX prices from Yahoo Finance."
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=START_DATE,
        help="first date to download in YYYY-MM-DD format (default: 1994-01-01)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="last date to download in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"destination Parquet file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    """Download the requested history and save it as Parquet."""
    args = parse_args()
    end = args.end or date.today()
    prices = download_prices(start=args.start, end=end)
    output = save_prices(prices, args.output)
    print(
        f"Saved {len(prices):,} rows from "
        f"{prices.index.min().date()} through {prices.index.max().date()} "
        f"to {output}"
    )


if __name__ == "__main__":
    main()
