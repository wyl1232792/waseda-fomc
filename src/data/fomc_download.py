"""Download historical FOMC meeting dates from the Federal Reserve."""

from __future__ import annotations

import argparse
import os
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATADIR")
if not DATA_DIR:
    raise RuntimeError("DATADIR must be set in .env or the environment")

START_YEAR = 2011
END_YEAR = 2018
DEFAULT_OUTPUT = Path(DATA_DIR) / "fomc_dates.parquet"
FEDERAL_RESERVE_URL = (
    "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
)
MEETING_PATTERN = re.compile(r"^(.*?)\s+Meeting\s+-\s+(\d{4})$")


def _month_number(value: str) -> int:
    """Convert a full or abbreviated English month name to its number."""
    for pattern in ("%B", "%b"):
        try:
            return datetime.strptime(value, pattern).month
        except ValueError:
            continue
    raise ValueError(f"unrecognized month name: {value}")


def parse_meeting_date(label: str, expected_year: int) -> date:
    """Return the policy-decision date represented by a Fed meeting label."""
    match = MEETING_PATTERN.fullmatch(label.strip())
    if not match:
        raise ValueError(f"unrecognized FOMC meeting label: {label}")

    date_text, year_text = match.groups()
    year = int(year_text)
    if year != expected_year:
        raise ValueError(f"expected year {expected_year}, found {year} in {label}")

    cross_month = re.fullmatch(
        r"[A-Za-z]+\s+\d{1,2}-([A-Za-z]+)\s+(\d{1,2})",
        date_text,
    )
    if cross_month:
        end_month_text, end_day_text = cross_month.groups()
    else:
        month_text, day_text = date_text.split(maxsplit=1)
        end_month_text = month_text.split("/")[-1]
        end_day_text = day_text.split("-")[-1]

    return date(year, _month_number(end_month_text), int(end_day_text))


def parse_fomc_page(html: str, year: int) -> pd.DataFrame:
    """Extract scheduled meeting dates and press-conference flags from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, object]] = []

    for heading in soup.find_all("h5"):
        label = " ".join(heading.stripped_strings)
        if not MEETING_PATTERN.fullmatch(label):
            continue

        panel = heading.find_parent("div", class_="panel")
        if panel is None:
            raise RuntimeError(f"could not find the content panel for {label}")

        records.append(
            {
                "date": parse_meeting_date(label, year),
                "PC": int("press conference" in panel.get_text(" ", strip=True).lower()),
            }
        )

    if not records:
        raise RuntimeError(f"no scheduled FOMC meetings found for {year}")

    meetings = pd.DataFrame.from_records(records)
    meetings["date"] = pd.to_datetime(meetings["date"])
    return meetings.sort_values("date", ignore_index=True)


def download_fomc_dates(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> pd.DataFrame:
    """Download scheduled FOMC meeting dates for the inclusive year range."""
    if start_year > end_year:
        raise ValueError(
            f"start year {start_year} must not be after end year {end_year}"
        )

    yearly_meetings: list[pd.DataFrame] = []
    headers = {"User-Agent": "waseda-fomc research data downloader"}

    for year in range(start_year, end_year + 1):
        url = FEDERAL_RESERVE_URL.format(year=year)
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        yearly_meetings.append(parse_fomc_page(response.text, year))

    meetings = pd.concat(yearly_meetings, ignore_index=True).sort_values("date")
    if meetings["date"].duplicated().any():
        duplicates = meetings.loc[meetings["date"].duplicated(), "date"]
        raise RuntimeError(f"duplicate FOMC dates found: {duplicates.dt.date.tolist()}")

    return meetings.reset_index(drop=True)


def save_fomc_dates(
    meetings: pd.DataFrame,
    output: Path = DEFAULT_OUTPUT,
) -> Path:
    """Write FOMC dates to a Parquet file and return its resolved path."""
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    meetings.to_parquet(output, engine="pyarrow", index=False)
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Download historical FOMC dates from the Federal Reserve."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=START_YEAR,
        help=f"first year to download (default: {START_YEAR})",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=END_YEAR,
        help=f"last year to download (default: {END_YEAR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"destination Parquet file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    """Download FOMC dates and save them as Parquet."""
    args = parse_args()
    meetings = download_fomc_dates(args.start_year, args.end_year)
    output = save_fomc_dates(meetings, args.output)
    print(
        f"Saved {len(meetings)} FOMC meeting dates "
        f"({int(meetings['PC'].sum())} with press conferences) to {output}"
    )


if __name__ == "__main__":
    main()
