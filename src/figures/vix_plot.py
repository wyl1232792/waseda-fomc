"""Plot average daily VIX changes around FOMC announcements."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import pandas as pd
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

load_dotenv()

DATA_DIR = os.getenv("DATADIR")
if not DATA_DIR:
    raise RuntimeError("DATADIR must be set in .env or the environment")

WINDOW = 5
VIX_DATA = Path(DATA_DIR) / "sp500_vix_daily.parquet"
FOMC_DATA = Path(DATA_DIR) / "fomc_dates.parquet"
DEFAULT_OUTPUT = Path("src/figures/vix_plot.png")


def load_data(
    vix_path: Path = VIX_DATA,
    fomc_path: Path = FOMC_DATA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the VIX and FOMC Parquet datasets."""
    missing = [path for path in (vix_path, fomc_path) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"required data file(s) not found: {missing_text}")

    prices = pd.read_parquet(vix_path)
    meetings = pd.read_parquet(fomc_path)

    if "vix_close" not in prices.columns:
        raise ValueError(f"{vix_path} does not contain a vix_close column")
    if not {"date", "PC"}.issubset(meetings.columns):
        raise ValueError(f"{fomc_path} must contain date and PC columns")

    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices.sort_index()
    meetings = meetings.loc[:, ["date", "PC"]].copy()
    meetings["date"] = pd.to_datetime(meetings["date"]).dt.normalize()

    invalid_flags = set(meetings["PC"].dropna().unique()) - {0, 1}
    if invalid_flags:
        raise ValueError(f"PC must contain only 0 and 1; found {invalid_flags}")

    return prices, meetings.sort_values("date", ignore_index=True)


def build_event_window(
    prices: pd.DataFrame,
    meetings: pd.DataFrame,
    window: int = WINDOW,
) -> pd.DataFrame:
    """Return daily VIX changes for each meeting's trading-day event window."""
    if window < 0:
        raise ValueError("window must be nonnegative")
    if prices.index.has_duplicates:
        raise ValueError("VIX data contains duplicate dates")

    vix_change = prices["vix_close"].astype(float).diff()
    trading_dates = prices.index
    records: list[dict[str, object]] = []

    for meeting in meetings.itertuples(index=False):
        meeting_date = pd.Timestamp(meeting.date)
        if meeting_date not in trading_dates:
            raise ValueError(f"FOMC date {meeting_date.date()} is not in VIX data")

        position = trading_dates.get_loc(meeting_date)
        if not isinstance(position, int):
            raise ValueError(f"FOMC date {meeting_date.date()} is not unique")

        start = position - window
        stop = position + window + 1
        if start < 0 or stop > len(vix_change):
            raise ValueError(
                f"VIX data does not cover the full window around "
                f"{meeting_date.date()}"
            )

        changes = vix_change.iloc[start:stop]
        if changes.isna().any():
            raise ValueError(
                f"VIX changes contain missing values around {meeting_date.date()}"
            )

        for event_day, change in zip(
            range(-window, window + 1),
            changes,
            strict=True,
        ):
            records.append(
                {
                    "meeting_date": meeting_date,
                    "event_day": event_day,
                    "vix_change": float(change),
                    "PC": int(meeting.PC),
                }
            )

    return pd.DataFrame.from_records(records)


def average_vix_changes(event_window: pd.DataFrame) -> pd.DataFrame:
    """Average VIX changes by event day and press-conference status."""
    return (
        event_window.groupby(["event_day", "PC"], as_index=False)
        .agg(average_vix_change=("vix_change", "mean"), meetings=("vix_change", "size"))
        .sort_values(["PC", "event_day"], ignore_index=True)
    )


def plot_vix_changes(
    averages: pd.DataFrame,
    output: Path = DEFAULT_OUTPUT,
) -> Path:
    """Plot the two event-study series and save the figure."""
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        1: ("Press conference", "red"),
        0: ("No press conference", "blue"),
    }

    for flag, (label, color) in styles.items():
        series = averages.loc[averages["PC"] == flag]
        if series.empty:
            raise ValueError(f"no meetings found for {label.lower()}")
        ax.plot(
            series["event_day"],
            series["average_vix_change"],
            color=color,
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set(
        title="Average Daily VIX Change Around FOMC Announcements",
        xlabel="Trading days relative to FOMC announcement",
        ylabel="Average daily change in VIX (points)",
        xticks=range(-WINDOW, WINDOW + 1),
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Plot average VIX changes around FOMC announcements."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"destination image file (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    """Build the event study and save its figure."""
    args = parse_args()
    prices, meetings = load_data()
    event_window = build_event_window(prices, meetings)
    averages = average_vix_changes(event_window)
    output = plot_vix_changes(averages, args.output)
    print(
        f"Saved VIX event-study figure using {meetings['PC'].eq(1).sum()} "
        f"press-conference and {meetings['PC'].eq(0).sum()} "
        f"non-press-conference meetings to {output}"
    )


if __name__ == "__main__":
    main()
