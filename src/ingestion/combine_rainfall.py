"""Combine and validate multiple years of processed rainfall data."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

START_YEAR = 2000
END_YEAR = 2024
EXPECTED_GRID_CELLS = 12

OUTPUT_FILE = (
    PROCESSED_DIR / "delhi_rainfall_2000_2024.csv"
)

REQUIRED_COLUMNS = [
    "date",
    "latitude",
    "longitude",
    "rainfall_mm",
]


def expected_days_in_year(year):
    """Return the expected number of days in a calendar year."""
    return (
        366
        if pd.Timestamp(
            year=year, month=12, day=31
        ).day_of_year == 366
        else 365
    )

def load_yearly_files():
    """Discover yearly CSV files and validate their structure."""

    frames = []
    discovered_years = set()

    for file_path in sorted(
        PROCESSED_DIR.glob("delhi_rainfall_*.csv")
    ):
        # Ignore combined datasets.
        if file_path == OUTPUT_FILE:
            continue

        suffix = file_path.stem.removeprefix("delhi_rainfall_")

        # Accept only files whose remaining suffix is a four-digit year.
        if not (suffix.isdigit() and len(suffix) == 4):
            continue

        year = int(suffix)

        # Ignore yearly files outside our requested range.
        if not START_YEAR <= year <= END_YEAR:
            continue

        if year in discovered_years:
            raise ValueError(
                f"Multiple files found for year {year}."
            )

        frame = pd.read_csv(
            file_path,
            parse_dates=["date"],
        )

        missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)

        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if frame.empty:
            raise ValueError(
                f"The dataset is empty: {file_path.name}"
            )

        if frame["date"].isna().any():
            raise ValueError(
                f"Invalid or missing dates in {file_path.name}"
            )

        if not (frame["date"].dt.year == year).all():
            raise ValueError(
                f"Dates do not match expected year {year} "
                f"in {file_path.name}"
            )

        expected_days = expected_days_in_year(year)
        expected_rows = expected_days * EXPECTED_GRID_CELLS

        if len(frame) != expected_rows:
            raise ValueError(
                f"{file_path.name} contains {len(frame):,} rows; "
                f"expected {expected_rows:,} "
                f"({expected_days} days x "
                f"{EXPECTED_GRID_CELLS} grid cells)."
            )

        actual_dates = frame["date"].dt.normalize().nunique()

        if actual_dates != expected_days:
            raise ValueError(
                f"{file_path.name} contains {actual_dates} unique "
                f"dates; expected {expected_days}."
            )

        frames.append(frame[REQUIRED_COLUMNS])
        discovered_years.add(year)

        print(
            f"[LOADED] {year}: {len(frame):,} rows, "
            f"{actual_dates} dates"
        )

    expected_years = set(range(START_YEAR, END_YEAR + 1))
    missing_years = sorted(expected_years - discovered_years)

    if missing_years:
        raise FileNotFoundError(
            "Missing yearly rainfall files for: "
            f"{missing_years}"
        )

    if not frames:
        raise FileNotFoundError(
            f"No yearly rainfall files found in {PROCESSED_DIR}"
        )

    return pd.concat(frames, ignore_index=True)


def validate_combined_data(data):
    """Validate observations across all included years."""

    if data[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing values detected in required columns."
        )

    numeric_columns = [
        "latitude",
        "longitude",
        "rainfall_mm",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if not np.isfinite(
        data[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite values detected in numeric columns."
        )

    if (data["rainfall_mm"] < 0).any():
        raise ValueError(
            "Negative rainfall values detected."
        )

    if data.duplicated(
        ["date", "latitude", "longitude"]
    ).any():
        raise ValueError(
            "Duplicate date and grid-cell observations detected."
        )

    # Validate that each year has the same spatial grid.
    grid_cells_by_year = (
        data.groupby(data["date"].dt.year)
        .apply(
            lambda group: group[
                ["latitude", "longitude"]
            ].drop_duplicates().shape[0],
            include_groups=False,
        )
    )

    if not (grid_cells_by_year == EXPECTED_GRID_CELLS).all():
        raise ValueError(
            "Unexpected number of grid cells in one or more years:\n"
            f"{grid_cells_by_year.to_string()}"
        )

    # Check that every year uses the same actual grid coordinates.
    grids = {}

    for year, group in data.groupby(data["date"].dt.year):
        grid = set(
            zip(group["latitude"], group["longitude"])
        )
        grids[year] = grid

    reference_year = START_YEAR
    reference_grid = grids[reference_year]

    inconsistent_years = [
        year
        for year, grid in grids.items()
        if grid != reference_grid
    ]

    if inconsistent_years:
        raise ValueError(
            "Spatial grid coordinates differ from "
            f"{reference_year} in years: {inconsistent_years}"
        )

    return grid_cells_by_year


def main():
    data = load_yearly_files()

    grid_cells_by_year = validate_combined_data(data)

    data = data.sort_values(
        ["date", "latitude", "longitude"]
    ).reset_index(drop=True)

    # Validate the expected overall observation count.
    expected_total_rows = sum(
        expected_days_in_year(year) * EXPECTED_GRID_CELLS
        for year in range(START_YEAR, END_YEAR + 1)
    )

    if len(data) != expected_total_rows:
        raise ValueError(
            f"Combined dataset contains {len(data):,} rows; "
            f"expected {expected_total_rows:,}."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 55)
    print("MULTI-YEAR RAINFALL DATASET CREATED SUCCESSFULLY")
    print("=" * 55)

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total observations: {len(data):,}")
    print(
        f"Date range: {data['date'].min().date()} to "
        f"{data['date'].max().date()}"
    )
    print(f"Years included: {START_YEAR}-{END_YEAR}")
    print(f"Expected grid cells per year: {EXPECTED_GRID_CELLS}")

    print("\nGrid cells by year:")
    print(grid_cells_by_year.to_string())

    print("\nRainfall summary (mm):")
    print(
        data["rainfall_mm"].describe(
            percentiles=[0.90, 0.95, 0.99]
        )
    )


if __name__ == "__main__":
    main()
