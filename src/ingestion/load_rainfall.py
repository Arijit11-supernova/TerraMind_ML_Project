"""Load and validate IMD gridded rainfall observations."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "raw" / "imd_rainfall_2024.nc"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "delhi_rainfall_2024.csv"
)

# Approximate geographical bounds for Delhi and its surroundings.
DELHI_BOUNDS = {
    "min_latitude": 28.2,
    "max_latitude": 28.9,
    "min_longitude": 76.7,
    "max_longitude": 77.6,
}


def load_rainfall(
    input_path=DEFAULT_INPUT,
    min_latitude=DELHI_BOUNDS["min_latitude"],
    max_latitude=DELHI_BOUNDS["max_latitude"],
    min_longitude=DELHI_BOUNDS["min_longitude"],
    max_longitude=DELHI_BOUNDS["max_longitude"],
):
    """Load daily rainfall observations within a geographical bounding box."""

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Rainfall dataset not found: {input_path}"
        )

    if min_latitude > max_latitude:
        raise ValueError(
            "Minimum latitude must not exceed maximum latitude."
        )

    if min_longitude > max_longitude:
        raise ValueError(
            "Minimum longitude must not exceed maximum longitude."
        )

    with xr.open_dataset(input_path) as dataset:

        # Validate the required variables and coordinates.
        required_variables = {
            "RAINFALL",
            "TIME",
            "LATITUDE",
            "LONGITUDE",
        }

        available_variables = set(dataset.variables)
        missing_variables = required_variables - available_variables

        if missing_variables:
            raise ValueError(
                "Dataset is missing required variables or coordinates: "
                f"{sorted(missing_variables)}"
            )

        rainfall = dataset["RAINFALL"]

        expected_dimensions = {
            "TIME",
            "LATITUDE",
            "LONGITUDE",
        }

        if set(rainfall.dims) != expected_dimensions:
            raise ValueError(
                f"Unexpected rainfall dimensions: {rainfall.dims}"
            )

        # Confirm the rainfall measurement units.
        units = rainfall.attrs.get("units", "").strip().lower()

        accepted_units = {
            "mm",
            "millimeter",
            "millimeters",
            "millimetre",
            "millimetres",
        }

        if units not in accepted_units:
            raise ValueError(
                "Expected rainfall units in millimetres; "
                f"found {units!r}."
            )

        # Select the grid cells covering the requested region.
        region = rainfall.sel(
            LATITUDE=slice(min_latitude, max_latitude),
            LONGITUDE=slice(min_longitude, max_longitude),
        )

        if region.sizes.get("LATITUDE", 0) == 0:
            raise ValueError(
                "No latitude grid cells found in the requested region."
            )

        if region.sizes.get("LONGITUDE", 0) == 0:
            raise ValueError(
                "No longitude grid cells found in the requested region."
            )

        # Convert the rainfall array into tabular form.
        frame = region.to_dataframe(
            name="rainfall_mm"
        ).reset_index()

    # Standardize column names.
    frame = frame.rename(
        columns={
            "TIME": "date",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
        }
    )

    # Validate data types.
    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="raise",
    )

    frame["rainfall_mm"] = pd.to_numeric(
        frame["rainfall_mm"],
        errors="raise",
    )

    if frame.empty:
        raise ValueError(
            "No rainfall observations found in the requested region."
        )

    # Geographic coordinates and dates must not be missing.
    coordinate_columns = [
        "date",
        "latitude",
        "longitude",
    ]

    if frame[coordinate_columns].isna().any().any():
        raise ValueError(
            "Missing dates or geographic coordinates detected."
        )

    # Check rainfall values without silently discarding missing values.
    valid_rainfall = frame["rainfall_mm"].dropna().to_numpy(
        dtype=float
    )

    if not np.isfinite(valid_rainfall).all():
        raise ValueError(
            "Infinite rainfall values detected."
        )

    if (valid_rainfall < 0).any():
        raise ValueError(
            "Negative rainfall values detected. Investigate the "
            "source data before using it for modelling."
        )

    # Every date and grid cell should have at most one observation.
    if frame.duplicated(
        ["date", "latitude", "longitude"]
    ).any():
        raise ValueError(
            "Duplicate date and grid-cell observations detected."
        )

    return frame.sort_values(
        ["date", "latitude", "longitude"]
    ).reset_index(drop=True)


def save_rainfall(
    frame,
    output_path=DEFAULT_OUTPUT,
):
    """Save validated rainfall observations to a CSV file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output_path,
        index=False,
    )

    return output_path


if __name__ == "__main__":

    observations = load_rainfall()
    output_file = save_rainfall(observations)

    unique_grid_cells = observations[
        ["latitude", "longitude"]
    ].drop_duplicates().shape[0]

    print("Rainfall ingestion completed successfully.")
    print(f"Output file: {output_file}")
    print(f"Total rows: {len(observations):,}")
    print(f"Unique grid cells: {unique_grid_cells:,}")

    print(
        f"Date range: {observations['date'].min().date()} "
        f"to {observations['date'].max().date()}"
    )

    print(
        "Missing rainfall values: "
        f"{observations['rainfall_mm'].isna().sum():,}"
    )

    print("\nRainfall summary (mm):")
    print(observations["rainfall_mm"].describe())