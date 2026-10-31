"""Download and validate historical IMD rainfall data for multiple years."""

from pathlib import Path

import pandas as pd
import requests
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DOWNLOAD_URL = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php"

YEARS = [2003]

BOUNDS = {
    "min_latitude": 28.2,
    "max_latitude": 28.9,
    "min_longitude": 76.7,
    "max_longitude": 77.6,
}


def download_year(session, year):
    """Download one year, validate it, and save its Delhi-region CSV."""

    raw_file = RAW_DIR / f"imd_rainfall_{year}.nc"
    csv_file = PROCESSED_DIR / f"delhi_rainfall_{year}.csv"

    # Do not overwrite previously collected data.
    if raw_file.exists() or csv_file.exists():
        print(f"[SKIP] {year}: an output file already exists.")
        print("       Inspect the existing files before retrying.")
        return False

    print(f"\nDownloading {year}...")

    response = session.post(
        DOWNLOAD_URL,
        data={"RF25": str(year)},
        timeout=(15, 180),
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()

    if "text/html" in content_type:
        raise ValueError("The server returned HTML, not a rainfall dataset.")

    if not response.content.startswith((b"CDF", b"\x89HDF")):
        raise ValueError("The response does not have a NetCDF signature.")

    # Validate the downloaded dataset before keeping it.
    raw_file.write_bytes(response.content)

    try:
        with xr.open_dataset(raw_file) as dataset:
            required = {"RAINFALL", "TIME", "LATITUDE", "LONGITUDE"}
            missing = required - set(dataset.variables)

            if missing:
                raise ValueError(
                    f"Missing dataset variables: {sorted(missing)}"
                )

            rainfall = dataset["RAINFALL"]

            if set(rainfall.dims) != {
                "TIME", "LATITUDE", "LONGITUDE"
            }:
                raise ValueError(
                    f"Unexpected rainfall dimensions: {rainfall.dims}"
                )

            units = rainfall.attrs.get("units", "").strip().lower()

            if units not in {
                "mm", "millimeter", "millimeters",
                "millimetre", "millimetres",
            }:
                raise ValueError(
                    f"Unexpected rainfall units: {units!r}"
                )

            dates = pd.to_datetime(dataset["TIME"].values)

            if len(dates) == 0:
                raise ValueError("The dataset contains no dates.")

            if (
                dates.min().year != year
                or dates.max().year != year
                or not (dates.year == year).all()
            ):
                raise ValueError(
                    f"The downloaded dates do not all belong to {year}."
                )

            expected_days = 366 if pd.Timestamp(
                year, 12, 31
            ).is_leap_year else 365

            if len(dates) != expected_days:
                raise ValueError(
                    f"Expected {expected_days} daily records, "
                    f"but found {len(dates)}."
                )

            region = rainfall.sel(
                LATITUDE=slice(
                    BOUNDS["min_latitude"],
                    BOUNDS["max_latitude"],
                ),
                LONGITUDE=slice(
                    BOUNDS["min_longitude"],
                    BOUNDS["max_longitude"],
                ),
            )

            if (
                region.sizes.get("LATITUDE", 0) == 0
                or region.sizes.get("LONGITUDE", 0) == 0
            ):
                raise ValueError("No grid cells found in the selected region.")

            frame = (
                region.to_dataframe(name="rainfall_mm")
                .reset_index()
                .rename(
                    columns={
                        "TIME": "date",
                        "LATITUDE": "latitude",
                        "LONGITUDE": "longitude",
                    }
                )
            )

        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["rainfall_mm"] = pd.to_numeric(
            frame["rainfall_mm"], errors="raise"
        )

        if frame.empty:
            raise ValueError("The regional dataset is empty.")

        if frame[
            ["date", "latitude", "longitude"]
        ].isna().any().any():
            raise ValueError("Missing dates or coordinates detected.")

        if frame["rainfall_mm"].isna().any():
            raise ValueError(
                "Missing rainfall values detected. Inspect source metadata."
            )

        if not frame["rainfall_mm"].map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        ).all():
            raise ValueError("Non-finite rainfall values detected.")

        if (frame["rainfall_mm"] < 0).any():
            raise ValueError(
                "Negative rainfall values detected. Inspect fill values "
                "and source metadata before deciding how to handle them."
            )

        if frame.duplicated(
            ["date", "latitude", "longitude"]
        ).any():
            raise ValueError("Duplicate date/grid-cell observations found.")

        grid_cells = frame[
            ["latitude", "longitude"]
        ].drop_duplicates().shape[0]

        if grid_cells != 12:
            raise ValueError(
                f"Expected 12 grid cells, found {grid_cells}."
            )

        expected_rows = expected_days * grid_cells

        if len(frame) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows, found {len(frame)}."
            )

        frame = frame.sort_values(
            ["date", "latitude", "longitude"]
        ).reset_index(drop=True)

        frame.to_csv(csv_file, index=False)

        print(f"[SUCCESS] {year}")
        print(f"  Download size: {len(response.content):,} bytes")
        print(f"  Dates: {frame['date'].min().date()} to "
              f"{frame['date'].max().date()}")
        print(f"  Grid cells: {grid_cells}")
        print(f"  Rows: {len(frame):,}")
        print(f"  CSV: {csv_file}")

        return True

    except Exception:
        # Preserve the downloaded NetCDF for investigation.
        print(f"[FAILED] {year}: validation or processing failed.")
        print(f"         Raw file preserved at: {raw_file}")
        raise


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    with requests.Session() as session:
        for year in YEARS:
            try:
                if download_year(session, year):
                    successful.append(year)
            except Exception as e:
                import traceback
                print(f"  Error for year {year}: {e}")
                traceback.print_exc()
                errors[year] = str(e)

    print("\n========== BATCH SUMMARY ==========")
    print(f"New downloads completed: {successful}")
    print(f"Failed years: {[year for year, _ in failed]}")

    if failed:
        print("\nErrors to investigate:")

        for year, error in failed:
            print(f"  {year}: {error}")

        raise RuntimeError(
            "Some years failed. Resolve the errors before proceeding."
        )

    print("\nBatch processing finished.")


if __name__ == "__main__":
    main()