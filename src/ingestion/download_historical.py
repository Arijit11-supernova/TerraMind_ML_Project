"""Download and validate one year of historical IMD rainfall data."""

from pathlib import Path

import requests
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

DOWNLOAD_URL = "https://www.imdpune.gov.in/cmpg/Griddata/RF25.php"

YEAR = 2023

BOUNDS = {
    "min_latitude": 28.2,
    "max_latitude": 28.9,
    "min_longitude": 76.7,
    "max_longitude": 77.6,
}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = RAW_DIR / f"imd_rainfall_{YEAR}.nc"
    output_file = OUTPUT_DIR / f"delhi_rainfall_{YEAR}.csv"

    # Avoid accidentally overwriting an existing dataset.
    if raw_file.exists() or output_file.exists():
        raise FileExistsError(
            f"Output already exists for {YEAR}. "
            "Inspect the existing files before retrying."
        )

    print(f"Downloading IMD rainfall data for {YEAR}...")

    response = requests.post(
        DOWNLOAD_URL,
        data={"RF25": str(YEAR)},
        timeout=(15, 180),
    )

    response.raise_for_status()

    # Check the response before saving it as a NetCDF file.
    content_type = response.headers.get("Content-Type", "").lower()

    if "text/html" in content_type:
        raise ValueError(
            "The server returned HTML instead of the rainfall dataset."
        )

    if not response.content.startswith((b"CDF", b"\x89HDF")):
        raise ValueError(
            "The response does not have a recognized NetCDF signature. "
            "It may be an error page or an invalid download."
        )

    raw_file.write_bytes(response.content)

    print(f"Downloaded {len(response.content):,} bytes.")

    try:
        with xr.open_dataset(raw_file) as dataset:

            required = {
                "RAINFALL",
                "TIME",
                "LATITUDE",
                "LONGITUDE",
            }

            missing = required - set(dataset.variables)

            if missing:
                raise ValueError(
                    f"Missing required variables: {sorted(missing)}"
                )

            rainfall = dataset["RAINFALL"]

            if set(rainfall.dims) != {
                "TIME",
                "LATITUDE",
                "LONGITUDE",
            }:
                raise ValueError(
                    f"Unexpected rainfall dimensions: {rainfall.dims}"
                )

            dates = dataset["TIME"].values

            if len(dates) == 0:
                raise ValueError("The dataset contains no dates.")

            first_date = str(dates[0])[:10]
            last_date = str(dates[-1])[:10]

            if first_date[:4] != str(YEAR):
                raise ValueError(
                    f"Unexpected first date: {first_date}"
                )

            if last_date[:4] != str(YEAR):
                raise ValueError(
                    f"Unexpected last date: {last_date}"
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
                raise ValueError(
                    "No grid cells found within the Delhi-region bounds."
                )

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

        frame["date"] = __import__("pandas").to_datetime(
            frame["date"],
            errors="raise",
        )

        if frame["rainfall_mm"].isna().any():
            raise ValueError(
                "Missing rainfall values detected; inspect the source "
                "before proceeding."
            )

        if (frame["rainfall_mm"] < 0).any():
            raise ValueError(
                "Negative rainfall values detected; inspect source "
                "metadata before proceeding."
            )

        if frame.duplicated(
            ["date", "latitude", "longitude"]
        ).any():
            raise ValueError("Duplicate date/grid-cell observations found.")

        frame = frame.sort_values(
            ["date", "latitude", "longitude"]
        ).reset_index(drop=True)

        frame.to_csv(output_file, index=False)

        print("\nHistorical rainfall download succeeded.")
        print(f"Year: {YEAR}")
        print(f"Date range: {frame['date'].min().date()} to "
              f"{frame['date'].max().date()}")
        print(f"Rows: {len(frame):,}")
        print(
            "Unique grid cells: "
            f"{frame[['latitude', 'longitude']].drop_duplicates().shape[0]}"
        )
        print(f"Processed CSV: {output_file}")
        print(f"Raw NetCDF preserved at: {raw_file}")

    except Exception:
        print(
            "\nProcessing failed. The downloaded NetCDF has been "
            "preserved for inspection."
        )
        raise


if __name__ == "__main__":
    main()