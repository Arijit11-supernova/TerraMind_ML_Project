"""Analyze rainfall patterns and evaluate a potential ML target."""

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delhi_rainfall_2000_2024.csv"
)

HEAVY_RAINFALL_THRESHOLD = 64.5


def main():
    print("=" * 65)
    print("TERRAMIND ML: RAINFALL TARGET ANALYSIS")
    print("=" * 65)

    # 1. Load the consolidated dataset.
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    required_columns = {
        "date",
        "latitude",
        "longitude",
        "rainfall_mm",
    }

    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Dataset must contain: {sorted(required_columns)}"
        )

    if data[list(required_columns)].isna().any().any():
        raise ValueError("Missing values detected.")

    if not np.isfinite(
        data[
            ["latitude", "longitude", "rainfall_mm"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite numeric values detected.")

    if (data["rainfall_mm"] < 0).any():
        raise ValueError("Negative rainfall detected.")

    data["date"] = pd.to_datetime(data["date"])

    # 2. Basic dataset information.
    print("\n[1] DATASET OVERVIEW")
    print(f"Total observations: {len(data):,}")
    print(f"Start date: {data['date'].min().date()}")
    print(f"End date: {data['date'].max().date()}")
    print(f"Grid cells: {data[['latitude', 'longitude']].drop_duplicates().shape[0]}")

    # 3. Calculate daily rainfall summaries across the region.
    daily = (
        data.groupby("date")["rainfall_mm"]
        .agg(
            mean_rainfall_mm="mean",
            max_grid_rainfall_mm="max",
            wet_grid_cells=lambda values: int((values > 0).sum()),
        )
        .reset_index()
        .sort_values("date")
    )

    print("\n[2] DAILY REGIONAL RAINFALL")
    print(daily[
        [
            "mean_rainfall_mm",
            "max_grid_rainfall_mm",
            "wet_grid_cells",
        ]
    ].describe(
        percentiles=[0.90, 0.95, 0.99]
    ).round(3))

    # 4. Analyze heavy rainfall at individual grid cells.
    data["heavy_rainfall"] = (
        data["rainfall_mm"] >= HEAVY_RAINFALL_THRESHOLD
    ).astype(int)

    positive_count = int(data["heavy_rainfall"].sum())
    total_count = len(data)
    negative_count = total_count - positive_count

    print("\n[3] GRID-CELL HEAVY-RAINFALL ANALYSIS")
    print(f"Threshold: >= {HEAVY_RAINFALL_THRESHOLD} mm/day")
    print(f"Positive observations: {positive_count:,}")
    print(f"Negative observations: {negative_count:,}")
    print(f"Positive percentage: {100 * positive_count / total_count:.4f}%")

    # 5. Analyze regional maximum rainfall.
    regional_events = (
        daily["max_grid_rainfall_mm"]
        >= HEAVY_RAINFALL_THRESHOLD
    )

    print("\n[4] REGIONAL MAXIMUM ANALYSIS")
    print(
        "Days when at least one grid cell reached the threshold: "
        f"{int(regional_events.sum()):,}"
    )
    print(
        "Percentage of days: "
        f"{100 * regional_events.mean():.3f}%"
    )

    # 6. Analyze daily regional mean rainfall.
    regional_mean_events = (
        daily["mean_rainfall_mm"]
        >= HEAVY_RAINFALL_THRESHOLD
    )

    print(
        "Days when regional mean reached the threshold: "
        f"{int(regional_mean_events.sum()):,}"
    )

    # 7. Check whether individual grid cells have enough positive cases.
    events_by_grid = (
        data.groupby(["latitude", "longitude"])["heavy_rainfall"]
        .agg(
            positive_days="sum",
            total_days="count",
        )
        .reset_index()
    )

    events_by_grid["positive_percentage"] = (
        100
        * events_by_grid["positive_days"]
        / events_by_grid["total_days"]
    )

    print("\n[5] HEAVY-RAINFALL EVENTS BY GRID CELL")
    print(
        events_by_grid.sort_values(
            "positive_days",
            ascending=False,
        ).to_string(
            index=False,
            formatters={
                "positive_percentage": "{:.3f}%".format
            },
        )
    )

    # 8. Create a preliminary next-day target.
    # The target is the next day's rainfall at the SAME grid cell.
    data = data.sort_values(
        ["latitude", "longitude", "date"]
    ).copy()

    data["next_day_rainfall_mm"] = (
        data.groupby(
            ["latitude", "longitude"]
        )["rainfall_mm"].shift(-1)
    )

    data["next_day_heavy_rainfall"] = (
        data["next_day_rainfall_mm"]
        >= HEAVY_RAINFALL_THRESHOLD
    ).astype("Int64")

    # The last date for each grid cell has no known next-day target.
    model_data = data.dropna(
        subset=["next_day_rainfall_mm"]
    ).copy()

    # 9. Summarize the actual prediction target.
    target_positive = int(
        (
            model_data["next_day_rainfall_mm"]
            >= HEAVY_RAINFALL_THRESHOLD
        ).sum()
    )

    target_total = len(model_data)

    print("\n[6] NEXT-DAY PREDICTION TARGET")
    print(
        "Task: Predict whether the SAME grid cell reaches "
        f"{HEAVY_RAINFALL_THRESHOLD} mm on the next day."
    )
    print(f"Usable observations: {target_total:,}")
    print(f"Positive targets: {target_positive:,}")
    print(
        f"Positive target rate: "
        f"{100 * target_positive / target_total:.4f}%"
    )

    # 10. Show the number of positive next-day targets by year.
    model_data["year"] = model_data["date"].dt.year

    yearly_target = (
        model_data.groupby("year")["next_day_rainfall_mm"]
        .agg(
            observations="count",
            heavy_rainfall_days=lambda values: int(
                (values >= HEAVY_RAINFALL_THRESHOLD).sum()
            ),
        )
    )

    print("\n[7] NEXT-DAY TARGET COUNTS BY YEAR")
    print(yearly_target.to_string())

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)
    print(
        "\nNo model has been trained yet. Use these results "
        "to decide whether the target is suitable."
    )

if __name__ == "__main__":
    main()