"""Build leakage-safe features for next-day rainfall prediction."""

from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delhi_rainfall_2000_2024.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delhi_rainfall_features.csv"
)

RAINFALL_THRESHOLD_MM = 35.0


def main():
    print("=" * 75)
    print("TERRAMIND ML: RAINFALL FEATURE ENGINEERING")
    print("=" * 75)

    # 1. Load the authentic historical rainfall dataset.
    data = pd.read_csv(DATA_FILE, parse_dates=["date"])

    required_columns = {
        "date",
        "latitude",
        "longitude",
        "rainfall_mm",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    data = data.sort_values(
        ["latitude", "longitude", "date"]
    ).reset_index(drop=True)

    print(f"\nInput observations: {len(data):,}")
    print(f"Input date range: {data['date'].min().date()} "
          f"to {data['date'].max().date()}")

    # 2. Check for duplicate grid-cell/date observations.
    duplicate_count = data.duplicated(
        subset=["latitude", "longitude", "date"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate grid-cell/date observations."
        )

    # 3. Construct features separately for each grid cell.
    grouped = data.groupby(
        ["latitude", "longitude"],
        sort=False,
    )["rainfall_mm"]

    # Rainfall observed on previous days.
    data["rainfall_lag_1"] = grouped.shift(1)
    data["rainfall_lag_2"] = grouped.shift(2)
    data["rainfall_lag_3"] = grouped.shift(3)

    # Rolling features use only rainfall from preceding days.
    # shift(1) prevents today's rainfall from entering these windows.
    data["rainfall_mean_3d"] = grouped.transform(
        lambda series: series.shift(1).rolling(3).mean()
    )

    data["rainfall_mean_7d"] = grouped.transform(
        lambda series: series.shift(1).rolling(7).mean()
    )

    data["rainfall_max_7d"] = grouped.transform(
        lambda series: series.shift(1).rolling(7).max()
    )

    # Regional rainfall statistics for each input date.
    # These use only observations from the current date.
    regional = data.groupby("date")["rainfall_mm"].agg(
        regional_mean_rainfall_mm="mean",
        regional_max_rainfall_mm="max",
    ).reset_index()

    data = data.merge(
        regional,
        on="date",
        how="left",
        validate="many_to_one",
    )

    # Number of grid cells exceeding selected rainfall thresholds.
    regional_counts = (
        data.assign(
            cells_over_10mm=(data["rainfall_mm"] >= 10).astype(int),
            cells_over_35mm=(data["rainfall_mm"] >= 35).astype(int),
        )
        .groupby("date")[
            ["cells_over_10mm", "cells_over_35mm"]
        ]
        .sum()
        .reset_index()
    )

    data = data.merge(
        regional_counts,
        on="date",
        how="left",
        validate="many_to_one",
    )

    # 4. Create the next-day target.
    data["target_date"] = data.groupby(
        ["latitude", "longitude"],
        sort=False,
    )["date"].shift(-1)

    data["next_day_rainfall_mm"] = grouped.shift(-1)

    data["target_heavy_rainfall"] = (
        data["next_day_rainfall_mm"] >= RAINFALL_THRESHOLD_MM
    ).astype("int8")

    # 5. Calendar features refer to the date being predicted.
    # The target date is known in advance, so its calendar month
    # can legitimately be used as a prediction feature.
    data["target_month"] = data["target_date"].dt.month
    # Cyclical seasonal features for the date being predicted.
    day_of_year = data["target_date"].dt.day_of_year
    data["target_day_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    data["target_day_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )
    # 6. Define chronological splits using the target date.
    target_year = data["target_date"].dt.year

    data["period"] = "Excluded"

    data.loc[
        target_year.between(2000, 2018),
        "period",
    ] = "train"

    data.loc[
        target_year.between(2019, 2021),
        "period",
    ] = "validation"

    data.loc[
        target_year.between(2022, 2024),
        "period",
    ] = "test"

    # 7. Remove rows that do not have the full historical feature
    # window or a valid next-day target.
    feature_columns = [
        "rainfall_lag_1",
        "rainfall_lag_2",
        "rainfall_lag_3",
        "rainfall_mean_3d",
        "rainfall_mean_7d",
        "rainfall_max_7d",
        "latitude",
        "longitude",
        "target_month",
        "target_day_sin",
        "target_day_cos",
        "regional_mean_rainfall_mm",
        "regional_max_rainfall_mm",
        "cells_over_10mm",
        "cells_over_35mm",
    ]

    required_for_model = feature_columns + [
        "target_date",
        "next_day_rainfall_mm",
        "target_heavy_rainfall",
    ]

    data = data.dropna(
        subset=required_for_model
    ).copy()

    # Keep only rows belonging to the explicitly defined periods.
    data = data[data["period"] != "Excluded"].copy()

    # 8. Save the engineered dataset.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # 9. Print validation information.
    print("\nFeature engineering completed successfully.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Usable observations: {len(data):,}")
    print(f"Unique grid cells: {data.groupby(['latitude', 'longitude']).ngroups}")

    print("\nTarget definition:")
    print(
        "1 = next-day rainfall >= "
        f"{RAINFALL_THRESHOLD_MM} mm"
    )
    print("0 = next-day rainfall < "
          f"{RAINFALL_THRESHOLD_MM} mm")

    print("\nChronological split:")
    for period in ["train", "validation", "test"]:
        subset = data[data["period"] == period]

        positives = int(
            subset["target_heavy_rainfall"].sum()
        )

        total = len(subset)

        positive_rate = (
            100 * positives / total
            if total
            else 0.0
        )

        print(
            f"{period.capitalize():12} "
            f"observations={total:7,}  "
            f"positive_cases={positives:4,}  "
            f"positive_rate={positive_rate:.4f}%"
        )

    print("\nFeatures:")
    for column in feature_columns:
        print(f"- {column}")

    # 10. Basic leakage checks.
    if not (
        data["target_date"] > data["date"]
    ).all():
        raise ValueError(
            "Leakage check failed: a target date is not after its input date."
        )

    if data[feature_columns].isna().any().any():
        raise ValueError(
            "Leakage check failed: model features contain missing values."
        )

    print("\nLeakage checks passed.")
    print("=" * 75)


if __name__ == "__main__":
    main()