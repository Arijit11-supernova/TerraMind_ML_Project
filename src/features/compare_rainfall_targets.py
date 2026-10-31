"""Compare candidate next-day rainfall prediction targets."""
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delhi_rainfall_2000_2024.csv"
)

THRESHOLDS = [35.0, 64.5]


def main():
    data = pd.read_csv(
        DATA_FILE,
        parse_dates=["date"],
    )

    data = data.sort_values(
        ["latitude", "longitude", "date"]
    ).copy()

    # Features are associated with date t.
    # Labels represent rainfall at the same grid cell on date t+1.
    data["target_date"] = (
        data.groupby(["latitude", "longitude"])["date"]
        .shift(-1)
    )

    data["next_day_rainfall_mm"] = (
        data.groupby(["latitude", "longitude"])["rainfall_mm"]
        .shift(-1)
    )

    # Remove observations without a next-day target.
    data = data.dropna(
        subset=["target_date", "next_day_rainfall_mm"]
    ).copy()

    # Assign periods by the date being predicted, not the feature date.
    data["target_year"] = data["target_date"].dt.year

    data["period"] = pd.cut(
        data["target_year"],
        bins=[1999, 2018, 2021, 2024],
        labels=[
            "Train: 2000-2018",
            "Validation: 2019-2021",
            "Test: 2022-2024",
        ],
    )

    print("=" * 75)
    print("TERRAMIND ML: CANDIDATE TARGET COMPARISON")
    print("=" * 75)

    print(f"\nTotal usable observations: {len(data):,}")

    for threshold in THRESHOLDS:
        target_name = f"rainfall_ge_{threshold:g}mm"

        data[target_name] = (
            data["next_day_rainfall_mm"] >= threshold
        ).astype(int)

        print("\n" + "-" * 75)
        print(f"TARGET: Next-day rainfall >= {threshold:g} mm")
        print("-" * 75)

        positives = int(data[target_name].sum())
        total = len(data)

        print(f"Positive observations: {positives:,}")
        print(f"Negative observations: {total - positives:,}")
        print(f"Positive rate: {100 * positives / total:.4f}%")

        summary = (
            data.groupby("period", observed=False)[target_name]
            .agg(
                observations="count",
                positive_cases="sum",
            )
        )

        summary["negative_cases"] = (
            summary["observations"] - summary["positive_cases"]
        )

        summary["positive_rate_pct"] = (
            100
            * summary["positive_cases"]
            / summary["observations"].replace(0, pd.NA)
        )

        print("\nChronological split:")
        print(summary.to_string(
            formatters={
                "positive_rate_pct": lambda x: (
                    f"{x:.4f}%" if pd.notna(x) else "N/A"
                )
            }
        ))

    print("\n" + "=" * 75)
    print("TARGET COMPARISON COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()