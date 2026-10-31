"""Exploratory data analysis for IMD rainfall observations."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "delhi_rainfall_2024.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "eda"

def load_data():
    """Load and perform basic checks on the processed rainfall data."""

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Processed rainfall data not found: {INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
    )

    required_columns = {
        "date",
        "latitude",
        "longitude",
        "rainfall_mm",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("The rainfall dataset is empty.")

    if data["rainfall_mm"].isna().any():
        raise ValueError("Missing rainfall values detected.")

    if (data["rainfall_mm"] < 0).any():
        raise ValueError("Negative rainfall values detected.")

    return data

def create_daily_summary(data):
    """Aggregate grid-cell observations into daily regional summaries."""

    daily = (
        data.groupby("date")["rainfall_mm"]
        .agg(
            regional_mean_mm="mean",
            regional_max_mm="max",
        )
        .reset_index()
        .sort_values("date")
    )

    return daily

def plot_daily_rainfall(daily):
    """Plot daily regional-average and maximum rainfall."""

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        daily["date"],
        daily["regional_mean_mm"],
        label="Mean across 12 grid cells",
    )

    ax.set_title("Daily Regional-Average Rainfall — Delhi Region, 2024")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "daily_mean_rainfall.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        daily["date"],
        daily["regional_max_mm"],
        label="Maximum across 12 grid cells",
    )

    ax.set_title("Daily Maximum Grid-Cell Rainfall — Delhi Region, 2024")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "daily_max_rainfall.png", dpi=150)
    plt.close(fig)


def plot_rainfall_distribution(data):
    """Plot the distribution of grid-cell rainfall observations."""

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        data["rainfall_mm"],
        bins=50,
        edgecolor="black",
    )

    ax.set_title("Distribution of Daily Grid-Cell Rainfall — 2024")
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Number of observations")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rainfall_distribution.png", dpi=150)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()
    daily = create_daily_summary(data)

    plot_daily_rainfall(daily)
    plot_rainfall_distribution(data)

    print("Rainfall EDA completed successfully.")
    print(f"Input rows: {len(data):,}")
    print(f"Daily records: {len(daily):,}")
    print(f"Plots saved to: {OUTPUT_DIR}")

    print("\nTop 10 dates by maximum grid-cell rainfall:")
    print(
        daily.nlargest(10, "regional_max_mm")[
            ["date", "regional_mean_mm", "regional_max_mm"]
        ].to_string(index=False)
    )

if __name__ == "__main__":
    main()