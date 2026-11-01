"""
Quantify the source shift between the IMD training data and the live Open-Meteo feed.

Pulls Open-Meteo archive rainfall for the same 12 cells over the 2024 monsoon and
compares it with the IMD rainfall stored in data/processed/delhi_rainfall_features.csv.
Run from the repo root:  python scripts/compare_sources.py
Put the printed numbers in the README / deck as an honest data-quality statement.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "delhi_rainfall_features.csv"
URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2024-06-01", "2024-09-30"

LATS = [28.25, 28.5, 28.75]
LONS = [76.75, 77.0, 77.25, 77.5]
CELLS = [(la, lo) for la in LATS for lo in LONS]


def fetch_open_meteo() -> pd.DataFrame:
    params = {
        "latitude": ",".join(str(la) for la, _ in CELLS),
        "longitude": ",".join(str(lo) for _, lo in CELLS),
        "start_date": START, "end_date": END,
        "daily": "precipitation_sum", "timezone": "Asia/Kolkata",
    }
    response = requests.get(URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    payload = payload if isinstance(payload, list) else [payload]
    rows = []
    for (lat, lon), loc in zip(CELLS, payload):
        for day, value in zip(loc["daily"]["time"], loc["daily"]["precipitation_sum"]):
            rows.append({"date": pd.Timestamp(day), "latitude": lat, "longitude": lon, "om_mm": value})
    return pd.DataFrame(rows)


def main() -> None:
    imd = pd.read_csv(FEATURES, usecols=["date", "latitude", "longitude", "rainfall_mm"], parse_dates=["date"])
    imd = imd[(imd["date"] >= START) & (imd["date"] <= END)]
    merged = imd.merge(fetch_open_meteo(), on=["date", "latitude", "longitude"]).dropna()

    corr_cell = merged["rainfall_mm"].corr(merged["om_mm"])
    regional = merged.groupby("date")[["rainfall_mm", "om_mm"]].mean()
    corr_regional = regional["rainfall_mm"].corr(regional["om_mm"])
    peak = merged.groupby("date")[["rainfall_mm", "om_mm"]].max()

    print(f"Period {START} to {END}, {len(merged):,} cell-days")
    print(f"Correlation, per cell-day:        {corr_cell:.3f}")
    print(f"Correlation, regional daily mean: {corr_regional:.3f}")
    print(f"Mean rain IMD / Open-Meteo (mm):  {merged['rainfall_mm'].mean():.2f} / {merged['om_mm'].mean():.2f}")
    print(f"Mean absolute error (mm):         {np.abs(merged['rainfall_mm'] - merged['om_mm']).mean():.2f}")
    print(f"Days with a >=35 mm cell, IMD / Open-Meteo: "
          f"{int((peak['rainfall_mm'] >= 35).sum())} / {int((peak['om_mm'] >= 35).sum())}")


if __name__ == "__main__":
    main()