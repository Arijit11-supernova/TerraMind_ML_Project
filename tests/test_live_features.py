"""
Checks that build_live_features() reproduces the training features exactly.

Method: take the real IMD rainfall series from the training CSV, hand it to the
live feature builder as if it were a fetched frame, and compare every engineered
column with what the training pipeline stored for the same date.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion.live_weather import GRID_CELLS, build_live_features

FEATURES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "delhi_rainfall_features.csv"
)

COMPARE_COLUMNS = [
    "rainfall_mm", "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3",
    "rainfall_mean_3d", "rainfall_mean_7d", "rainfall_max_7d",
    "regional_mean_rainfall_mm", "regional_max_rainfall_mm",
    "cells_over_10mm", "cells_over_35mm",
    "target_month", "target_day_sin", "target_day_cos",
]


@pytest.mark.skipif(not FEATURES_PATH.exists(), reason="training features CSV not present")
@pytest.mark.parametrize("as_of", ["2005-03-15", "2013-08-02", "2019-07-25", "2023-07-09", "2024-12-29"])
def test_live_features_match_training(as_of):
    train = pd.read_csv(FEATURES_PATH, parse_dates=["date", "target_date"])

    day = pd.Timestamp(as_of)
    window = train[(train["date"] >= day - pd.Timedelta(days=8)) & (train["date"] <= day)]
    frame = window.rename(columns={"rainfall_mm": "precip_mm"})[["date", "latitude", "longitude", "precip_mm"]]

    live, meta = build_live_features(frame, as_of=day)
    expected = train[train["date"] == day].set_index(["latitude", "longitude"])

    assert len(live) == len(GRID_CELLS) == len(expected)
    for _, row in live.iterrows():
        truth = expected.loc[(row["latitude"], row["longitude"])]
        for column in COMPARE_COLUMNS:
            assert np.isclose(row[column], truth[column], atol=1e-6), (
                f"{column} differs on {as_of} at {row['latitude']},{row['longitude']}: "
                f"{row[column]} vs {truth[column]}"
            )
    assert meta["target_date"] == (day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")