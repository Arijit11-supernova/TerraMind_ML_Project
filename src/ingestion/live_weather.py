"""
Live rainfall ingestion (Open-Meteo) aligned to the IMD training grid.

The champion model was trained on IMD 0.25-degree gridded rainfall for 12 Delhi
cells. This module pulls real, current daily rainfall for the SAME 12 cell
centres from Open-Meteo and rebuilds the exact engineered features used in
training (verified against data/processed/delhi_rainfall_features.csv).

Source note: Open-Meteo is a reanalysis/forecast product, not IMD station or
gridded observations, so live inputs carry a source shift. The app says so.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / "data" / "live_cache" / "open_meteo_last_fetch.json"

API_URL = "https://api.open-meteo.com/v1/forecast"
SOURCE_NAME = "Open-Meteo (daily precipitation_sum)"

LATS = [28.25, 28.5, 28.75]
LONS = [76.75, 77.0, 77.25, 77.5]
GRID_CELLS = [(la, lo) for la in LATS for lo in LONS]  # same 12 cells as training

PAST_DAYS = 10  # need as_of-7 .. as_of
IST = timezone(timedelta(hours=5, minutes=30))


class LiveDataError(RuntimeError):
    """Raised when live data cannot be fetched or is incomplete."""


@dataclass
class LiveFetch:
    frame: pd.DataFrame      # columns: date, latitude, longitude, precip_mm
    fetched_at: str          # ISO timestamp (UTC) of the successful API call
    from_cache: bool         # True if the live call failed and the last real fetch was reused
    source: str = SOURCE_NAME


def today_ist():
    return datetime.now(IST).date()


# --------------------------------------------------
# Fetching
# --------------------------------------------------

def _request_open_meteo(retries: int = 3, timeout: int = 20):
    params = {
        "latitude": ",".join(str(la) for la, _ in GRID_CELLS),
        "longitude": ",".join(str(lo) for _, lo in GRID_CELLS),
        "daily": "precipitation_sum",
        "past_days": PAST_DAYS,
        "forecast_days": 2,
        "timezone": "Asia/Kolkata",
    }
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, list) else [payload]
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise LiveDataError(f"Open-Meteo request failed after {retries} attempts: {last_error}")


def _frame_from_payload(payload: list) -> pd.DataFrame:
    if len(payload) != len(GRID_CELLS):
        raise LiveDataError(
            f"Expected {len(GRID_CELLS)} locations, API returned {len(payload)}."
        )
    rows = []
    for (lat, lon), location in zip(GRID_CELLS, payload):
        daily = location.get("daily") or {}
        for day, value in zip(daily.get("time", []), daily.get("precipitation_sum", [])):
            rows.append({"date": day, "latitude": lat, "longitude": lon, "precip_mm": value})
    if not rows:
        raise LiveDataError("API response contained no daily rainfall values.")
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _save_cache(frame: pd.DataFrame, fetched_at: str) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        records = frame.assign(date=frame["date"].dt.strftime("%Y-%m-%d")).to_dict("records")
        CACHE_PATH.write_text(json.dumps({"fetched_at": fetched_at, "records": records}))
    except OSError:
        pass  # cache is best-effort


def _load_cache() -> LiveFetch | None:
    try:
        blob = json.loads(CACHE_PATH.read_text())
        frame = pd.DataFrame(blob["records"])
        frame["date"] = pd.to_datetime(frame["date"])
        return LiveFetch(frame=frame, fetched_at=blob["fetched_at"], from_cache=True)
    except (OSError, ValueError, KeyError):
        return None


def fetch_live_rainfall() -> LiveFetch:
    """Fetch real rainfall now; if the API is down, reuse the last REAL fetch (flagged)."""
    try:
        frame = _frame_from_payload(_request_open_meteo())
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save_cache(frame, fetched_at)
        return LiveFetch(frame=frame, fetched_at=fetched_at, from_cache=False)
    except LiveDataError as exc:
        cached = _load_cache()
        if cached is None:
            raise LiveDataError(f"{exc} No earlier successful fetch is available.") from exc
        return cached


# --------------------------------------------------
# Feature engineering (mirrors training definitions)
# --------------------------------------------------

def build_live_features(frame: pd.DataFrame, as_of=None):
    """
    Rebuild the training features for every cell as of `as_of` (default: the last
    complete day, yesterday in IST). The prediction target is as_of + 1 day.

    Definitions (verified against the training CSV):
      rainfall_mm          rainfall on as_of
      rainfall_lag_k       rainfall on as_of - k
      rainfall_mean_3d     mean of lag_1..lag_3   (excludes as_of)
      rainfall_mean_7d     mean of lag_1..lag_7   (excludes as_of)
      rainfall_max_7d      max  of lag_1..lag_7   (excludes as_of)
      regional_*           mean / max across the 12 cells on as_of
      cells_over_10mm/35mm count of cells with rainfall strictly above 10 / 35 mm
      target_month / target_day_sin / target_day_cos  from as_of + 1 (doy / 365.25)
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp(today_ist() - timedelta(days=1))
    as_of = as_of.normalize()
    target = as_of + timedelta(days=1)

    pivot = frame.pivot(index="date", columns=["latitude", "longitude"], values="precip_mm").sort_index()
    needed = pd.date_range(as_of - timedelta(days=7), as_of)
    absent = [d.strftime("%Y-%m-%d") for d in needed if d not in pivot.index]
    if absent:
        raise LiveDataError(f"Missing days in live data: {', '.join(absent)}")
    window = pivot.loc[needed]
    if window.isna().any().any():
        raise LiveDataError("Live data has missing rainfall values in the 8-day window.")

    today_values = window.loc[as_of]
    regional_mean = float(today_values.mean())
    regional_max = float(today_values.max())
    over10 = int((today_values > 10).sum())
    over35 = int((today_values > 35).sum())

    angle = 2 * np.pi * target.day_of_year / 365.25
    rows = []
    for lat, lon in GRID_CELLS:
        series = window[(lat, lon)]
        lags = [float(series.loc[as_of - timedelta(days=k)]) for k in range(1, 8)]
        rows.append({
            "date": as_of,
            "latitude": lat,
            "longitude": lon,
            "rainfall_mm": float(series.loc[as_of]),
            "rainfall_lag_1": lags[0],
            "rainfall_lag_2": lags[1],
            "rainfall_lag_3": lags[2],
            "rainfall_mean_3d": float(np.mean(lags[:3])),
            "rainfall_mean_7d": float(np.mean(lags[:7])),
            "rainfall_max_7d": float(np.max(lags[:7])),
            "regional_mean_rainfall_mm": regional_mean,
            "regional_max_rainfall_mm": regional_max,
            "cells_over_10mm": over10,
            "cells_over_35mm": over35,
            "target_date": target,
            "target_month": float(target.month),
            "target_day_sin": float(np.sin(angle)),
            "target_day_cos": float(np.cos(angle)),
        })
    features = pd.DataFrame(rows)

    nwp = pivot.loc[target] if target in pivot.index and not pivot.loc[target].isna().all() else None
    meta = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "target_date": target.strftime("%Y-%m-%d"),
        "regional_mean_mm": regional_mean,
        "regional_max_mm": regional_max,
        "nwp_target_regional_mean_mm": None if nwp is None else float(nwp.mean()),
        "nwp_target_regional_max_mm": None if nwp is None else float(nwp.max()),
    }
    return features, meta