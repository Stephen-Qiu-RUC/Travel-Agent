"""Weather tool using OpenWeatherMap."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
import time
from typing import Any, Dict

import requests


OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OWM_GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/direct"


def get_weather(city: str, date: str) -> Dict[str, Any]:
    """Fetch weather stats via OWM 5-day forecast with clamping and retries.

    Steps:
    1) Parse target date; clamp to [today, today+4] to stay within OWM window.
    2) Geocode city; if fails and city is 北京, retry with "Beijing".
    3) Call forecast API (with a small retry helper) and slice entries matching the target date.
    4) Derive min/max temp and precipitation probability (pop), boosting pop if rain volume exists.
    5) Map precipitation probability to crowd_risk. If no segments are found, return an error dict.
    Returns a compact dict with city/date/precipitation_probability/temperature_range_c/crowd_risk.
    """

    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return {"error": "OPENWEATHERMAP_API_KEY missing"}

    try:
        target_date = datetime.fromisoformat(date).date()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Invalid date: {exc}"}

    today = datetime.utcnow().date()
    # Clamp to OWM 5-day window to avoid 404/empty data
    if target_date < today:
        target_date = today
    if target_date > today + timedelta(days=4):
        target_date = today + timedelta(days=4)
    adjusted_date = target_date.isoformat()

    loc = _owm_geocode_city(city, api_key)
    if not loc and city and city.lower() == "北京":
        # Fallback to English name to improve geocoding resilience
        loc = _owm_geocode_city("Beijing", api_key)
    if not loc:
        return {"error": "Geocoding failed for city"}

    params = {"lat": loc[1], "lon": loc[0], "units": "metric", "appid": api_key}

    payload, err = _get_json_with_retry(OWM_FORECAST_URL, params=params, timeout=10)
    if err or payload is None:
        return {"error": f"Weather API failed: {err}"}

    forecasts = payload.get("list", []) if isinstance(payload, dict) else []
    daily_segments = []
    for entry in forecasts:
        dt_txt = entry.get("dt_txt")
        if not dt_txt:
            continue
        try:
            segment_date = datetime.fromisoformat(dt_txt).date()
        except Exception:
            continue
        if segment_date != target_date:
            continue
        main = entry.get("main", {}) or {}
        pop = entry.get("pop", 0) or 0
        rain_vol = (entry.get("rain", {}) or {}).get("3h", 0) or 0
        pop = max(pop, 1.0 if rain_vol > 0 else pop)
        daily_segments.append(
            {
                "temp_min": main.get("temp_min"),
                "temp_max": main.get("temp_max"),
                "pop": pop,
            }
        )

    if not daily_segments:
        return {"error": "No forecast data for the requested date"}

    min_temp = min(s.get("temp_min") for s in daily_segments if s.get("temp_min") is not None)
    max_temp = max(s.get("temp_max") for s in daily_segments if s.get("temp_max") is not None)
    pop_avg = sum(s.get("pop", 0) for s in daily_segments) / max(len(daily_segments), 1)
    precipitation_probability = round(pop_avg * 100)
    if precipitation_probability > 60:
        crowd_risk = "high"
    elif precipitation_probability > 35:
        crowd_risk = "medium"
    else:
        crowd_risk = "low"

    return {
        "city": city,
        "date": adjusted_date,
        "precipitation_probability": precipitation_probability,
        "temperature_range_c": [min_temp, max_temp],
        "crowd_risk": crowd_risk,
        "note": "data from OpenWeatherMap 5-day forecast (date clamped to available window)",
    }


def _owm_geocode_city(city: str, api_key: str) -> tuple[float, float] | None:
    """Geocode a city to (lon, lat) for OWM.

    Uses the direct geocode endpoint with limit=1, wraps network/parse failures by returning None.
    """

    params = {"q": city, "limit": 1, "appid": api_key}
    payload, err = _get_json_with_retry(OWM_GEOCODE_URL, params=params, timeout=8)
    if err or payload is None:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    entry = payload[0]
    try:
        return float(entry.get("lon")), float(entry.get("lat"))
    except Exception:
        return None


def _get_json_with_retry(url: str, params: Dict[str, Any], timeout: int, attempts: int = 2):
    """HTTP GET with a small retry.

    Attempts the request up to `attempts` times with a short sleep, returning (json, None) on
    success or (None, last_err) on failure to keep callers simple and resilient.
    """

    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json(), None
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt + 1 < attempts:
                time.sleep(0.8)
    return None, last_err
