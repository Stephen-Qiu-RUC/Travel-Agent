"""Routing via AMap."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import requests

AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
AMAP_DRIVING_URL = "https://restapi.amap.com/v3/direction/driving"
AMAP_WALKING_URL = "https://restapi.amap.com/v3/direction/walking"
AMAP_TRANSIT_URL = "https://restapi.amap.com/v3/direction/transit/integrated"


def calculate_route(origin: str, destination: str, mode: str) -> Dict[str, Any]:
    """Compute route duration via AMap with guardrails.

    Steps:
    1) Geocode origin/destination via AMap geocode.
    2) Request route in requested mode.
    3) If duration > 120 minutes and mode is not drive, auto-try driving; keep faster if within limit.
    4) If still > 120 minutes, return an error advising to pick nearer POIs.
    Returns either a {origin,destination,mode,duration_minutes,note} dict or an error dict.
    """

    key = os.getenv("AMAP_API_KEY")
    if not key:
        return {"error": "AMAP_API_KEY missing"}

    origin_loc = _amap_geocode(origin, key)
    dest_loc = _amap_geocode(destination, key)
    if not origin_loc or not dest_loc:
        return {"error": "Geocoding failed"}

    origin_str = f"{origin_loc[0]},{origin_loc[1]}"
    dest_str = f"{dest_loc[0]},{dest_loc[1]}"

    primary = _fetch_route(mode, origin_str, dest_str, key)
    if primary.get("error"):
        return primary

    duration_minutes = primary["duration_minutes"]

    if duration_minutes > 120 and mode != "drive":
        # Try a faster mode automatically.
        fallback = _fetch_route("drive", origin_str, dest_str, key)
        if not fallback.get("error") and fallback["duration_minutes"] <= 120:
            fallback["note"] = "auto-switched to drive for speed"
            fallback["origin"] = origin
            fallback["destination"] = destination
            return fallback
        if not fallback.get("error"):
            duration_minutes = fallback["duration_minutes"]

    if duration_minutes > 120:
        return {
            "error": "Route too long",
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "duration_minutes": duration_minutes,
            "note": "duration exceeds 120 minutes; pick nearer POIs",
        }

    return {
        "origin": origin,
        "destination": destination,
        "mode": primary.get("mode", mode),
        "duration_minutes": duration_minutes,
        "note": primary.get("note", "data from AMap"),
    }


def _fetch_route(mode: str, origin_str: str, dest_str: str, key: str) -> Dict[str, Any]:
    """Low-level route fetcher.

    Calls the appropriate AMap direction endpoint by mode, parses the first path/transit, and
    returns duration_minutes plus a note. On HTTP/parse issues, returns an error dict instead
    of raising to keep the agent loop resilient.
    """

    try:
        if mode == "drive":
            params = {"key": key, "origin": origin_str, "destination": dest_str}
            resp = requests.get(AMAP_DRIVING_URL, params=params, timeout=10)
        elif mode == "walk":
            params = {"key": key, "origin": origin_str, "destination": dest_str}
            resp = requests.get(AMAP_WALKING_URL, params=params, timeout=10)
        else:
            params = {
                "key": key,
                "origin": origin_str,
                "destination": dest_str,
                "city1": "",
                "city2": "",
            }
            resp = requests.get(AMAP_TRANSIT_URL, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Route API failed: {exc}"}

    paths: list[dict] = []
    if mode in {"drive", "walk"}:
        paths = (payload or {}).get("route", {}).get("paths", []) if isinstance(payload, dict) else []
    elif mode == "transit":
        paths = (payload or {}).get("route", {}).get("transits", []) if isinstance(payload, dict) else []

    if not paths:
        return {"error": "No route found"}

    best = paths[0]
    try:
        duration_seconds = int(best.get("duration", 0))
    except Exception:
        duration_seconds = 0
    duration_minutes = max(1, duration_seconds // 60)

    return {
        "mode": mode,
        "duration_minutes": duration_minutes,
        "note": "data from AMap",
    }


def _amap_geocode(address: str, key: str) -> Tuple[float, float] | None:
    """Geocode any address to (lng, lat) via AMap.

    Returns None on missing geocodes or malformed location strings to let callers decide
    how to handle failures without throwing.
    """

    params = {"key": key, "address": address}
    try:
        resp = requests.get(AMAP_GEOCODE_URL, params=params, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
        geocodes = (payload or {}).get("geocodes", []) if isinstance(payload, dict) else []
        if not geocodes:
            return None
        loc = geocodes[0].get("location", "")
        if not loc:
            return None
        lng, lat = loc.split(",")
        return float(lng), float(lat)
    except Exception:
        return None
