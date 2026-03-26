"""Place search using AMap."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import requests

AMAP_PLACE_URL = "https://restapi.amap.com/v3/place/text"
AMAP_PLACE_AROUND_URL = "https://restapi.amap.com/v3/place/around"
AMAP_GEOCODE_URL = "https://restapi.amap.com/v3/geocode/geo"
MIXED_INDOOR_FLAG = "mixed"


def search_places(city: str, query: str, price_level: str) -> List[Dict[str, Any]]:
    """Search POIs via AMap.

    Workflow:
    1) Text search scoped to city with citylimit=true.
    2) If empty, re-run without city limit.
    3) If still empty, geocode city and use place/around within 5km.
    4) Map AMap POI fields to a compact dict: name/type/address/price_level/indoor/location.
    5) If no usable POIs remain, return a single error entry so the model knows to adjust.
    6) When a price_level is requested (budget/mid/premium), filter to matching results.
    """

    key = os.getenv("AMAP_API_KEY")
    if not key:
        return []

    params_primary = {
        "key": key,
        "keywords": query,
        "city": city,
        "citylimit": "true",
        "offset": 20,
        "page": 1,
        "extensions": "all",
        "output": "json",
    }

    pois = _amap_search(params_primary)
    if not pois:
        # broaden search without city limit as fallback
        params_fallback = {
            **params_primary,
            "citylimit": "false",
            "city": "",
        }
        pois = _amap_search(params_fallback)

    if not pois:
        # as a last resort, search around city center within 5km
        loc = _amap_geocode_city(city, key)
        if loc:
            params_around = {
                "key": key,
                "location": f"{loc[0]},{loc[1]}",
                "keywords": query,
                "radius": 5000,
                "offset": 20,
                "extensions": "all",
                "output": "json",
            }
            pois = _amap_search(params_around, around=True)

    results: List[Dict[str, Any]] = []
    for poi in pois:
        if isinstance(poi, dict) and "error" in poi:
            results.append(poi)
            continue
        indoor_flag = poi.get("indoor_map") or poi.get("business_area")
        name = poi.get("name", "")
        typecode = poi.get("typecode", "")
        address = poi.get("address", "")
        location = poi.get("location", "")
        price = poi.get("biz_ext", {}).get("cost") if isinstance(poi.get("biz_ext"), dict) else None
        price_tag = _price_tag(price_level, price)
        results.append(
            {
                "name": name,
                "type": typecode,
                "address": address,
                "price_level": price_tag,
                "indoor": bool(indoor_flag),
                "location": location,
            }
        )

    if not results:
        return [
            {
                "error": "AMap returned 0 places after all fallbacks",
                "query": query,
                "city": city,
                "price_level": price_level,
            }
        ]

    if price_level != "any":
        results = [r for r in results if r.get("price_level") == price_level or "error" in r]
    return results


def _price_tag(requested: str, cost: Any) -> str:
    """Normalize price level.

    If caller asked for a fixed band (budget/mid/premium), honor it. Otherwise infer from
    numeric cost: <50 budget, <120 mid, else premium; fall back to "any" when unknown.
    """

    if requested in {"budget", "mid", "premium"}:
        return requested
    try:
        value = float(cost)
    except Exception:
        return "any"
    if value < 50:
        return "budget"
    if value < 120:
        return "mid"
    return "premium"


def _amap_search(params: Dict[str, Any], around: bool = False) -> List[Dict[str, Any]]:
    """Call AMap text/around search and normalize responses.

    Returns either a POI list or a single error dict capturing status/info/count so the
    planner can react. Network exceptions are wrapped as error dicts instead of raising.
    """

    url = AMAP_PLACE_AROUND_URL if around else AMAP_PLACE_URL
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict):
            return [{"error": "AMap response not dict"}]
        if payload.get("status") != "1":
            return [
                {
                    "error": "AMap request failed",
                    "info": payload.get("info"),
                    "infocode": payload.get("infocode"),
                }
            ]
        pois = payload.get("pois", []) or []
        if not pois:
            return [
                {
                    "error": "AMap returned 0 pois",
                    "info": payload.get("info"),
                    "count": payload.get("count"),
                }
            ]
        return pois
    except Exception as exc:
        return [{"error": f"AMap exception: {exc}"}]


def _amap_geocode_city(city: str, key: str) -> Tuple[float, float] | None:
    """Geocode a city to (lng, lat).

    Tries AMap geocode with city hint; if empty and city is not already Beijing, retries
    with "Beijing" as a loose fallback. Returns None on any failure or malformed payload.
    """

    params = {"key": key, "address": city, "city": city, "output": "json"}
    try:
        resp = requests.get(AMAP_GEOCODE_URL, params=params, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
        geocodes = (payload or {}).get("geocodes", []) if isinstance(payload, dict) else []
        if not geocodes:
            # fallback: try without city hint (may help for non-Chinese input)
            if city and city.lower() != "beijing":
                return _amap_geocode_city("Beijing", key)
            return None
        loc = geocodes[0].get("location", "")
        if not loc or "," not in loc:
            return None
        lng, lat = loc.split(",")
        return float(lng), float(lat)
    except Exception:
        if city and city.lower() != "beijing":
            return _amap_geocode_city("Beijing", key)
        return None
