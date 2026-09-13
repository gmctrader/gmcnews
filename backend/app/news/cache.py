"""
Minimal TTL cache, backed by a single JSON file on disk.

Why this exists: gemini_news.py used to get called fresh on every single
page load/refresh for every calendar event. On a free-tier key that
burns through the (small) daily quota almost immediately -- see the
429 RESOURCE_EXHAUSTED you hit. This cache means each distinct
event/query only ever triggers one real Gemini call per `ttl_seconds`
window; every request in between is served from disk for free.

Not a database, not multi-process safe in a strict sense (fine for a
single small Render web service) -- just enough to cut call volume by
~95%+ with zero new dependencies and zero cost.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gemini_cache.json"
)


def _load() -> dict:
    if not os.path.exists(_CACHE_FILE):
        return {}
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass  # cache is best-effort; never crash the app over it


def get(key: str) -> Optional[Any]:
    entry = _load().get(key)
    if not entry:
        return None
    if time.time() > entry.get("expires_at", 0):
        return None
    return entry.get("value")


def set(key: str, value: Any, ttl_seconds: float) -> None:
    data = _load()
    data[key] = {"value": value, "expires_at": time.time() + ttl_seconds}
    _save(data)
