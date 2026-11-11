"""
Shared, device-independent store for AI news briefs and their scored outcomes.

Why this exists
---------------
Briefs used to live only in each browser's localStorage (plus an optional
per-account copy). A phone with no Groq/Tavily keys, or one that was not signed
in, therefore never saw the call and showed "neutral" / "No AI brief yet", and two
devices that each ran their own brief could disagree and overwrite each other.

Rules enforced here
-------------------
* One canonical PRE-release call per event. The first valid call wins and can
  never be replaced, so every device shows the same prediction.
* A call is only accepted if it was saved BEFORE the release time.
* Outcome (actual vs consensus) fields are only accepted from the current
  scoring version (SCORING_V) and only when verified, so old mis-scored results
  are ignored and replaced.
* Device-specific fields (pair directions, watchlist scans) are never shared.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .. import db

SCORING_V = 2
BIAS = {"bullish", "bearish", "neutral"}
MAX_BYTES = 60_000

# Never shared: they depend on one device's watchlist or retry state.
LOCAL_ONLY = (
    "pair_directions", "pairs_scanned_at", "market_data",
    "actual_attempts", "actual_last_attempt",
)
OUTCOME_KEYS = (
    "actual_usd_bias", "actual_surprise", "actual_summary", "actual_sources",
    "actual_status", "actual_researched_at", "actual_method", "scoring_v",
)

_ready = False


def init_db() -> None:
    with db.get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shared_briefs (
                event_id TEXT PRIMARY KEY,
                event_utc TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)


def _ensure() -> None:
    global _ready
    if not _ready:
        init_db()
        _ready = True


def _dt(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def has_valid_outcome(rec: dict) -> bool:
    return (
        rec.get("scoring_v") == SCORING_V
        and rec.get("actual_status") == "verified"
        and rec.get("actual_usd_bias") in BIAS
    )


def _clean(payload: dict) -> dict:
    rec = {k: v for k, v in payload.items() if k not in LOCAL_ONLY}
    if not has_valid_outcome(rec):
        for key in OUTCOME_KEYS:
            rec.pop(key, None)
    archive = rec.get("archive")
    if archive is not None:
        try:
            ok = isinstance(archive, dict) and float(archive.get("move_pct")) == float(archive.get("move_pct"))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            rec.pop("archive", None)
    return rec


def validate_prediction(event_id: str, payload: dict) -> dict:
    """Return a cleaned record or raise ValueError."""
    if len(json.dumps(payload)) > MAX_BYTES:
        raise ValueError("Record too large.")
    event = payload.get("event")
    if not isinstance(event, dict) or event.get("id") != event_id:
        raise ValueError("event_id must match payload.event.id")
    release = _dt(event.get("utc"))
    if release is None:
        raise ValueError("Missing or invalid event time.")
    if payload.get("was_pre_release") is not True:
        raise ValueError("Only pre-release calls are shared.")
    if payload.get("usd_bias") not in BIAS:
        raise ValueError("usd_bias must be bullish, bearish or neutral.")
    called = _dt(payload.get("researched_at"))
    if called is None or called >= release:
        raise ValueError("A call must be saved before the release time.")
    rec = _clean(payload)
    rec["event"] = event
    return rec


def merge(existing: Optional[dict], incoming: dict) -> dict:
    """First valid prediction wins; outcome and archive may be added later."""
    if existing is None:
        return incoming
    out = dict(existing)  # the saved prediction is immutable
    if has_valid_outcome(incoming):
        newer = str(incoming.get("actual_researched_at", "")) > str(existing.get("actual_researched_at", ""))
        if not has_valid_outcome(existing) or newer:
            for key in OUTCOME_KEYS:
                if key in incoming:
                    out[key] = incoming[key]
    if not out.get("archive") and incoming.get("archive"):
        out["archive"] = incoming["archive"]
    return out


def put(event_id: str, payload: dict) -> dict:
    incoming = validate_prediction(event_id, payload)
    _ensure()
    release = _dt(incoming["event"]["utc"]).isoformat()
    now = datetime.now(timezone.utc).isoformat()
    with db.get_conn() as conn:
        row = conn.execute("SELECT payload FROM shared_briefs WHERE event_id = ?", (event_id,)).fetchone()
        existing = None
        if row:
            try:
                existing = _clean(json.loads(row["payload"]))
                existing["event"] = json.loads(row["payload"]).get("event", incoming["event"])
            except (KeyError, TypeError, json.JSONDecodeError):
                existing = None
        merged = merge(existing, incoming)
        conn.execute("""
            INSERT INTO shared_briefs (event_id, event_utc, payload, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
              payload = excluded.payload,
              updated_at = excluded.updated_at
        """, (event_id, release, json.dumps(merged), now))
    return merged


def list_recent(days: int = 90) -> list[dict]:
    _ensure()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT payload FROM shared_briefs WHERE event_utc >= ? ORDER BY event_utc DESC",
            (cutoff,),
        ).fetchall()
    out = []
    for row in rows:
        try:
            data = json.loads(row["payload"])
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        event = data.get("event")
        rec = _clean(data)
        rec["event"] = event
        out.append(rec)
    return out
