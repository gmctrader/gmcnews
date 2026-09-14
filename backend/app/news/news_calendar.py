"""
Pre-calculated high-impact U.S. economic news calendar.

The engine doesn't "look up" news in real time — it already *knows* the
exact date/time of every major release for the year, hard-coded below
from the official calendars (BLS, Federal Reserve). That's what lets it
plan around news instead of reacting to it:

  - Employment Situation ("NFP")  -- BLS, 8:30am ET, ~first Friday/month
  - Consumer Price Index ("CPI")  -- BLS, 8:30am ET, ~monthly, week 2
  - Producer Price Index ("PPI")  -- BLS, 8:30am ET, ~monthly, week 2-3
  - FOMC rate decision            -- Federal Reserve, 2:00pm ET, 8x/year

All times below were transcribed from bls.gov/schedule and the Fed's
official FOMC calendar and are stored as (year, month, day, hour, minute)
in US Eastern local time, then converted to UTC with correct DST
handling at import time -- so the underlying wall-clock release time
(8:30am ET / 2:00pm ET) is always right even though the UTC offset
shifts twice a year.

Schedules occasionally move (holidays, government shutdowns, agency
catch-up releases). Rather than silently going stale, this module also
reads `data/news_overrides.json` if present -- a small editable file
where you can add, correct, or append events (e.g. next year's
calendar once BLS publishes it, usually every August) without touching
this code. Overrides are matched by (category, date) and replace the
hard-coded entry; anything with a new date is just added.

Nothing here calls out to the internet — MT5-side automation should
never depend on a live connection to a news API to know what NFP is.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

# How "big" each category is by default, used for default blackout sizing
# and dashboard styling. FOMC > NFP > CPI > PPI in typical realized vol,
# though CPI has overtaken NFP in some regimes -- tune to taste.
IMPACT = {
    "FOMC": "critical",
    "NFP": "high",
    "CPI": "high",
    "PPI": "medium",
}

CATEGORY_LABEL = {
    "FOMC": "FOMC Rate Decision",
    "NFP": "Employment Situation (NFP)",
    "CPI": "Consumer Price Index (CPI)",
    "PPI": "Producer Price Index (PPI)",
}


def _et(y: int, m: int, d: int, hh: int, mm: int) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


@dataclass(frozen=True)
class NewsEvent:
    category: str          # "FOMC" | "NFP" | "CPI" | "PPI"
    dt_utc: datetime
    label: str = ""
    impact: str = "high"
    source: str = "official"   # "official" | "override" | "estimated"

    @property
    def id(self) -> str:
        return f"{self.category}-{self.dt_utc.date().isoformat()}"

    def to_dict(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(UTC)
        seconds = (self.dt_utc - now).total_seconds()
        return {
            "id": self.id,
            "category": self.category,
            "label": self.label or CATEGORY_LABEL.get(self.category, self.category),
            "impact": self.impact,
            "source": self.source,
            "utc": self.dt_utc.isoformat(),
            "local_et": self.dt_utc.astimezone(ET).strftime("%Y-%m-%d %H:%M ET"),
            # Auto-detects whatever timezone this machine is actually set
            # to (no hardcoded region) -- astimezone() with no argument
            # asks the OS for its local zone, so this is correct whether
            # the engine is running from South Africa, the US, or anywhere
            # else, with zero config.
            "local": self.dt_utc.astimezone().strftime("%Y-%m-%d %H:%M %Z"),
            "seconds_until": round(seconds),
            "is_past": seconds < 0,
        }


# ---------------------------------------------------------------------------
# Official 2026 calendar, transcribed from bls.gov/schedule/2026 and the
# Federal Reserve's published FOMC calendar. Release time is always
# 8:30am ET for BLS prints; FOMC decisions land at 2:00pm ET on the
# second day of the two-day meeting.
# ---------------------------------------------------------------------------

_RAW_2026 = [
    # -- FOMC rate decisions (2:00pm ET, day 2 of the meeting) --
    ("FOMC", 2026, 1, 28, 14, 0),
    ("FOMC", 2026, 3, 18, 14, 0),
    ("FOMC", 2026, 4, 29, 14, 0),
    ("FOMC", 2026, 6, 17, 14, 0),
    ("FOMC", 2026, 7, 29, 14, 0),
    ("FOMC", 2026, 9, 16, 14, 0),
    ("FOMC", 2026, 10, 28, 14, 0),
    ("FOMC", 2026, 12, 9, 14, 0),

    # -- Employment Situation / Nonfarm Payrolls (8:30am ET) --
    ("NFP", 2026, 1, 9, 8, 30),
    ("NFP", 2026, 2, 11, 8, 30),
    ("NFP", 2026, 3, 6, 8, 30),
    ("NFP", 2026, 4, 3, 8, 30),
    ("NFP", 2026, 5, 8, 8, 30),
    ("NFP", 2026, 6, 5, 8, 30),
    ("NFP", 2026, 7, 2, 8, 30),
    ("NFP", 2026, 8, 7, 8, 30),
    ("NFP", 2026, 9, 4, 8, 30),
    ("NFP", 2026, 10, 2, 8, 30),
    ("NFP", 2026, 11, 6, 8, 30),
    ("NFP", 2026, 12, 4, 8, 30),

    # -- Consumer Price Index (8:30am ET) --
    ("CPI", 2026, 1, 13, 8, 30),
    ("CPI", 2026, 2, 13, 8, 30),
    ("CPI", 2026, 3, 11, 8, 30),
    ("CPI", 2026, 4, 10, 8, 30),
    ("CPI", 2026, 5, 12, 8, 30),
    ("CPI", 2026, 6, 10, 8, 30),
    ("CPI", 2026, 7, 14, 8, 30),
    ("CPI", 2026, 8, 12, 8, 30),
    ("CPI", 2026, 9, 11, 8, 30),
    ("CPI", 2026, 10, 14, 8, 30),
    ("CPI", 2026, 11, 10, 8, 30),
    ("CPI", 2026, 12, 10, 8, 30),

    # -- Producer Price Index (8:30am ET) --
    # Jan has two prints (Nov-2025 catch-up + Dec-2025 on schedule).
    ("PPI", 2026, 1, 14, 8, 30),
    ("PPI", 2026, 1, 30, 8, 30),
    ("PPI", 2026, 2, 27, 8, 30),
    ("PPI", 2026, 3, 18, 8, 30),
    ("PPI", 2026, 4, 14, 8, 30),
    ("PPI", 2026, 5, 13, 8, 30),
    ("PPI", 2026, 6, 11, 8, 30),
    ("PPI", 2026, 7, 15, 8, 30),
    ("PPI", 2026, 8, 13, 8, 30),
    ("PPI", 2026, 9, 10, 8, 30),
    ("PPI", 2026, 10, 15, 8, 30),
    ("PPI", 2026, 11, 13, 8, 30),
    ("PPI", 2026, 12, 15, 8, 30),
]

_HARDCODED: list[NewsEvent] = [
    NewsEvent(
        category=cat,
        dt_utc=_et(y, m, d, hh, mm).astimezone(UTC),
        impact=IMPACT.get(cat, "medium"),
        source="official",
    )
    for (cat, y, m, d, hh, mm) in _RAW_2026
]

_OVERRIDES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "news_overrides.json"
)


def _load_overrides(path: str = _OVERRIDES_FILE) -> list[NewsEvent]:
    """
    Optional data/news_overrides.json, format:
        [{"category": "CPI", "date": "2027-01-13", "time_et": "08:30",
          "label": "CPI (Jan)", "impact": "high"}, ...]
    A date already present for that category in the hard-coded list is
    replaced (schedule correction); a new one is appended (e.g. next
    year's calendar, or a one-off event you want the engine to respect
    like a surprise Fed press conference).
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    events = []
    for row in rows:
        try:
            cat = str(row["category"]).upper()
            y, m, d = (int(x) for x in row["date"].split("-"))
            hh, mm = (int(x) for x in row.get("time_et", "08:30").split(":"))
            events.append(NewsEvent(
                category=cat,
                dt_utc=_et(y, m, d, hh, mm).astimezone(UTC),
                label=row.get("label", ""),
                impact=row.get("impact", IMPACT.get(cat, "medium")),
                source="override",
            ))
        except (KeyError, ValueError, AttributeError):
            continue
    return events


def get_events(include_overrides: bool = True) -> list[NewsEvent]:
    """Full merged, sorted calendar: hard-coded events, with any
    data/news_overrides.json entries replacing same-day/category
    matches or appended if new."""
    events = {e.id: e for e in _HARDCODED}
    if include_overrides:
        for e in _load_overrides():
            events[e.id] = e
    return sorted(events.values(), key=lambda e: e.dt_utc)


def upcoming(now: datetime | None = None, within_hours: float | None = None,
             categories: tuple[str, ...] | None = None) -> list[NewsEvent]:
    now = now or datetime.now(UTC)
    events = [e for e in get_events() if e.dt_utc >= now]
    if categories:
        events = [e for e in events if e.category in categories]
    if within_hours is not None:
        cutoff = now + timedelta(hours=within_hours)
        events = [e for e in events if e.dt_utc <= cutoff]
    return events


def today(now: datetime | None = None) -> list[NewsEvent]:
    now = now or datetime.now(UTC)
    return [e for e in get_events() if e.dt_utc.date() == now.date()]


def next_event(now: datetime | None = None,
                categories: tuple[str, ...] | None = None) -> NewsEvent | None:
    ups = upcoming(now, categories=categories)
    return ups[0] if ups else None


@dataclass
class BlackoutStatus:
    active: bool
    event: NewsEvent | None
    phase: str              # "pre" | "post" | "none"
    seconds_remaining: float = 0.0

    def to_dict(self) -> dict:
        return {
            "active": self.active,
            "phase": self.phase,
            "seconds_remaining": round(self.seconds_remaining),
            "event": self.event.to_dict() if self.event else None,
        }


def blackout_status(now: datetime | None = None, pre_minutes: float = 15.0,
                     post_minutes: float = 30.0,
                     categories: tuple[str, ...] = ("FOMC", "NFP", "CPI", "PPI")) -> BlackoutStatus:
    """
    Is `now` inside the no-new-signals window around a watched release?

    pre_minutes:  stop opening fresh setups this long before the print
                  (spreads widen, price often drifts/positions ahead of
                  the number -- not a clean ICT read).
    post_minutes: keep the window open this long after the print too --
                  the initial spike is usually the stop-hunt, not the
                  trade; better setups tend to resolve a few minutes in.
    """
    now = now or datetime.now(UTC)
    for e in get_events():
        if e.category not in categories:
            continue
        pre_start = e.dt_utc - timedelta(minutes=pre_minutes)
        post_end = e.dt_utc + timedelta(minutes=post_minutes)
        if pre_start <= now <= post_end:
            phase = "pre" if now < e.dt_utc else "post"
            remaining = (post_end - now).total_seconds()
            return BlackoutStatus(active=True, event=e, phase=phase, seconds_remaining=remaining)
    return BlackoutStatus(active=False, event=None, phase="none")


def is_post_news_window(now: datetime | None = None, post_minutes: float = 30.0,
                         categories: tuple[str, ...] = ("FOMC", "NFP", "CPI", "PPI")) -> NewsEvent | None:
    """The narrower, *tradeable* window right after a release -- the one
    ICT setups actually want (liquidity was just swept by the number
    itself; a clean CHoCH back the other way is often the highest-
    quality sweep of the whole session). Returns the event if `now` is
    in [release, release + post_minutes], else None."""
    now = now or datetime.now(UTC)
    for e in get_events():
        if e.category not in categories:
            continue
        if e.dt_utc <= now <= e.dt_utc + timedelta(minutes=post_minutes):
            return e
    return None
