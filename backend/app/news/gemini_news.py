"""
Gemini-written narrative text, WITHOUT Google Search grounding.

Why no grounding: grounded ("Search the web for X") requests are billed
and rate-limited separately from plain generation, and the free-tier
grounding quota is small enough that a live app polling it runs out
almost immediately (this is the 429 RESOURCE_EXHAUSTED you'll see if
you re-enable `tools=[types.Tool(google_search=...)]` below without a
billed project). Plain (ungrounded) generate_content calls are NOT
subject to that quota and work fine on a free key.

The tradeoff: without grounding, Gemini cannot reliably tell you
TODAY's actual forecast/previous/actual numbers for CPI or NFP -- it
would have to guess, which is exactly the kind of hallucination you
don't want feeding a trading app. So this module intentionally does
NOT ask Gemini to invent those numbers. Instead:

  - enrich_event() only fills in real forecast/previous/actual figures
    if YOU pass them in (e.g. from a real economic-calendar data
    provider you wire in later -- Finnhub/TradingEconomics/etc. both
    have free tiers for this). Gemini is then just used, ungrounded,
    to phrase a one-sentence USD read from numbers you already have --
    a task it's fine at without search.
  - dollar_sentiment_summary() still returns a paragraph, but it's
    explicitly labeled as general-knowledge commentary rather than a
    live-news read, since that's what it honestly is without grounding.

Both calls are cached (see cache.py) so the SAME event/query is only
ever sent to Gemini once per TTL window, instead of on every page
load -- this alone cuts call volume by ~95%+ regardless of quota tier.

Requires: GEMINI_API_KEY environment variable. Falls back to a clearly
-labeled stub response if the key is missing, so the rest of the app
still runs (e.g. in local dev without the key configured).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

from . import cache

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

# How long a cached result is served before we call Gemini again.
# Calendar events rarely need re-phrasing more than a few times a day;
# the sentiment summary can safely refresh less often than that too.
_ENRICH_TTL_SECONDS = 6 * 60 * 60       # 6 hours
_SENTIMENT_TTL_SECONDS = 2 * 60 * 60    # 2 hours

_PHRASING_SYSTEM_PROMPT = """You are a forex fundamentals research assistant.
You will be given an economic release name, date, and (if known) its
forecast/previous/actual values. Do NOT invent any numbers that are not
given to you. Using only what's given, respond ONLY with minified JSON,
no prose, no markdown fences, in this exact shape:
{"surprise": "beat" | "miss" | "inline" | "pending",
 "usd_read": "one sentence, plain language, on what this means for USD
 strength -- or if no actual/forecast values were given, one sentence
 noting the print is still pending"}"""


@dataclass
class NewsEnrichment:
    forecast: Optional[str]
    previous: Optional[str]
    actual: Optional[str]
    surprise: str
    usd_read: str
    stubbed: bool = False

    def to_dict(self) -> dict:
        return {
            "forecast": self.forecast, "previous": self.previous,
            "actual": self.actual, "surprise": self.surprise,
            "usd_read": self.usd_read, "stubbed": self.stubbed,
        }


def _stub(event_label: str, reason: str) -> NewsEnrichment:
    return NewsEnrichment(
        forecast=None, previous=None, actual=None, surprise="pending",
        usd_read=f"{reason} -- showing placeholder for {event_label}.",
        stubbed=True,
    )


def _client(api_key_override: Optional[str] = None):
    api_key = (api_key_override or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not api_key:
        return None
    from google import genai
    return genai.Client(api_key=api_key)


def enrich_event(
    event_label: str,
    event_date_iso: str,
    forecast: Optional[str] = None,
    previous: Optional[str] = None,
    actual: Optional[str] = None,
) -> NewsEnrichment:
    """
    Phrases a one-sentence USD read for one calendar event.

    forecast/previous/actual: pass these in if you have them from a
    real data source (a free economic-calendar API, or your own
    numbers). If you don't have them yet, this returns a clearly
    -labeled "pending" result WITHOUT calling Gemini at all -- there's
    nothing useful for it to phrase yet, so we save the API call.
    """
    if forecast is None and previous is None and actual is None:
        return _stub(event_label, "No forecast/actual data source configured yet")

    cache_key = f"enrich:{event_label}:{event_date_iso}:{forecast}:{previous}:{actual}"
    cached = cache.get(cache_key)
    if cached is not None:
        return NewsEnrichment(**cached)

    client = _client()
    if client is None:
        return _stub(event_label, "GEMINI_API_KEY not set")

    try:
        from google.genai import types

        prompt = (
            f"Event: {event_label}\nDate: {event_date_iso}\n"
            f"Forecast: {forecast}\nPrevious: {previous}\nActual: {actual}"
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_PHRASING_SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        text = (response.text or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)
        result = NewsEnrichment(
            forecast=forecast, previous=previous, actual=actual,
            surprise=data.get("surprise", "pending"),
            usd_read=data.get("usd_read", ""), stubbed=False,
        )
        cache.set(cache_key, result.to_dict(), _ENRICH_TTL_SECONDS)
        return result
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the app
        return NewsEnrichment(
            forecast=forecast, previous=previous, actual=actual, surprise="pending",
            usd_read=f"Gemini phrasing failed ({exc.__class__.__name__}); showing raw values only.",
            stubbed=True,
        )


def dollar_sentiment_summary(dxy_bias: str, confidence: str, api_key_override: Optional[str] = None) -> str:
    """
    A short Gemini-written paragraph on DXY's structural bias.

    IMPORTANT: this is ungrounded (no live search), so it's general
    macro commentary informed by Gemini's training data -- not a live
    news read. Said explicitly in the output so nobody trades on it
    thinking it's today's headlines. Falls back to a templated
    sentence with no key, same as before.
    """
    api_key = (api_key_override or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not api_key:
        return (f"DXY structure reads {dxy_bias} with {confidence} confidence. "
                f"(Add a Gemini API key in dashboard settings for AI commentary.)")

    key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    cache_key = f"sentiment:{key_fingerprint}:{dxy_bias}:{confidence}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = (
            f"The US Dollar Index's own price structure currently reads "
            f"'{dxy_bias}' with '{confidence}' confidence. Using your general "
            f"knowledge of how Fed policy expectations, yields, and risk "
            f"sentiment typically interact with dollar structure, write ONE "
            f"short paragraph (max 3 sentences) on plausible drivers behind "
            f"a read like this and whether such structure typically holds or "
            f"reverses. Do not claim this reflects today's actual news -- "
            f"speak generally. Plain language, no headers, no markdown."
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3),
        )
        text = (response.text or "").strip()
        result = text + "\n\n(General AI commentary, not a live-news read.)"
        cache.set(cache_key, result, _SENTIMENT_TTL_SECONDS)
        return result
    except Exception as exc:  # noqa: BLE001
        return (f"DXY structure reads {dxy_bias} with {confidence} confidence. "
                f"(AI commentary unavailable: {exc.__class__.__name__}.)")
