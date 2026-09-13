"""
Live news content via Gemini (with Google Search grounding).

news_calendar.py already knows WHEN every high-impact release happens
(hard-coded, no internet). This module fills in WHAT the market
expects and WHAT actually happened, plus a plain-language read on
dollar sentiment -- using Gemini's search grounding instead of a
separate news-data API/key.

Model: gemini-3.5-flash-lite -- cheapest current Gemini model, plenty
for "summarize forecast/actual/sentiment" style tasks. Swap
GEMINI_MODEL if you want more depth (e.g. gemini-3.5-flash).

Requires: GEMINI_API_KEY environment variable. Falls back to a clearly
-labeled stub response if the key is missing, so the rest of the app
still runs (e.g. in local dev without the key configured).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

_SYSTEM_PROMPT = """You are a forex fundamentals research assistant.
Given an economic release name and date, use search to find:
- forecast (consensus estimate)
- previous value
- actual value (if the release has already happened)
- a one-sentence plain-language read on what it means for USD strength
Respond ONLY with minified JSON, no prose, no markdown fences, in this
exact shape:
{"forecast": "...", "previous": "...", "actual": "... or null",
 "surprise": "beat" | "miss" | "inline" | "pending",
 "usd_read": "one sentence, plain language"}
If you cannot find reliable data, use null for missing fields and
"pending" for surprise."""


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


def _stub(event_label: str) -> NewsEnrichment:
    return NewsEnrichment(
        forecast=None, previous=None, actual=None, surprise="pending",
        usd_read=f"GEMINI_API_KEY not set -- showing placeholder for {event_label}.",
        stubbed=True,
    )


def enrich_event(event_label: str, event_date_iso: str) -> NewsEnrichment:
    """
    Looks up forecast/previous/actual + a USD sentiment read for one
    calendar event (e.g. "Consumer Price Index (CPI)", "2026-09-11").
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _stub(event_label)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = f"Event: {event_label}\nDate: {event_date_iso}"
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )
        text = (response.text or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)
        return NewsEnrichment(
            forecast=data.get("forecast"), previous=data.get("previous"),
            actual=data.get("actual"), surprise=data.get("surprise", "pending"),
            usd_read=data.get("usd_read", ""), stubbed=False,
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the app
        return NewsEnrichment(
            forecast=None, previous=None, actual=None, surprise="pending",
            usd_read=f"Gemini lookup failed ({exc.__class__.__name__}); showing calendar data only.",
            stubbed=True,
        )


def dollar_sentiment_summary(dxy_bias: str, confidence: str) -> str:
    """
    One short Gemini-written paragraph synthesizing overall dollar
    sentiment, given the DXY structural bias this engine already
    computed locally. Falls back to a templated sentence with no key.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (f"DXY structure reads {dxy_bias} with {confidence} confidence. "
                f"(Set GEMINI_API_KEY for a live news-driven narrative.)")
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt = (
            f"The US Dollar Index's own price structure currently reads "
            f"'{dxy_bias}' with '{confidence}' confidence. Using current news "
            f"search, write ONE short paragraph (max 3 sentences) on what's "
            f"driving dollar sentiment right now (Fed policy expectations, "
            f"recent data surprises, yields, risk sentiment) and whether it "
            f"supports or conflicts with that structural read. Plain language, "
            f"no headers, no markdown."
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3,
            ),
        )
        return (response.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        return (f"DXY structure reads {dxy_bias} with {confidence} confidence. "
                f"(Live narrative unavailable: {exc.__class__.__name__}.)")
