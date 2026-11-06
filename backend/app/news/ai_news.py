"""
AI news research: live web search (Tavily) + analysis (Groq), both free tiers.

This replaces the old Gemini module. Gemini did search and writing in one
call through Google Search grounding. Here the two jobs are split:

  1. Tavily  -> fetches fresh web results (title, url, snippet) for a few
                queries. Free plan, no card needed. https://app.tavily.com
  2. Groq    -> reads ONLY those results and writes the answer. Free plan,
                no card needed. https://console.groq.com/keys
                Default model: openai/gpt-oss-120b, with automatic fallback
                to llama-3.3-70b-versatile if it is rate limited or down.

Sources returned to the app are the real URLs Tavily found, never URLs the
model wrote, so citations cannot be hallucinated. If no sources are found
the call fails instead of returning an ungrounded brief.

Keys:
  - The dashboard sends each user's own keys per request (headers), so users
    spend their own free quota.
  - GROQ_API_KEY / TAVILY_API_KEY env vars are used only by the server-side
    helpers (enrich_event, dollar_sentiment_summary), never for anonymous
    /api/news/ai calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

from . import cache

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TAVILY_URL = "https://api.tavily.com/search"

_PRIMARY_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b"
_FALLBACK_MODEL = "llama-3.3-70b-versatile"
GROQ_MODELS = [_PRIMARY_MODEL] + ([_FALLBACK_MODEL] if _PRIMARY_MODEL != _FALLBACK_MODEL else [])

_MAX_QUERIES = 3
_RESULTS_PER_QUERY = 5
_MAX_SOURCES = 8
_SNIPPET_CHARS = 700
_CONTEXT_CHARS = 6500      # keeps requests small enough for free-tier token limits
_MAX_PROMPT_CHARS = 14000

_ENRICH_TTL_SECONDS = 6 * 60 * 60
_SENTIMENT_TTL_SECONDS = 2 * 60 * 60


class AIError(Exception):
    """A failure with an HTTP status and a message that is safe to show the user."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------- #
# Tavily search
# --------------------------------------------------------------------------- #

def _tavily_search(query: str, api_key: str, topic: str, time_range: Optional[str]) -> list[dict]:
    body: dict[str, Any] = {
        "query": query[:300],
        "search_depth": "basic",
        "topic": topic if topic in ("news", "general", "finance") else "news",
        "max_results": _RESULTS_PER_QUERY,
        "include_answer": False,
    }
    if time_range in ("day", "week", "month", "year"):
        body["time_range"] = time_range
    try:
        resp = httpx.post(
            TAVILY_URL, json=body, timeout=20.0,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise AIError(502, f"Could not reach Tavily search ({exc.__class__.__name__}).") from exc

    if resp.status_code in (401, 403):
        raise AIError(401, "Tavily rejected the search key. Check it in Settings.")
    if resp.status_code in (429, 432, 433):
        raise AIError(429, "Tavily free search quota or rate limit reached. Try again later.")
    if resp.status_code >= 400:
        raise AIError(502, f"Tavily search failed (HTTP {resp.status_code}).")
    try:
        results = resp.json().get("results") or []
    except ValueError as exc:
        raise AIError(502, "Tavily returned an unreadable response.") from exc
    return [r for r in results if isinstance(r, dict) and r.get("url")]


def gather_sources(
    queries: list[str], api_key: str, topic: str = "news", time_range: Optional[str] = "week",
) -> list[dict]:
    """Run up to 3 searches in parallel and merge into one de-duplicated, ranked list."""
    queries = [q.strip() for q in queries if q and q.strip()][:_MAX_QUERIES]
    if not queries:
        raise AIError(400, "No search query was provided.")

    def run(q: str):
        try:
            return _tavily_search(q, api_key, topic, time_range), None
        except AIError as exc:
            return [], exc

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        outcomes = list(pool.map(run, queries))

    errors = [err for _, err in outcomes if err is not None]
    if len(errors) == len(outcomes):
        # Every search failed: surface the most useful reason (bad key beats rate limit).
        errors.sort(key=lambda e: 0 if e.status == 401 else 1)
        raise errors[0]

    # Round-robin so each query contributes its best hits before any gets a second slot.
    merged: list[dict] = []
    seen: set[str] = set()
    per_query = [res for res, _ in outcomes]
    for rank in range(_RESULTS_PER_QUERY):
        for res in per_query:
            if rank < len(res):
                item = res[rank]
                url = str(item["url"]).strip()
                if url and url not in seen:
                    seen.add(url)
                    merged.append({
                        "title": str(item.get("title") or "Web source").strip()[:200],
                        "url": url,
                        "content": str(item.get("content") or "").strip(),
                        "published": item.get("published_date") or "",
                    })
    return merged[:_MAX_SOURCES]


def _format_context(sources: list[dict]) -> str:
    blocks, used = [], 0
    for i, s in enumerate(sources, start=1):
        snippet = s["content"][:_SNIPPET_CHARS]
        date = f" ({s['published']})" if s.get("published") else ""
        block = f"[{i}] {s['title']}{date}\n{s['url']}\n{snippet}"
        if used + len(block) > _CONTEXT_CHARS and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Groq
# --------------------------------------------------------------------------- #

def _extract_json(text: str) -> dict:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        data = json.loads(t)
    except ValueError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found")
        data = json.loads(t[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON was not an object")
    return data


def _groq_chat(
    messages: list[dict], api_key: str, max_tokens: int, temperature: float,
    validate: Optional[Callable[[str], Any]] = None,
) -> tuple[str, Any, str]:
    """
    Calls Groq, trying each model in turn. Returns (text, validated_value, model).
    `validate` may raise ValueError to reject a reply (e.g. malformed JSON) and
    move on to the next model.
    """
    last: Optional[AIError] = None
    for model in GROQ_MODELS:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"  # fast, and keeps reasoning from eating the token budget
        try:
            resp = httpx.post(
                GROQ_URL, json=payload, timeout=45.0,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
        except httpx.HTTPError as exc:
            last = AIError(502, f"Could not reach Groq ({exc.__class__.__name__}).")
            continue

        if resp.status_code in (401, 403):
            raise AIError(401, "Groq rejected the API key. Check it in Settings.")
        if resp.status_code == 429:
            last = AIError(429, "Groq free rate limit reached. Wait a minute and try again.")
            continue
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = (resp.json().get("error") or {}).get("message", "")[:160]
            except ValueError:
                pass
            last = AIError(502, f"Groq request failed (HTTP {resp.status_code}). {detail}".strip())
            continue

        try:
            text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
        except (ValueError, KeyError, IndexError, TypeError):
            last = AIError(502, "Groq returned an unreadable response.")
            continue
        if not text:
            last = AIError(502, "Groq returned an empty answer.")
            continue

        if validate is None:
            return text, None, model
        try:
            return text, validate(text), model
        except ValueError:
            last = AIError(502, "The AI reply was not valid structured JSON. Please retry once.")
            continue

    raise last or AIError(502, "Groq request failed.")


def _system_prompt(json_mode: bool) -> str:
    today = datetime.now(timezone.utc).strftime("%A %Y-%m-%d %H:%M UTC")
    base = (
        f"Current date and time: {today}. "
        "You are a careful macroeconomic and forex news research assistant. "
        "Take every fact about news, consensus, previous values, releases or market moves ONLY from the "
        "WEB SEARCH RESULTS provided, or from data given in the user's request. "
        "If the results do not verify something, say it is unknown. Never invent numbers or links. "
    )
    if json_mode:
        return base + (
            "Reply with ONE JSON object and nothing else: no markdown fences, no commentary. "
            "If the request asks for a 'sources' field, return an empty array for it; real sources are "
            "attached automatically."
        )
    return base + (
        "Answer directly and concisely in plain text (no markdown tables or headings). Cite claims with the "
        "bracket numbers of the search results, like [1] or [2][3]. State uncertainty plainly."
    )


def research(
    prompt: str, queries: list[str], groq_key: str, tavily_key: str, *,
    json_mode: bool = False, topic: str = "news", time_range: Optional[str] = "week",
    max_tokens: int = 1200, temperature: float = 0.2,
) -> dict:
    """Search the web, then have Groq answer from those results. Returns text/data + sources."""
    if not groq_key or not tavily_key:
        raise AIError(400, "Both a Groq key and a Tavily key are required.")

    sources = gather_sources(queries, tavily_key, topic=topic, time_range=time_range)
    if not sources:
        raise AIError(502, "No web sources were found for this research, so no brief was produced.")

    user_content = (
        f"WEB SEARCH RESULTS:\n{_format_context(sources)}\n\n"
        f"REQUEST:\n{prompt[:_MAX_PROMPT_CHARS]}"
    )
    messages = [
        {"role": "system", "content": _system_prompt(json_mode)},
        {"role": "user", "content": user_content},
    ]
    text, data, model = _groq_chat(
        messages, groq_key, max_tokens, temperature,
        validate=_extract_json if json_mode else None,
    )
    public_sources = [{"title": s["title"], "url": s["url"]} for s in sources]
    out: dict[str, Any] = {"sources": public_sources, "model": model}
    if json_mode:
        out["data"] = data
    else:
        out["text"] = text
    return out


# --------------------------------------------------------------------------- #
# Server-side helpers (use GROQ_API_KEY / TAVILY_API_KEY from the environment)
# --------------------------------------------------------------------------- #

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


def _env_key(name: str, override: Optional[str] = None) -> str:
    return (override or os.environ.get(name, "")).strip()


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in ("null", "none", "unknown", "n/a") else None


def enrich_event(
    event_label: str,
    event_date_iso: str,
    forecast: Optional[str] = None,
    previous: Optional[str] = None,
    actual: Optional[str] = None,
    groq_key: Optional[str] = None,
    tavily_key: Optional[str] = None,
) -> NewsEnrichment:
    """
    Forecast / previous / actual plus a one-sentence USD read for one calendar event.

    If you pass figures in, Groq only phrases the read (no search needed). If you
    pass none, the figures are looked up live with Tavily and read by Groq.
    Falls back to a clearly labelled placeholder when keys are missing.
    """
    groq = _env_key("GROQ_API_KEY", groq_key)
    tavily = _env_key("TAVILY_API_KEY", tavily_key)
    have_figures = not (forecast is None and previous is None and actual is None)

    if not groq:
        return _stub(event_label, "GROQ_API_KEY not set")
    if not have_figures and not tavily:
        return _stub(event_label, "TAVILY_API_KEY not set, so live figures cannot be looked up")

    cache_key = f"enrich:{event_label}:{event_date_iso}:{forecast}:{previous}:{actual}"
    cached = cache.get(cache_key)
    if cached is not None:
        return NewsEnrichment(**cached)

    shape = (
        '{"forecast": string|null, "previous": string|null, "actual": string|null, '
        '"surprise": "beat"|"miss"|"inline"|"pending", "usd_read": "one plain sentence on what this means '
        'for USD strength, or that the print is still pending"}'
    )
    try:
        if have_figures:
            prompt = (
                f"Event: {event_label}\nDate: {event_date_iso}\nForecast: {forecast}\n"
                f"Previous: {previous}\nActual: {actual}\n"
                f"Use only these figures. Reply as JSON: {shape}"
            )
            messages = [
                {"role": "system", "content": _system_prompt(True)},
                {"role": "user", "content": prompt},
            ]
            _, data, _ = _groq_chat(messages, groq, 500, 0.2, validate=_extract_json)
        else:
            prompt = (
                f"Find the consensus forecast, previous reading and (if already released) the actual value "
                f"for: {event_label}, scheduled {event_date_iso}. If it has not been released, actual is null "
                f"and surprise is pending. Reply as JSON: {shape}"
            )
            res = research(
                prompt,
                [f"{event_label} forecast previous actual {event_date_iso}", f"{event_label} US release consensus"],
                groq, tavily, json_mode=True, max_tokens=700,
            )
            data = res["data"]

        surprise = str(data.get("surprise", "pending")).lower()
        if surprise not in ("beat", "miss", "inline", "pending"):
            surprise = "pending"
        result = NewsEnrichment(
            forecast=forecast if forecast is not None else _opt_str(data.get("forecast")),
            previous=previous if previous is not None else _opt_str(data.get("previous")),
            actual=actual if actual is not None else _opt_str(data.get("actual")),
            surprise=surprise, usd_read=str(data.get("usd_read", "")), stubbed=False,
        )
        cache.set(cache_key, result.to_dict(), _ENRICH_TTL_SECONDS)
        return result
    except AIError as exc:
        return NewsEnrichment(
            forecast=forecast, previous=previous, actual=actual, surprise="pending",
            usd_read=f"AI enrichment unavailable: {exc.message}", stubbed=True,
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash the app
        return NewsEnrichment(
            forecast=forecast, previous=previous, actual=actual, surprise="pending",
            usd_read=f"AI enrichment failed ({exc.__class__.__name__}); showing raw values only.",
            stubbed=True,
        )


def dollar_sentiment_summary(
    dxy_bias: str, confidence: str,
    groq_key: Optional[str] = None, tavily_key: Optional[str] = None,
) -> str:
    """
    Short paragraph on the dollar's current backdrop. With a Tavily key it reads live
    headlines; without one it falls back to clearly labelled general commentary.
    """
    groq = (groq_key or "").strip()
    tavily = (tavily_key or "").strip()
    if not groq:
        return (f"DXY structure reads {dxy_bias} with {confidence} confidence. "
                f"(Add a Groq API key in dashboard settings for AI commentary.)")

    fingerprint = hashlib.sha256(f"{groq}|{tavily}".encode("utf-8")).hexdigest()[:12]
    cache_key = f"sentiment:{fingerprint}:{dxy_bias}:{confidence}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    base_q = (
        f"The US Dollar Index's own price structure currently reads '{dxy_bias}' with '{confidence}' "
        f"confidence. Write ONE short paragraph (max 3 sentences, plain language, no headers or markdown) "
    )
    try:
        if tavily:
            res = research(
                base_q + "on what current news (Fed expectations, yields, risk sentiment) is driving the dollar "
                "and whether it supports or contradicts that structure. Cite with [n] numbers.",
                ["US dollar index DXY outlook Fed yields", "dollar news this week Federal Reserve"],
                groq, tavily, json_mode=False, max_tokens=500, temperature=0.3,
            )
            result = res["text"].strip() + "\n\n(AI summary of live web results.)"
        else:
            messages = [
                {"role": "system", "content": _system_prompt(False)},
                {"role": "user", "content": base_q + "on plausible drivers behind a read like this and whether "
                 "such structure typically holds or reverses. Speak generally; do not claim this reflects "
                 "today's actual news."},
            ]
            text, _, _ = _groq_chat(messages, groq, 400, 0.3)
            result = text.strip() + "\n\n(General AI commentary, not a live-news read.)"
        cache.set(cache_key, result, _SENTIMENT_TTL_SECONDS)
        return result
    except AIError as exc:
        return (f"DXY structure reads {dxy_bias} with {confidence} confidence. "
                f"(AI commentary unavailable: {exc.message})")
    except Exception as exc:  # noqa: BLE001
        return (f"DXY structure reads {dxy_bias} with {confidence} confidence. "
                f"(AI commentary unavailable: {exc.__class__.__name__}.)")
