from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from ..news import news_calendar as nc
from ..news import ai_news
from ..news import user_history
from ..news import shared_briefs
from ..users import User
from .auth import get_current_user

router = APIRouter(prefix="/api/news", tags=["news"])


class NewsHistoryIn(BaseModel):
    event_id: str = Field(min_length=1, max_length=120)
    payload: dict[str, Any]


class SharedBriefIn(BaseModel):
    payload: dict[str, Any]


class AIResearchIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    queries: list[str] = Field(min_length=1, max_length=3)
    format: Literal["json", "text"] = "text"
    topic: Literal["news", "general", "finance"] = "news"
    time_range: Optional[Literal["day", "week", "month", "year"]] = "week"
    max_tokens: int = Field(default=1200, ge=100, le=2500)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


@router.get("/upcoming")
def upcoming(within_hours: float = Query(120.0, ge=1, le=720, description="Lookahead window in hours"),
             enrich: bool = Query(False, description="Fetch forecast/actual via Tavily + Groq (server env keys)")):
    events = nc.upcoming(within_hours=within_hours)
    out = []
    for e in events:
        payload = e.to_dict()
        if enrich:
            enrichment = ai_news.enrich_event(payload["label"], payload["utc"][:10])
            payload["enrichment"] = enrichment.to_dict()
        out.append(payload)
    return {"events": out}


@router.get("/recent")
def recent(within_days: int = Query(5, ge=1, le=90, description="Past calendar window in days")):
    """Return recently scheduled releases; this is not an observed market outcome."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=within_days)
    events = [e.to_dict(now=now) for e in nc.get_events() if start <= e.dt_utc < now]
    return {"events": events, "window_days": within_days}


@router.get("/history")
def get_history(current_user: User = Depends(get_current_user)):
    return {"records": user_history.list_for_user(current_user.id)}


@router.put("/history")
def save_history(body: NewsHistoryIn, current_user: User = Depends(get_current_user)):
    event = body.payload.get("event") or {}
    if event.get("id") != body.event_id:
        raise HTTPException(400, "event_id must match payload.event.id")
    user_history.put(current_user.id, body.event_id, body.payload)
    return {"saved": True, "event_id": body.event_id}


@router.get("/shared")
def shared_list(within_days: int = Query(90, ge=1, le=120)):
    """Canonical pre-release calls and scored outcomes, identical on every device."""
    try:
        return {"records": shared_briefs.list_recent(within_days), "available": True}
    except Exception:
        return {"records": [], "available": False}


@router.put("/shared/{event_id}")
def shared_put(event_id: str, body: SharedBriefIn):
    """
    Publish a pre-release call (first valid call wins) or add its scored outcome.
    Returns the canonical record so the caller can adopt it.
    """
    try:
        record = shared_briefs.put(event_id, body.payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "Shared brief storage is unavailable.") from exc
    return {"saved": True, "record": record}


@router.get("/blackout")
def blackout():
    return nc.blackout_status().to_dict()


@router.post("/ai")
def ai_research(body: AIResearchIn,
                x_groq_api_key: str | None = Header(default=None),
                x_tavily_api_key: str | None = Header(default=None)):
    """
    Web-grounded AI research: Tavily search + Groq analysis.

    The caller supplies their own free keys in headers, so anonymous requests can
    never spend the server owner's quota. Keys are used for this request only and
    are never stored or logged.
    """
    groq_key = (x_groq_api_key or "").strip()
    tavily_key = (x_tavily_api_key or "").strip()
    if not groq_key or not tavily_key:
        raise HTTPException(400, "Add both your Groq key and Tavily key in Settings first.")
    try:
        return ai_news.research(
            body.prompt, body.queries, groq_key, tavily_key,
            json_mode=body.format == "json", topic=body.topic, time_range=body.time_range,
            max_tokens=body.max_tokens, temperature=body.temperature,
        )
    except ai_news.AIError as exc:
        raise HTTPException(exc.status, exc.message) from exc


@router.get("/sentiment")
def sentiment(dxy_bias: str = Query(...), confidence: str = Query(...),
              x_groq_api_key: str | None = Header(default=None),
              x_tavily_api_key: str | None = Header(default=None)):
    return {"summary": ai_news.dollar_sentiment_summary(
        dxy_bias, confidence, groq_key=x_groq_api_key, tavily_key=x_tavily_api_key
    )}
