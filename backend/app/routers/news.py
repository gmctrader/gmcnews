from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from ..news import news_calendar as nc
from ..news import gemini_news
from ..news import user_history
from ..users import User
from .auth import get_current_user

router = APIRouter(prefix="/api/news", tags=["news"])


class NewsHistoryIn(BaseModel):
    event_id: str = Field(min_length=1, max_length=120)
    payload: dict[str, Any]


@router.get("/upcoming")
def upcoming(within_hours: float = Query(120.0, ge=1, le=720, description="Lookahead window in hours"),
             enrich: bool = Query(False, description="Fetch forecast/actual via Gemini")):
    events = nc.upcoming(within_hours=within_hours)
    out = []
    for e in events:
        payload = e.to_dict()
        if enrich:
            enrichment = gemini_news.enrich_event(payload["label"], payload["utc"][:10])
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


@router.get("/blackout")
def blackout():
    return nc.blackout_status().to_dict()


@router.get("/sentiment")
def sentiment(dxy_bias: str = Query(...), confidence: str = Query(...),
              x_gemini_api_key: str | None = Header(default=None)):
    return {"summary": gemini_news.dollar_sentiment_summary(
        dxy_bias, confidence, api_key_override=x_gemini_api_key
    )}
