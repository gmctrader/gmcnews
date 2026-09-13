from __future__ import annotations

from fastapi import APIRouter, Query

from ..news import news_calendar as nc
from ..news import gemini_news

router = APIRouter(prefix="/api/news", tags=["news"])


@router.get("/upcoming")
def upcoming(within_hours: float = Query(168.0, description="Lookahead window in hours"),
             enrich: bool = Query(True, description="Fetch forecast/actual via Gemini")):
    events = nc.upcoming(within_hours=within_hours)
    out = []
    for e in events:
        payload = e.to_dict()
        if enrich:
            enrichment = gemini_news.enrich_event(payload["label"], payload["utc"][:10])
            payload["enrichment"] = enrichment.to_dict()
        out.append(payload)
    return {"events": out}


@router.get("/blackout")
def blackout():
    return nc.blackout_status().to_dict()


@router.get("/sentiment")
def sentiment(dxy_bias: str = Query(...), confidence: str = Query(...)):
    return {"summary": gemini_news.dollar_sentiment_summary(dxy_bias, confidence)}
