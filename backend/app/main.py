from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import profiles as db
from .routers import profile, news, dxy, signals

app = FastAPI(title="DXY News Signals", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    db.init_db()


app.include_router(profile.router)
app.include_router(news.router)
app.include_router(dxy.router)
app.include_router(signals.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
