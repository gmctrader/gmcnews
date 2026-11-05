from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import profiles as db
from . import users as user_db
from . import security
from .news import user_history
from .routers import auth, profile, news, dxy, signals

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
    user_db.init_db()
    user_history.init_db()


app.include_router(auth.router)
app.include_router(profile.router)
app.include_router(news.router)
app.include_router(dxy.router)
app.include_router(signals.router)


@app.get("/api/health")
def health():
    on_vercel = bool(os.environ.get("VERCEL"))
    persistent = db.USE_POSTGRES
    if persistent:
        try:
            with db.get_conn() as conn:
                conn.execute("SELECT 1").fetchone()
        except Exception:
            persistent = False
    secure_secret = security.AUTH_SECRET_CONFIGURED
    ready = (persistent and secure_secret) if on_vercel else True
    return {
        "status": "ok",
        "accounts": {
            "ready": ready,
            "persistent": persistent or not on_vercel,
            "storage": "Postgres" if persistent else ("Postgres unavailable" if db.USE_POSTGRES else ("ephemeral SQLite" if on_vercel else "local SQLite")),
            "secure_signing_key": secure_secret,
        },
    }
