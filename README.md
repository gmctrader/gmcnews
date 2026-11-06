# PipX Ledger — DXY Desk (News, ICT Signals & Precision Trade Plans)

**What's new in this pass:**
- **Full frontend redesign** — a new launch/landing page explains the engine (ICT structure, DXY macro bias, news timing, how levels are derived) before dropping into a restyled desk with a dark violet/cyan design system, an animated particle-network background, and pointer-driven 3D tilt on cards. All existing functionality (auth, onboarding, MT5 web terminal, chat, journal, settings) is preserved.
- **Real entry/stop/target levels, not guesses** — `backend/app/analysis/risk.py` computes an ATR-anchored invalidation stop off the actual structural point the trade depends on, plus staged 1R/2R/3.5R targets that snap to real unswept swing levels when one sits closer. Every number is rounded to the correct decimal precision for the instrument (JPY pairs, metals, crypto, standard FX). If there isn't a valid stop structure yet, the plan is withheld rather than invented — you'll see that explicitly in the UI.
- **Send signals as normal trading signals** — `backend/app/notify/dispatch.py` + new endpoints `POST /api/signals/{symbol}/send` and `POST /api/signals/broadcast?min_grade=high` format a signal (direction, grade, entry, stop, staged targets, reasoning) and push it to Telegram, Discord, and/or a generic webhook — whichever you configure via env vars. Nothing is sent if nothing is configured; the frontend's "Send signal" and "Broadcast" buttons surface the result either way.

A trading desk that combines three angles on every symbol:

1. **ICT/SMC structure** (`backend/app/analysis/market_structure.py`) — swing
   points, BOS/CHoCH, fair value gaps, liquidity sweeps. This is new — your
   uploaded `fundamental.py` imports `current_bias` from it but the module
   itself wasn't in your upload, so it's been built to match the interface
   your code already expects.
2. **DXY macro bias** (`backend/app/analysis/fundamental.py` — your file,
   unchanged except one import path fix) — reads the dollar's own structure
   and maps it onto each symbol via the correlation table.
3. **News timing + forecast/actual** — `news_calendar.py` (your file,
   unchanged) gives exact release times with zero internet dependency;
   `ai_news.py` (new) searches the live web with Tavily and has Groq read the
   results to fill in forecast/previous/actual values and a plain-language
   dollar-sentiment read, since your original calendar deliberately doesn't
   fetch that. Both services have free plans and need no card.

`confluence.py` (new) combines all three into one 0–100 score per symbol,
with a plain-language "why" for every point awarded.

## What's in here

```
backend/
  app/
    analysis/
      market_structure.py   NEW  — ICT primitives (BOS/CHoCH/FVG/sweep)
      fundamental.py         YOURS (1-line import fix)
      confluence.py         NEW  — combines all three angles into a score
    news/
      news_calendar.py       YOURS (unchanged)
      ai_news.py             NEW  — Tavily web search + Groq AI for forecasts/research
    data/
      candles.py             NEW  — free yfinance price feed (swap for MT5 later)
    routers/                 NEW  — FastAPI endpoints
    profiles.py              NEW  — trader profile storage (SQLite)
    main.py                   NEW  — FastAPI app
  requirements.txt
  .env.example
frontend/
  index.html                 NEW  — the mobile-first web app (works standalone)
  manifest.json, icon-*.png  — installable "add to home screen" support
```

## Running the backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional: add GROQ_API_KEY and TAVILY_API_KEY
uvicorn app.main:app --reload --port 8000
```

Get two free keys (no card needed): a Groq key at https://console.groq.com/keys
(AI analysis) and a Tavily key at https://app.tavily.com (live web search).
Dashboard users paste their own keys in Settings. The server-side keys in `.env`
are only used by `?enrich=true`. The app runs fine without any keys, it just
shows placeholder text instead of live forecast/sentiment data.

Endpoints:
- `GET /api/dxy/bias` — current DXY structural bias
- `GET /api/signals?symbols=EURUSD,GBPUSD,XAUUSD` — confluence scan, now including a `levels` object (entry/stop/targets/R:R/pips) per signal
- `POST /api/signals/{symbol}/send` — sends that symbol's current signal to whichever channels are configured
- `POST /api/signals/broadcast?min_grade=high` — scans the default watchlist and sends every signal at/above the given grade
- `GET /api/news/upcoming?within_hours=168` — calendar + Tavily/Groq enrichment (needs server keys)
- `GET /api/news/sentiment?dxy_bias=bullish&confidence=moderate` — AI narrative (send `x-groq-api-key`, optionally `x-tavily-api-key`)
- `POST /api/news/ai` — web-grounded research: Tavily search + Groq analysis; keys sent as `x-groq-api-key` and `x-tavily-api-key` headers
- `POST /api/profile`, `GET/PATCH /api/profile/{id}` — trader profiles

To actually deliver sent signals, set one or more of these on the backend (see `.env.example`): `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`, `DISCORD_WEBHOOK_URL`, or `GENERIC_WEBHOOK_URL` (for your own server or an MT5 EA bridge).

## Running the frontend

The frontend works **standalone with realistic demo data** — just open
`frontend/index.html` in a browser, or drag it into any static host
(Vercel, Netlify, GitHub Pages). To connect it to your live backend, set
`API_BASE` near the top of the `<script>` block in `index.html` to your
backend's URL (e.g. `"https://your-api.onrender.com"`) — every fetch
already falls back to the demo data if a request fails, so it degrades
gracefully.

On a phone, open the page in the browser and use "Add to Home Screen" —
`manifest.json` makes it install like a native app.

## Notes on what's real vs. mocked right now

- The **ICT structure, DXY bias, and confluence scoring logic are real and
  tested** — see the module docstrings; they run on actual OHLC data via
  yfinance.
- **Enrichment needs `GROQ_API_KEY` and `TAVILY_API_KEY`** to return live forecasts —
  without it, endpoints return clearly-labeled placeholder text so nothing
  breaks.
- The **frontend ships with demo data** so you can see and use the full
  design immediately; point `API_BASE` at your hosted backend when ready.
- **yfinance is a placeholder price feed.** Your original engine was
  MT5/broker-connected — when you're ready, replace `data/candles.py`'s
  `get_candles`/`get_dxy_candles` with your MT5 connector; every downstream
  module already just consumes a `{"1h": df, "15m": df}` dict, so nothing
  else needs to change.

## Suggested next steps

1. Get free Groq and Tavily keys and deploy the backend (Render/Railway/Fly.io all
   have free tiers that work for this).
2. Point the frontend's `API_BASE` at it.
3. Swap `candles.py` for your real MT5/broker feed when you're ready to
   trade off it instead of demo/yfinance data.

## Accounts, mobile, and data provenance

The hosted app at `https://pipx-six.vercel.app/` uses the same `/api` service on desktop and mobile. A sign-in can sync a user's saved news research across devices. Groq and Tavily API keys are isolated by account on each browser/device. They are sent only to this app's own `/api` service, which forwards them to Groq and Tavily for that single request and never stores or logs them; users enter their keys once per device.

Before production signup is enabled, connect persistent PostgreSQL storage to the Vercel project and set `AUTH_SECRET_KEY` in Vercel Environment Variables. The app reports account readiness from `/api/health` and disables signup when production storage or a secure signing key is missing. Vercel's Marketplace supports external Postgres integrations such as Neon and Supabase: https://vercel.com/docs/marketplace-storage.

The calendar links each scheduled release to its official BLS/Federal Reserve schedule. Market structure uses Yahoo Finance ticker feeds with provider candle timestamps; quotes may be delayed and are not broker execution prices. DXY fallback data is explicitly labeled synthetic. News briefs require web-search source citations (Tavily), and pair directions are derived from the sourced USD lean plus fresh live structure; disagreements are marked mixed.
