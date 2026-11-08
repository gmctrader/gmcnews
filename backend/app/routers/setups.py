from __future__ import annotations

import re
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from ..analysis import setups as engine
from ..analysis.fundamental import compute_dxy_bias, fundamental_alignment
from ..data.candles import get_candles, get_dxy_candles, provider_name
from ..news import news_calendar as nc

router = APIRouter(prefix="/api/setups", tags=["setups"])

UNIVERSE = {
    "Majors": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"],
    "Crosses": ["EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURAUD", "GBPCAD", "CADJPY", "CHFJPY"],
    "Metals & crypto": ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD"],
    "Indices & oil": ["US500", "NAS100", "US30", "GER40", "USOIL"],
}
MAX_BATCH = 6
_SYMBOL = re.compile(r"^[A-Z0-9]{3,10}$")

_dxy_cache: tuple[float, object] | None = None
_scan_cache: dict[tuple, tuple[float, dict]] = {}
_SCAN_TTL = 120.0
_DXY_TTL = 180.0


def _dxy():
    global _dxy_cache
    now = time.time()
    if _dxy_cache and now - _dxy_cache[0] < _DXY_TTL:
        return _dxy_cache[1]
    candles, source = get_dxy_candles()
    bias = compute_dxy_bias(candles, source=source)
    _dxy_cache = (now, bias)
    return bias


def _news_state() -> dict:
    blackout = nc.blackout_status(categories=("FOMC", "NFP", "CPI", "PPI"))
    soon = None
    try:
        upcoming = nc.upcoming(within_hours=3)
        if upcoming:
            e = upcoming[0]
            minutes = max(0, round((e.dt_utc - datetime.now(timezone.utc)).total_seconds() / 60))
            soon = f"{e.label or e.category} in {minutes} min: expect a spike and stop-hunt around the print"
    except Exception:
        pass
    return {
        "blackout": blackout.active,
        "blackout_note": (f"Inside the {blackout.phase}-release blackout for {blackout.event.label}; "
                          f"stand aside for {round(blackout.seconds_remaining / 60)} more min") if blackout.active else "",
        "news_soon": soon if not blackout.active else None,
    }


def _external(symbol: str) -> dict:
    dxy = _dxy()

    def align(direction: str):
        a = fundamental_alignment(symbol, dxy, direction)
        return a.score_bonus, a.note, a.agrees

    return {"align": align, **_news_state()}


def _parse_symbol(raw: str) -> str:
    s = raw.strip().upper().replace("/", "")
    if not _SYMBOL.match(s):
        raise HTTPException(400, f"Invalid symbol: {raw!r}")
    return s


def _strategies(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    bad = [k for k in keys if k not in engine.STRATEGIES]
    if bad:
        raise HTTPException(400, f"Unknown strategy: {', '.join(bad)}")
    return keys or None


def _scan_one(symbol: str, keys: list[str] | None, backtest: bool) -> dict:
    cache_key = (symbol, tuple(keys or ()), backtest)
    hit = _scan_cache.get(cache_key)
    if hit and time.time() - hit[0] < _SCAN_TTL:
        return hit[1]
    candles = get_candles(symbol)
    if not candles:
        result = {"symbol": symbol, "setups": [], "note": "No market data is available for this symbol right now."}
    else:
        result = engine.scan_symbol(symbol, candles, _external(symbol), keys, with_backtest=backtest)
        provider = provider_name(symbol)
        result["provider"] = provider
        result["provider_note"] = f"{provider} candles; quotes may be delayed and are not broker execution prices."
    _scan_cache[cache_key] = (time.time(), result)
    if len(_scan_cache) > 200:
        for k in sorted(_scan_cache, key=lambda x: _scan_cache[x][0])[:50]:
            _scan_cache.pop(k, None)
    return result


@router.get("/catalog")
def catalog():
    return {"strategies": engine.catalog(), "universe": UNIVERSE}


@router.get("/context")
def context():
    """Market-wide backdrop for the scanner: dollar bias, news risk, current session."""
    now = datetime.now(timezone.utc)
    session, killzone = engine.session_label(now.hour + now.minute / 60)
    news = _news_state()
    return {
        "dxy": _dxy().to_dict(),
        "session": session,
        "killzone": killzone,
        "news": {k: news[k] for k in ("blackout", "blackout_note", "news_soon")},
        "as_of": now.isoformat(),
    }


@router.get("/scan/{symbol}")
def scan_symbol(symbol: str,
                strategies: str | None = Query(None, description="Comma-separated strategy keys; default all"),
                backtest: bool = Query(True, description="Include walk-forward statistics")):
    return _scan_one(_parse_symbol(symbol), _strategies(strategies), backtest)


@router.get("")
def scan_many(symbols: str = Query(..., description=f"Comma-separated symbols, max {MAX_BATCH}"),
              strategies: str | None = Query(None), backtest: bool = Query(True),
              min_score: float = Query(0, ge=0, le=100)):
    syms = [_parse_symbol(s) for s in symbols.split(",") if s.strip()][:MAX_BATCH]
    if not syms:
        raise HTTPException(400, "Provide at least one symbol")
    keys = _strategies(strategies)
    results = [_scan_one(s, keys, backtest) for s in syms]
    flat = [x for r in results for x in r.get("setups", []) if x["score"] >= min_score]
    flat.sort(key=lambda x: x["score"], reverse=True)
    return {"setups": flat, "markets": [{k: v for k, v in r.items() if k != "setups"} for r in results]}
