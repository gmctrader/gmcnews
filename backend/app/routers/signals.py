from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..analysis.fundamental import compute_dxy_bias
from ..analysis.confluence import build_signal
from ..data.candles import _YF_TICKERS, get_candles, get_dxy_candles
from ..notify.dispatch import dispatch_signal

router = APIRouter(prefix="/api/signals", tags=["signals"])

DEFAULT_WATCHLIST = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY",
    "USDCHF", "USDCAD", "XAUUSD", "BTCUSD",
]


def _scan(watchlist: list[str]) -> tuple[dict, list[dict]]:
    dxy_candles, dxy_source = get_dxy_candles()
    dxy = compute_dxy_bias(dxy_candles, source=dxy_source)

    results = []
    for symbol in watchlist:
        candles = get_candles(symbol)
        if not candles:
            continue
        signal = build_signal(symbol, candles, dxy)
        if signal:
            payload = signal.to_dict()
            frame = candles.get("15m")
            if frame is None:
                frame = candles.get("1h")
            observed_at = None
            last_price = None
            if frame is not None and not frame.empty:
                stamp = frame.iloc[-1]["time"]
                observed_at = stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp)
                last_price = float(frame.iloc[-1]["close"])
            payload["market_data"] = {
                "provider": "Yahoo Finance",
                "symbol": _YF_TICKERS.get(symbol.upper()),
                "timeframe": "15m" if "15m" in candles else "1h",
                "as_of": observed_at,
                "last_price": last_price,
                "note": "Yahoo Finance feed; quotes may be delayed and are not broker execution prices.",
            }
            results.append(payload)

    results.sort(key=lambda s: s["score"], reverse=True)
    return dxy.to_dict(), results


@router.get("")
def scan(symbols: str = Query(",".join(DEFAULT_WATCHLIST),
                               description="Comma-separated symbol list")):
    watchlist = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    dxy, results = _scan(watchlist)
    return {"dxy": dxy, "signals": results}


@router.post("/{symbol}/send")
async def send_one(symbol: str):
    """
    Sends the current signal for one symbol to whichever channels are
    configured (Telegram/Discord/generic webhook) as a normal,
    plain-language trading signal with entry/stop/targets attached.
    """
    symbol = symbol.strip().upper()
    dxy_candles, dxy_source = get_dxy_candles()
    dxy = compute_dxy_bias(dxy_candles, source=dxy_source)
    candles = get_candles(symbol)
    if not candles:
        raise HTTPException(404, f"No candle data for {symbol}")
    signal = build_signal(symbol, candles, dxy)
    if not signal:
        raise HTTPException(404, f"No signal currently forming for {symbol}")

    payload = signal.to_dict()
    report = await dispatch_signal(payload)
    return {"signal": payload, "dispatch": report}


@router.post("/broadcast")
async def broadcast(
    symbols: str = Query(",".join(DEFAULT_WATCHLIST)),
    min_grade: str = Query("high", description="Minimum grade to send: sniper | high | moderate | low"),
):
    """Scans the watchlist and sends every signal at or above min_grade."""
    grade_rank = {"low": 0, "moderate": 1, "high": 2, "sniper": 3}
    threshold = grade_rank.get(min_grade, 2)

    watchlist = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    _, results = _scan(watchlist)
    qualifying = [s for s in results if grade_rank.get(s["grade"], 0) >= threshold and not s["veto"]]

    reports = []
    for signal in qualifying:
        report = await dispatch_signal(signal)
        reports.append({"symbol": signal["symbol"], **report})

    return {"sent_count": len(reports), "signals": reports}
