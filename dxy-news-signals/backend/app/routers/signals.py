from __future__ import annotations

from fastapi import APIRouter, Query

from ..analysis.fundamental import compute_dxy_bias
from ..analysis.confluence import build_signal
from ..data.candles import get_candles, get_dxy_candles

router = APIRouter(prefix="/api/signals", tags=["signals"])

DEFAULT_WATCHLIST = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY",
    "USDCHF", "USDCAD", "XAUUSD", "BTCUSD",
]


@router.get("")
def scan(symbols: str = Query(",".join(DEFAULT_WATCHLIST),
                               description="Comma-separated symbol list")):
    watchlist = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    dxy_candles, dxy_source = get_dxy_candles()
    dxy = compute_dxy_bias(dxy_candles, source=dxy_source)

    results = []
    for symbol in watchlist:
        candles = get_candles(symbol)
        if not candles:
            continue
        signal = build_signal(symbol, candles, dxy)
        if signal:
            results.append(signal.to_dict())

    results.sort(key=lambda s: s["score"], reverse=True)
    return {"dxy": dxy.to_dict(), "signals": results}
