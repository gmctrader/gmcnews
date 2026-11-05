from __future__ import annotations

from fastapi import APIRouter

from ..analysis.fundamental import compute_dxy_bias
from ..data.candles import get_dxy_candles

router = APIRouter(prefix="/api/dxy", tags=["dxy"])


@router.get("/bias")
def bias():
    candles, source = get_dxy_candles()
    dxy = compute_dxy_bias(candles, source=source)
    return dxy.to_dict()


@router.get("/snapshot")
def snapshot():
    candles, source = get_dxy_candles()
    frame = candles.get("15m")
    if frame is None:
        frame = candles.get("1h")
    if frame is None or frame.empty:
        return {"available": False, "provider": "Yahoo Finance", "source": source}
    latest = frame.iloc[-1]
    stamp = latest["time"]
    return {
        "available": True,
        "price": float(latest["close"]),
        "time": stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp),
        "source": source,
        "provider": "Yahoo Finance",
        "interval": "15m" if "15m" in candles else "1h",
        "note": "Yahoo Finance feed; quotes may be delayed. A synthetic fallback is a basket proxy, not the official DXY index.",
    }
