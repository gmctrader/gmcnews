from __future__ import annotations

from fastapi import APIRouter

from ..analysis.fundamental import compute_dxy_bias
from ..data.candles import get_dxy_candles, provider_name

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
        return {"available": False, "provider": provider_name("DXY"), "source": source}
    latest = frame.iloc[-1]
    stamp = latest["time"]
    return {
        "available": True,
        "price": float(latest["close"]),
        "time": stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp),
        "source": source,
        "provider": provider_name("DXY"),
        "interval": "15m" if "15m" in candles else "1h",
        "note": "Quotes may be delayed. \"computed\" = ICE-formula DXY from live component pairs; \"synthetic\" is a basket proxy, not the official DXY index.",
    }
