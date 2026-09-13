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
