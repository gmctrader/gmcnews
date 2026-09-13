"""
Fundamental / DXY macro-bias engine.

This is the "why" layer sitting above the ICT engine's "where" layer.
ICT/SMC finds the liquidity geometry (sweeps, CHoCH, FVGs); this module
answers a simpler but important question first: **which side of the
market does the dollar's own trend actually favor right now?**

Approach:
  1. Pull DXY (or a synthetic proxy built from the six pairs that make
     up the real index, if the broker doesn't list DXY/USDX directly).
  2. Read its own market structure (BOS/CHoCH, same primitives already
     used everywhere else in this engine) on HTF + MTF to get a
     bias + a confidence grade, not just a direction.
  3. Map that DXY bias onto every other symbol via a correlation table
     (most USD pairs are inversely correlated to DXY, USD-base pairs
     and gold are the two big exception families).
  4. Compare the mapped "expected" direction against whatever the ICT
     engine already concluded for that symbol. Agreement earns a
     confluence bonus; disagreement either shaves points or -- if
     `veto_on_conflict` is on -- blocks the trade outright. Nothing
     here ever *invents* a trade; it only agrees or disagrees with one
     ICT already found.

Nothing is fetched from the internet. DXY is either a real tradeable
symbol on the connected broker, or a synthetic index computed locally
from live MT5 candles of its basket components -- same "no live news
API" philosophy as analysis/news_calendar.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .market_structure import current_bias

# ---------------------------------------------------------------------------
# Which way each symbol tends to move when DXY (the US Dollar Index) rises.
#
#   "inverse" -- DXY up  -> symbol down, DXY down -> symbol up.
#                Almost every EURUSD-style "USD is the quote currency"
#                pair, plus gold/silver (priced in USD, so a stronger
#                dollar mechanically makes them cheaper, all else equal).
#   "direct"  -- DXY up  -> symbol up,   DXY down -> symbol down.
#                Pairs where USD is the BASE currency (USDJPY, USDCHF,
#                USDCAD, ...) move the same direction as the dollar
#                itself, plus USD-denominated risk-off/DXY-correlated
#                baskets like DJ30/US500/USTEC in some regimes (weak,
#                and regime-dependent, so indices are intentionally
#                left out of the hard table below -- see NEUTRAL note).
#
# This table is intentionally simple/static (real-world correlation
# drifts and regime-shifts), which is exactly why it's used as one
# input that can be overridden or disabled, never as the sole signal.
# ---------------------------------------------------------------------------
DXY_CORRELATION: dict[str, str] = {
    # Direct USD-quote majors/minors -> inverse to DXY
    "EURUSD": "inverse", "GBPUSD": "inverse", "AUDUSD": "inverse",
    "NZDUSD": "inverse",
    # USD-base majors -> direct with DXY
    "USDJPY": "direct", "USDCHF": "direct", "USDCAD": "direct",
    "USDSEK": "direct", "USDNOK": "direct", "USDMXN": "direct",
    "USDZAR": "direct", "USDTRY": "direct", "USDSGD": "direct",
    "USDHKD": "direct", "USDCNH": "direct",
    # Metals -- priced in USD, mechanically inverse to a stronger dollar
    "XAUUSD": "inverse", "XAGUSD": "inverse", "XPTUSD": "inverse", "XPDUSD": "inverse",
    # Crypto vs USD -- typically trades as a risk asset, inverse to DXY
    # strength in most regimes (not a law of physics, just the common lean)
    "BTCUSD": "inverse", "ETHUSD": "inverse",
    # USOIL/UKOIL are USD-denominated commodities -- lean inverse, though
    # supply/demand usually dominates the correlation far more than DXY does
    "USOIL": "inverse", "UKOIL": "inverse",
}

# Cross pairs (no USD leg at all, e.g. EURGBP, GBPJPY) aren't in the table
# on purpose -- DXY bias doesn't map cleanly onto a pair with no dollar
# leg, so those symbols simply get no fundamental opinion (neutral).


@dataclass
class DXYBias:
    bias: str                 # "bullish" | "bearish" | "neutral"
    confidence: str            # "weak" | "moderate" | "strong"
    htf_bias: Optional[str] = None
    mtf_bias: Optional[str] = None
    momentum_pct: float = 0.0  # % change over the lookback window on MTF
    source: str = "broker"     # "broker" (real DXY/USDX symbol) | "synthetic"

    def to_dict(self) -> dict:
        return {
            "bias": self.bias, "confidence": self.confidence,
            "htf_bias": self.htf_bias, "mtf_bias": self.mtf_bias,
            "momentum_pct": round(self.momentum_pct, 3), "source": self.source,
        }


def compute_dxy_bias(candles: dict, momentum_lookback: int = 20,
                      source: str = "broker") -> DXYBias:
    """
    candles: {"1h": df, "15m": df, ...} -- same shape as everywhere else
    in this engine. Uses 1h (falls back to 30m) for the primary bias,
    same as the ICT engine's own HTF read, plus 15m for a secondary
    confirmation and a simple % momentum reading to grade confidence.
    """
    htf_df = candles.get("1h", candles.get("30m"))
    mtf_df = candles.get("15m", candles.get("5m"))

    htf_bias = current_bias(htf_df) if htf_df is not None else None
    mtf_bias = current_bias(mtf_df) if mtf_df is not None else None

    momentum_pct = 0.0
    if mtf_df is not None and len(mtf_df) > momentum_lookback:
        recent = mtf_df.iloc[-1]["close"]
        past = mtf_df.iloc[-momentum_lookback]["close"]
        if past:
            momentum_pct = (recent - past) / past * 100.0

    if htf_bias is None:
        return DXYBias(bias="neutral", confidence="weak", htf_bias=None,
                        mtf_bias=mtf_bias, momentum_pct=momentum_pct, source=source)

    bias = "bullish" if htf_bias == "bullish" else "bearish"

    # Confidence: HTF/MTF agreement + momentum magnitude. Cheap and
    # transparent on purpose -- this is a lean, not a forecast.
    agree = (mtf_bias == htf_bias)
    strong_move = abs(momentum_pct) >= 0.35   # ~35 pips-equivalent move on DXY's own scale
    if agree and strong_move:
        confidence = "strong"
    elif agree or strong_move:
        confidence = "moderate"
    else:
        confidence = "weak"

    return DXYBias(bias=bias, confidence=confidence, htf_bias=htf_bias,
                    mtf_bias=mtf_bias, momentum_pct=momentum_pct, source=source)


def expected_direction(symbol: str, dxy: DXYBias) -> Optional[str]:
    """
    Maps the DXY bias onto a symbol via DXY_CORRELATION.
    Returns "bullish" / "bearish" for that SYMBOL (not for DXY), or
    None if the symbol isn't in the table (crosses, unmapped instruments)
    or DXY itself is neutral.
    """
    if dxy.bias == "neutral":
        return None
    corr = DXY_CORRELATION.get(symbol.upper())
    if corr is None:
        return None
    if corr == "direct":
        return dxy.bias
    # inverse
    return "bearish" if dxy.bias == "bullish" else "bullish"


@dataclass
class FundamentalAlignment:
    expected_bias: Optional[str]
    agrees: Optional[bool]        # None if no opinion (unmapped symbol / neutral DXY)
    confidence: str
    score_bonus: float             # additive confluence points, can be negative
    veto: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "expected_bias": self.expected_bias, "agrees": self.agrees,
            "confidence": self.confidence, "score_bonus": self.score_bonus,
            "veto": self.veto, "note": self.note,
        }


# How much a confident macro read can nudge the confluence score. Kept
# modest and symmetric so it can tip a borderline setup but never
# manufacture a sniper grade out of a genuinely weak one -- same
# philosophy as the news_bonus in analysis/confluence.py.
_BONUS_BY_CONFIDENCE = {"weak": 2.0, "moderate": 5.0, "strong": 9.0}
_PENALTY_BY_CONFIDENCE = {"weak": -1.0, "moderate": -4.0, "strong": -8.0}


def fundamental_alignment(symbol: str, dxy: DXYBias, symbol_bias: str,
                           veto_on_strong_conflict: bool = False) -> FundamentalAlignment:
    """
    Compares the ICT engine's own directional bias for `symbol`
    (`symbol_bias`, e.g. the HTF bias analyze_symbol already computed)
    against what DXY's macro trend would expect.

    veto_on_strong_conflict: if True, a "strong" confidence DXY read
    that directly contradicts the ICT bias blocks the trade instead of
    just penalizing its score -- e.g. don't buy EURUSD off an ICT
    sweep while DXY is in a strong, HTF/MTF-aligned uptrend. Off by
    default so this module only ever nudges scoring unless explicitly
    turned into a hard filter.
    """
    expected = expected_direction(symbol, dxy)
    if expected is None:
        return FundamentalAlignment(
            expected_bias=None, agrees=None, confidence=dxy.confidence,
            score_bonus=0.0, veto=False,
            note="No DXY correlation mapping for this symbol -- fundamental read skipped.",
        )

    agrees = (expected == symbol_bias)
    if agrees:
        bonus = _BONUS_BY_CONFIDENCE.get(dxy.confidence, 0.0)
        note = f"DXY {dxy.bias} ({dxy.confidence}) supports the {symbol_bias} bias."
    else:
        bonus = _PENALTY_BY_CONFIDENCE.get(dxy.confidence, 0.0)
        note = f"DXY {dxy.bias} ({dxy.confidence}) conflicts with the {symbol_bias} bias."

    veto = bool(veto_on_strong_conflict and not agrees and dxy.confidence == "strong")
    if veto:
        note += " Vetoed -- fundamental conflict filter is on."

    return FundamentalAlignment(
        expected_bias=expected, agrees=agrees, confidence=dxy.confidence,
        score_bonus=bonus, veto=veto, note=note,
    )
