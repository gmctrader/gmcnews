"""
ICT / Smart Money Concepts market-structure primitives.

This is the "where" layer: given a candle dataframe, find the swing
points, the break-of-structure / change-of-character sequence, fair
value gaps, and liquidity sweeps that everything else in this engine
(fundamental.py's DXY read, confluence.py's scoring) builds on top of.

Expected candle dataframe shape (same convention used throughout):
    columns: ["time", "open", "high", "low", "close"], indexed 0..n-1,
    oldest row first. `time` is a tz-aware UTC datetime.

Nothing here talks to a broker or the internet -- it is pure price-
structure math over whatever candles are handed to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Bias = Literal["bullish", "bearish"]


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: Literal["high", "low"]
    time: object = None


def find_swings(df, lookback: int = 3) -> list[SwingPoint]:
    """
    A swing high/low is a candle whose high (or low) is the most
    extreme within `lookback` candles on both sides -- the standard
    fractal definition used across ICT/SMC tooling.
    """
    swings: list[SwingPoint] = []
    n = len(df)
    for i in range(lookback, n - lookback):
        window_high = df["high"].iloc[i - lookback: i + lookback + 1]
        window_low = df["low"].iloc[i - lookback: i + lookback + 1]
        # Coerce to plain floats defensively: if the source dataframe
        # ever ends up with duplicate/MultiIndex columns upstream,
        # df["high"] silently becomes a DataFrame instead of a Series
        # and .iloc[i] returns a Series instead of a scalar -- which
        # makes `==` raise "truth value of a Series is ambiguous"
        # instead of comparing two numbers like it's supposed to.
        hi = float(df["high"].iloc[i])
        lo = float(df["low"].iloc[i])
        if hi == float(window_high.max()):
            swings.append(SwingPoint(index=i, price=hi, kind="high",
                                      time=df["time"].iloc[i] if "time" in df else None))
        if lo == float(window_low.min()):
            swings.append(SwingPoint(index=i, price=lo, kind="low",
                                      time=df["time"].iloc[i] if "time" in df else None))
    return swings


def current_bias(df, lookback: int = 3) -> Optional[Bias]:
    """
    Reads the last confirmed break-of-structure to decide whether
    structure is currently bullish (higher highs / higher lows, last
    break was to the upside) or bearish (lower lows / lower highs,
    last break was to the downside).

    Returns None if there isn't enough data to form an opinion.
    """
    if df is None or len(df) < lookback * 2 + 5:
        return None

    swings = find_swings(df, lookback=lookback)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return None

    last_close = df["close"].iloc[-1]

    # Most recent swing high/low the price has actually traded through
    # (a break of structure) tells us which side is currently in control.
    broken_high = next((h for h in reversed(highs) if last_close > h.price), None)
    broken_low = next((l for l in reversed(lows) if last_close < l.price), None)

    broken_high_idx = broken_high.index if broken_high else -1
    broken_low_idx = broken_low.index if broken_low else -1

    if broken_high_idx == -1 and broken_low_idx == -1:
        # No clean break yet -- fall back to higher-high/higher-low vs
        # lower-high/lower-low comparison of the last two swings of
        # each kind.
        if highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price:
            return "bullish"
        if highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price:
            return "bearish"
        return None

    return "bullish" if broken_high_idx > broken_low_idx else "bearish"


@dataclass
class StructureEvent:
    kind: Literal["BOS", "CHoCH"]
    direction: Bias
    index: int
    price: float


def structure_events(df, lookback: int = 3) -> list[StructureEvent]:
    """
    Walks the swing sequence and labels each break as either:
      BOS   -- Break of Structure: price breaks a swing point in the
               SAME direction as the prevailing trend (continuation).
      CHoCH -- Change of Character: price breaks a swing point AGAINST
               the prevailing trend (the first sign of a reversal).
    """
    swings = sorted(find_swings(df, lookback=lookback), key=lambda s: s.index)
    events: list[StructureEvent] = []
    trend: Optional[Bias] = None

    highs_seen: list[SwingPoint] = []
    lows_seen: list[SwingPoint] = []

    for s in swings:
        if s.kind == "high":
            for h in highs_seen:
                pass
            highs_seen.append(s)
        else:
            lows_seen.append(s)

    # Re-walk chronologically, checking each new swing against the
    # most recent opposite-side swing to see if it breaks it.
    for i, s in enumerate(swings):
        if s.kind == "high" and len(lows_seen) and s.index > 0:
            prior_highs = [h for h in swings[:i] if h.kind == "high"]
            if prior_highs and s.price > prior_highs[-1].price:
                direction: Bias = "bullish"
                kind = "BOS" if trend in (None, "bullish") else "CHoCH"
                events.append(StructureEvent(kind=kind, direction=direction,
                                              index=s.index, price=s.price))
                trend = "bullish"
        elif s.kind == "low":
            prior_lows = [l for l in swings[:i] if l.kind == "low"]
            if prior_lows and s.price < prior_lows[-1].price:
                direction = "bearish"
                kind = "BOS" if trend in (None, "bearish") else "CHoCH"
                events.append(StructureEvent(kind=kind, direction=direction,
                                              index=s.index, price=s.price))
                trend = "bearish"

    return events


@dataclass
class FVG:
    direction: Bias
    top: float
    bottom: float
    index: int
    filled: bool = False


def find_fair_value_gaps(df, max_lookback: int = 60) -> list[FVG]:
    """
    A bullish FVG is the gap between candle[i-1].high and candle[i+1].low
    when candle[i] pushes hard enough that those two don't overlap
    (the classic 3-candle ICT imbalance); bearish is the mirror image.
    Only scans the most recent `max_lookback` candles for relevance.
    """
    gaps: list[FVG] = []
    n = len(df)
    start = max(1, n - max_lookback)
    for i in range(start, n - 1):
        prev_high = df["high"].iloc[i - 1]
        prev_low = df["low"].iloc[i - 1]
        next_high = df["high"].iloc[i + 1]
        next_low = df["low"].iloc[i + 1]

        if next_low > prev_high:
            gaps.append(FVG(direction="bullish", top=next_low, bottom=prev_high, index=i))
        elif next_high < prev_low:
            gaps.append(FVG(direction="bearish", top=prev_low, bottom=next_high, index=i))

    last_close = df["close"].iloc[-1]
    for g in gaps:
        if g.bottom <= last_close <= g.top:
            g.filled = True
    return gaps


@dataclass
class LiquiditySweep:
    direction: Bias
    swept_price: float
    index: int


def find_liquidity_sweep(df, lookback: int = 3, recent_candles: int = 5) -> Optional[LiquiditySweep]:
    """
    A liquidity sweep: price pokes beyond a prior swing high/low
    (running the stops resting there) and then closes back on the
    other side of it within the same or next candle -- the classic
    ICT "stop hunt then reverse" signature that this engine looks
    for as its highest-quality entry trigger.
    """
    swings = find_swings(df, lookback=lookback)
    if not swings or len(df) < recent_candles + 1:
        return None

    recent = df.iloc[-recent_candles:]
    for kind, points in (("high", [s for s in swings if s.kind == "high"]),
                         ("low", [s for s in swings if s.kind == "low"])):
        if not points:
            continue
        level = points[-1].price
        for i in range(len(recent) - 1):
            row = recent.iloc[i]
            if kind == "high" and row["high"] > level and row["close"] < level:
                return LiquiditySweep(direction="bearish", swept_price=level,
                                       index=len(df) - recent_candles + i)
            if kind == "low" and row["low"] < level and row["close"] > level:
                return LiquiditySweep(direction="bullish", swept_price=level,
                                       index=len(df) - recent_candles + i)
    return None
