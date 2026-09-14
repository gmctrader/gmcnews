"""
Confluence scorer -- combines the three angles this engine looks at
before calling anything a "signal":

  1. ICT/SMC structure  (market_structure.py) -- the "where": has
     liquidity just been swept, is there a CHoCH, is there an FVG to
     react from.
  2. DXY macro bias       (fundamental.py)     -- the "why": does the
     dollar's own trend support this direction.
  3. News timing          (news_calendar.py)   -- the "when": are we
     inside a blackout window where fresh setups shouldn't be opened.

Each symbol gets one Signal with a 0-100 confluence score, a
direction, and a plain-language breakdown of what fed the score --
never a black-box number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .market_structure import (
    current_bias, structure_events, find_fair_value_gaps, find_liquidity_sweep,
    find_swings,
)
from .fundamental import DXYBias, fundamental_alignment, FundamentalAlignment
from .risk import build_trade_levels, TradeLevels
from ..news import news_calendar as nc

# Base points for each ICT ingredient. Kept modest and additive, same
# philosophy as fundamental.py's confidence bonuses -- no single
# ingredient can manufacture a top-grade signal alone.
_POINTS = {
    "htf_mtf_agree": 20,
    "choch": 18,
    "bos": 8,
    "fvg_present": 12,
    "liquidity_sweep": 22,
    "in_news_blackout_penalty": -30,
    "post_news_window_bonus": 10,
}

GRADE_THRESHOLDS = [
    (80, "sniper"),
    (60, "high"),
    (40, "moderate"),
    (0, "low"),
]


def _grade(score: float) -> str:
    for threshold, label in GRADE_THRESHOLDS:
        if score >= threshold:
            return label
    return "low"


@dataclass
class Signal:
    symbol: str
    direction: Optional[str]           # "bullish" | "bearish" | None
    score: float
    grade: str
    htf_bias: Optional[str]
    mtf_bias: Optional[str]
    has_liquidity_sweep: bool
    has_choch: bool
    has_fvg: bool
    fundamental: FundamentalAlignment
    news_note: str
    reasons: list[str] = field(default_factory=list)
    veto: bool = False
    levels: Optional[TradeLevels] = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "score": round(self.score, 1),
            "grade": self.grade,
            "htf_bias": self.htf_bias,
            "mtf_bias": self.mtf_bias,
            "has_liquidity_sweep": self.has_liquidity_sweep,
            "has_choch": self.has_choch,
            "has_fvg": self.has_fvg,
            "fundamental": self.fundamental.to_dict(),
            "news_note": self.news_note,
            "reasons": self.reasons,
            "veto": self.veto,
            "levels": self.levels.to_dict() if self.levels else None,
        }


def build_signal(symbol: str, candles: dict, dxy: DXYBias,
                  veto_on_strong_conflict: bool = False,
                  veto_in_news_blackout: bool = True) -> Optional[Signal]:
    """
    candles: {"1h": df, "15m": df, ...} for THIS symbol (same shape
    fundamental.py expects for DXY itself).
    dxy: already-computed DXYBias, shared across all symbols in a scan.
    """
    htf_df = candles.get("1h", candles.get("30m"))
    mtf_df = candles.get("15m", candles.get("5m"))
    if htf_df is None or mtf_df is None:
        return None

    htf_bias = current_bias(htf_df)
    mtf_bias = current_bias(mtf_df)
    if htf_bias is None:
        return None

    direction = htf_bias
    reasons: list[str] = []
    score = 0.0

    if mtf_bias == htf_bias:
        score += _POINTS["htf_mtf_agree"]
        reasons.append("HTF and MTF structure agree.")

    events = structure_events(mtf_df)
    last_events = [e for e in events if e.direction == direction][-1:]
    has_choch = any(e.kind == "CHoCH" for e in last_events)
    has_bos = any(e.kind == "BOS" for e in last_events)
    if has_choch:
        score += _POINTS["choch"]
        reasons.append("Recent CHoCH confirms a fresh shift into this direction.")
    elif has_bos:
        score += _POINTS["bos"]
        reasons.append("Recent BOS continues the existing trend.")

    fvgs = find_fair_value_gaps(mtf_df)
    unfilled_matching = [g for g in fvgs if g.direction == direction and not g.filled]
    has_fvg = bool(unfilled_matching)
    if has_fvg:
        score += _POINTS["fvg_present"]
        reasons.append(f"{len(unfilled_matching)} unfilled FVG(s) to react from.")

    sweep = find_liquidity_sweep(mtf_df)
    has_sweep = bool(sweep and sweep.direction == direction)
    if has_sweep:
        score += _POINTS["liquidity_sweep"]
        reasons.append("Liquidity sweep just triggered in this direction.")

    align = fundamental_alignment(symbol, dxy, direction,
                                   veto_on_strong_conflict=veto_on_strong_conflict)
    score += align.score_bonus
    reasons.append(align.note)

    blackout = nc.blackout_status(categories=("FOMC", "NFP", "CPI", "PPI"))
    news_note = "No high-impact news in range."
    veto = align.veto
    if blackout.active:
        news_note = (
            f"Inside blackout for {blackout.event.label} "
            f"({blackout.phase}, {round(blackout.seconds_remaining)}s remaining)."
        )
        score += _POINTS["in_news_blackout_penalty"]
        if veto_in_news_blackout:
            veto = True
    else:
        post = nc.is_post_news_window(post_minutes=30)
        if post:
            score += _POINTS["post_news_window_bonus"]
            news_note = f"In the tradeable post-news window after {post.label}."

    score = max(0.0, min(100.0, score))

    entry = float(mtf_df["close"].iloc[-1])
    mtf_swings = find_swings(mtf_df)
    levels = build_trade_levels(
        symbol=symbol, direction=direction, entry=entry, mtf_df=mtf_df,
        swings=mtf_swings, sweep=(sweep if has_sweep else None),
    )
    if levels is None:
        reasons.append("No valid stop structure yet -- level plan withheld until one forms.")

    return Signal(
        symbol=symbol, direction=direction, score=score, grade=_grade(score),
        htf_bias=htf_bias, mtf_bias=mtf_bias,
        has_liquidity_sweep=has_sweep, has_choch=has_choch, has_fvg=has_fvg,
        fundamental=align, news_note=news_note, reasons=reasons, veto=veto,
        levels=levels,
    )
