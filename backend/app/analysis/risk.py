"""
Trade-level engine -- turns a directional signal into an actual,
precise trade plan: entry, invalidation (stop), and staged targets,
each expressed to the correct decimal precision for the instrument.

This deliberately does NOT "guess" a round-number stop or a flat
1%-away target. Every level is derived from something observable in
the candles:

  - Stop  = the structural invalidation point (the most recent swing
            the setup depends on NOT breaking, e.g. the low that was
            just swept for a bullish sweep-reversal) minus a
            volatility buffer sized off ATR, so normal noise doesn't
            stop the trade out at the exact structural line.
  - Targets = staged R-multiples off the resulting risk distance
            (1R / 2R / 3.5R by default), snapped toward the nearest
            untouched opposing liquidity (an unswept swing high/low
            ahead of price) when one exists closer than the pure
            R-multiple would land, since that's where price is
            actually likely to react.

Nothing here trades or sizes a position for the user -- it reports
distances and levels; position sizing off account risk is left to the
trader/broker side, since this engine has no visibility into account
equity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Bias = Literal["bullish", "bearish"]

# ---------------------------------------------------------------------------
# Decimal precision + pip size per symbol family. Used only to format
# numbers and to express distances in a unit traders actually think in
# ("38 pips", "1.2 points", "$14.30"), never to change the underlying
# math -- the math always happens in raw price units.
# ---------------------------------------------------------------------------
_JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"}
_METALS = {"XAUUSD": (0.01, 2), "XAGUSD": (0.001, 3), "XPTUSD": (0.01, 2), "XPDUSD": (0.01, 2)}
_CRYPTO = {"BTCUSD": (1.0, 1), "ETHUSD": (0.1, 2)}
_OIL = {"USOIL": (0.01, 2), "UKOIL": (0.01, 2)}


def symbol_precision(symbol: str) -> tuple[float, int]:
    """Returns (pip_size, decimal_places) for a symbol."""
    sym = symbol.upper()
    if sym in _METALS:
        return _METALS[sym]
    if sym in _CRYPTO:
        return _CRYPTO[sym]
    if sym in _OIL:
        return _OIL[sym]
    if sym in _JPY_PAIRS:
        return (0.01, 3)
    return (0.0001, 5)  # standard FX majors/minors


def round_price(symbol: str, price: float) -> float:
    _, digits = symbol_precision(symbol)
    return round(price, digits)


def to_pips(symbol: str, distance: float) -> float:
    pip_size, _ = symbol_precision(symbol)
    if pip_size <= 0:
        return 0.0
    return round(abs(distance) / pip_size, 1)


def compute_atr(df, period: int = 14) -> Optional[float]:
    """
    Classic Average True Range over the last `period` candles.
    Returns None if there isn't enough data.
    """
    if df is None or len(df) < period + 1:
        return None
    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()
    closes = df["close"].astype(float).tolist()

    trs = []
    for i in range(1, len(df)):
        h, l, prev_c = highs[i], lows[i], closes[i - 1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)

    if len(trs) < period:
        return None
    recent = trs[-period:]
    return sum(recent) / len(recent)


@dataclass
class TradeLevels:
    symbol: str
    direction: Bias
    entry: float
    stop: float
    targets: list[float]                  # staged, nearest-first
    risk_reward: list[float]              # R-multiple actually achieved per target
    risk_distance: float                  # entry -> stop, in price units
    risk_pips: float
    reward_pips: list[float]
    atr: Optional[float]
    invalidation_source: str              # what the stop is actually anchored to
    target_source: list[str]              # what each target is anchored to

    def to_dict(self) -> dict:
        d = self.symbol
        return {
            "entry": round_price(d, self.entry),
            "stop": round_price(d, self.stop),
            "targets": [round_price(d, t) for t in self.targets],
            "risk_reward": [round(r, 2) for r in self.risk_reward],
            "risk_pips": self.risk_pips,
            "reward_pips": self.reward_pips,
            "atr": round(self.atr, 6) if self.atr is not None else None,
            "invalidation_source": self.invalidation_source,
            "target_source": self.target_source,
        }


_R_MULTIPLES = (1.0, 2.0, 3.5)
_ATR_STOP_BUFFER_MULT = 0.25   # extra room beyond the structural line
_MIN_ATR_STOP_MULT = 0.6       # floor: stop is never tighter than this many ATRs


def build_trade_levels(
    symbol: str,
    direction: Bias,
    entry: float,
    mtf_df,
    swings: list,
    sweep=None,
) -> Optional[TradeLevels]:
    """
    entry: last traded price on the entry timeframe (mtf_df's last close).
    swings: list of SwingPoint from market_structure.find_swings(mtf_df).
    sweep: optional LiquiditySweep already found for this direction.
    """
    atr = compute_atr(mtf_df, period=14)
    buffer = (atr or 0.0) * _ATR_STOP_BUFFER_MULT
    min_stop_distance = (atr or 0.0) * _MIN_ATR_STOP_MULT

    highs = sorted([s for s in swings if s.kind == "high"], key=lambda s: s.index)
    lows = sorted([s for s in swings if s.kind == "low"], key=lambda s: s.index)

    if direction == "bullish":
        if sweep is not None:
            invalidation_price = sweep.swept_price
            source = "swept liquidity low"
        elif lows:
            invalidation_price = lows[-1].price
            source = "most recent swing low"
        else:
            return None
        stop = invalidation_price - buffer
        if entry - stop < min_stop_distance:
            stop = entry - min_stop_distance
            source += " (ATR floor applied)"
        risk = entry - stop
        if risk <= 0:
            return None

        # Structural target candidates: unswept swing highs ahead of entry.
        structural_targets = sorted(
            {round(h.price, 8) for h in highs if h.price > entry}
        )
        targets, sources, rr = [], [], []
        for mult in _R_MULTIPLES:
            r_target = entry + risk * mult
            # Snap to the nearest structural level that sits between the
            # previous target and 1.4x the pure R-multiple distance, if
            # one exists -- that's where liquidity/reaction is likely,
            # rather than an arbitrary round number.
            candidates = [t for t in structural_targets if t > (targets[-1] if targets else entry) and t <= r_target * 1.4]
            if candidates:
                chosen = min(candidates, key=lambda t: abs(t - r_target))
                targets.append(chosen)
                sources.append(f"prior swing high near {mult:g}R")
            else:
                targets.append(r_target)
                sources.append(f"{mult:g}R (no structural level nearby)")

    else:  # bearish
        if sweep is not None:
            invalidation_price = sweep.swept_price
            source = "swept liquidity high"
        elif highs:
            invalidation_price = highs[-1].price
            source = "most recent swing high"
        else:
            return None
        stop = invalidation_price + buffer
        if stop - entry < min_stop_distance:
            stop = entry + min_stop_distance
            source += " (ATR floor applied)"
        risk = stop - entry
        if risk <= 0:
            return None

        structural_targets = sorted(
            {round(l.price, 8) for l in lows if l.price < entry}, reverse=True
        )
        targets, sources, rr = [], [], []
        for mult in _R_MULTIPLES:
            r_target = entry - risk * mult
            candidates = [t for t in structural_targets if t < (targets[-1] if targets else entry) and t >= r_target * 0.6]
            if candidates:
                chosen = max(candidates, key=lambda t: -abs(t - r_target))
                targets.append(chosen)
                sources.append(f"prior swing low near {mult:g}R")
            else:
                targets.append(r_target)
                sources.append(f"{mult:g}R (no structural level nearby)")

    for t in targets:
        realized_r = abs(t - entry) / risk if risk else 0.0
        rr.append(realized_r)

    reward_pips = [to_pips(symbol, abs(t - entry)) for t in targets]

    return TradeLevels(
        symbol=symbol, direction=direction, entry=entry, stop=stop,
        targets=targets, risk_reward=rr, risk_distance=risk,
        risk_pips=to_pips(symbol, risk), reward_pips=reward_pips,
        atr=atr, invalidation_source=source, target_source=sources,
    )
