"""
Intraday setup engine -- the "scan everything" layer.

Runs eight independent strategies (SMC/ICT, session, trend and volatility
families) over a symbol's 15m candles, with 1h/4h structure as higher-timeframe
context, and returns every setup that is actionable *right now*, each with a
structural intraday stop (three tiers), staged liquidity targets, a 0-100
confluence score and walk-forward statistics for that strategy on that symbol.

Design rules
------------
* Every detector is written once, for longs, over a sign-flipped "view" of the
  candles. Shorts run the identical logic on the mirror image, so the two sides
  can never drift apart.
* Detectors are causal: detect(view, i) only reads bars <= i and swings that
  were already confirmed at bar i. That is what makes the walk-forward
  backtest honest (no look-ahead).
* Nothing here talks to the network. Candles in, plain dicts out.

Candle input: {"15m": df, "1h": df}, columns time/open/high/low/close,
oldest first (the same shape data/candles.py already returns).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import risk as _risk

SWING_LB = 3
BAR_SEC = 900
WARMUP = 210                 # bars before EMA200 / ATR are trustworthy
MAX_RISK_ATR = 3.0           # intraday: reject structures needing a stop wider than this
MIN_RISK_ATR = 0.6
STOP_TIERS = {"tight": 0.10, "standard": 0.25, "wide": 0.60}   # ATR buffer beyond structure
SCALE_OUT = (0.5, 0.3, 0.2)  # share of position closed at TP1/TP2/TP3 in the backtest
MAX_HOLD_BARS = 64           # 16h
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
GRADES = ((80, "sniper"), (60, "high"), (40, "moderate"), (0, "low"))

# Direction filter ("strict" mode): suppresses the setups that keep printing against the real move.
REVERSAL_KEYS = {"sweep_choch", "asia_sweep", "pdhl_raid"}   # counter-structure by design
STRICT_MIN_SCORE = 40        # hide "low" grade noise while the filter is on
ALIGN_OK = -0.2              # trend alignment (-1..+1, trade direction) needed with no further proof
ALIGN_FLOOR = -0.6           # a reversal may never fight a trend worse than this
CONFLICT_GAP = 10            # side-strength gap needed to keep one side when both directions are live

NAN = float("nan")


_INDICES = {"US500", "SPX500", "NAS100", "US100", "DJ30", "US30", "UK100", "GER40", "DE40", "JP225"}


def round_price(symbol: str, price: float) -> float:
    return round(price, 1) if symbol.upper() in _INDICES else _risk.round_price(symbol, price)


def to_pips(symbol: str, distance: float) -> float:
    """Pips for FX/metals/crypto; whole index points for indices."""
    return round(abs(distance), 1) if symbol.upper() in _INDICES else _risk.to_pips(symbol, distance)


def grade_for(score: float) -> str:
    for threshold, label in GRADES:
        if score >= threshold:
            return label
    return "low"


# --------------------------------------------------------------------------- #
# Indicator + session context
# --------------------------------------------------------------------------- #

def _arrays(df):
    ts = pd.to_datetime(df["time"], utc=True)
    ep = ((ts - EPOCH) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
    g = lambda k: df[k].to_numpy(dtype=float)
    return ep, g("open"), g("high"), g("low"), g("close")


def _swings(h, l, lb=SWING_LB):
    n = len(h)
    if n < 2 * lb + 1:
        return np.array([], int), np.array([], int)
    wh = sliding_window_view(h, 2 * lb + 1).max(axis=1)
    wl = sliding_window_view(l, 2 * lb + 1).min(axis=1)
    centre = np.arange(lb, n - lb)
    hi = centre[h[lb:n - lb] == wh]
    lo = centre[l[lb:n - lb] == wl]
    if len(hi):
        hi = hi[np.r_[True, np.diff(hi) > 1]]
    if len(lo):
        lo = lo[np.r_[True, np.diff(lo) > 1]]
    return hi, lo


def _ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def _atr(h, l, c, p=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / p, adjust=False).mean().to_numpy()


def _rsi(c, p=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / p, adjust=False).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / p, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).to_numpy()


def _last_confirmed(conf, px, n):
    """Per bar: price of the newest swing already confirmed at that bar."""
    if len(px) == 0:
        return np.full(n, NAN)
    k = np.searchsorted(conf, np.arange(n), side="right")
    return np.where(k > 0, px[np.maximum(k - 1, 0)], NAN)


def _structure_state(h, l, c, lb):
    """+1 / -1 / 0 per bar: direction of the last confirmed structure break."""
    n = len(c)
    hi, lo = _swings(h, l, lb)
    st = np.zeros(n, np.int8)
    last_hi = last_lo = NAN
    ph = pl = 0
    cur = 0
    for j in range(n):
        while ph < len(hi) and hi[ph] + lb <= j:
            last_hi = h[hi[ph]]
            ph += 1
        while pl < len(lo) and lo[pl] + lb <= j:
            last_lo = l[lo[pl]]
            pl += 1
        if last_hi == last_hi and c[j] > last_hi:
            cur, last_hi = 1, NAN
        if last_lo == last_lo and c[j] < last_lo:
            cur, last_lo = -1, NAN
        st[j] = cur
    return st


def _htf_bias(df_htf, lb, base_ep, tf_sec):
    if df_htf is None or len(df_htf) < 2 * lb + 6:
        return np.zeros(len(base_ep), np.int8)
    ep, _, h, l, c = _arrays(df_htf)
    st = _structure_state(h, l, c, lb)
    k = np.searchsorted(ep + tf_sec, base_ep, side="right")   # HTF bars fully closed at t
    out = np.zeros(len(base_ep), np.int8)
    m = k > 0
    out[m] = st[k[m] - 1]
    return out


def _resample_4h(df_1h):
    if df_1h is None or len(df_1h) < 12:
        return None
    s = df_1h.set_index(pd.to_datetime(df_1h["time"], utc=True))
    out = s.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return out.rename_axis("time").reset_index()


class Ctx:
    """Everything a detector may read, computed once per symbol."""

    def __init__(self, symbol: str, candles: dict):
        self.symbol = symbol.upper()
        m15 = candles.get("15m")
        h1 = candles.get("1h")
        ep, o, h, l, c = _arrays(m15)
        self.live_price = float(c[-1])
        if ep[-1] + BAR_SEC > time.time() - 5:        # drop the still-forming candle
            ep, o, h, l, c = ep[:-1], o[:-1], h[:-1], l[:-1], c[:-1]
        self.ep, self.o, self.h, self.l, self.c = ep, o, h, l, c
        n = self.n = len(c)
        self.atr = _atr(h, l, c)
        self.e21, self.e50, self.e200 = _ema(c, 21), _ema(c, 50), _ema(c, 200)
        self.rsi = _rsi(c)
        sma = pd.Series(c).rolling(20).mean()
        sd = pd.Series(c).rolling(20).std(ddof=0)
        self.sma20 = sma.bfill().to_numpy()
        self.bb_up = (sma + 2 * sd).bfill().to_numpy()
        self.bb_lo = (sma - 2 * sd).bfill().to_numpy()
        kc_mid = pd.Series(_ema(c, 20))
        atr_s = pd.Series(self.atr)
        self.sq_on = ((sma + 2 * sd) < (kc_mid + 1.5 * atr_s)).to_numpy() & ((sma - 2 * sd) > (kc_mid - 1.5 * atr_s)).to_numpy()
        self.sh_idx, self.sl_idx = _swings(h, l)
        rh = pd.Series(h).rolling(96, min_periods=24).max()
        rl = pd.Series(l).rolling(96, min_periods=24).min()
        self.eq = ((rh + rl) / 2).bfill().to_numpy()

        self.hour = (ep % 86400) / 3600.0
        self.uday = ep // 86400
        sday = (ep - 22 * 3600) // 86400
        self.sday = sday
        self._sessions(n)

        self.b15 = _structure_state(h, l, c, SWING_LB)
        self.b1 = _htf_bias(h1, 3, ep, 3600)
        self.b4 = _htf_bias(_resample_4h(h1), 2, ep, 4 * 3600)
        self.views = {1: View(self, 1), -1: View(self, -1)}

    def _sessions(self, n):
        h, l, ep, hour, uday, sday = self.h, self.l, self.ep, self.hour, self.uday, self.sday
        # Asian range (00:00-07:00 UTC) -- visible from 07:00 onward the same UTC day
        self.asia_hi = np.full(n, NAN)
        self.asia_lo = np.full(n, NAN)
        for d in np.unique(uday):
            idx = np.nonzero(uday == d)[0]
            asia = idx[hour[idx] < 7]
            after = idx[hour[idx] >= 7]
            if len(asia) >= 16 and len(after):
                self.asia_hi[after] = h[asia].max()
                self.asia_lo[after] = l[asia].min()
        # prior completed session day (22:00 UTC roll) high/low + average daily range
        days = np.unique(sday)
        counts = {d: int((sday == d).sum()) for d in days}
        full_min = 0.5 * float(np.median(list(counts.values())))
        rng, hi_d, lo_d = {}, {}, {}
        for d in days:
            if counts[d] >= full_min:
                m = sday == d
                hi_d[d], lo_d[d] = h[m].max(), l[m].min()
                rng[d] = hi_d[d] - lo_d[d]
        self.pdh = np.full(n, NAN)
        self.pdl = np.full(n, NAN)
        self.adr = np.full(n, NAN)
        done = []
        for d in days:
            m = sday == d
            if done:
                self.pdh[m], self.pdl[m] = hi_d[done[-1]], lo_d[done[-1]]
                self.adr[m] = np.mean([rng[x] for x in done[-5:]])
            if d in rng:
                done.append(d)
        g = pd.Series(h).groupby(sday)
        self.day_hi = g.cummax().to_numpy()
        self.day_lo = pd.Series(l).groupby(sday).cummin().to_numpy()


_WORDS = {
    1: {"side": "Sell-side", "lowhigh": "lower high", "block": "demand", "gap": "bullish", "alow": "Asian low",
        "ahigh": "Asian high", "beyond": "above", "turn": "up", "stack": "21 > 50 > 200 and rising",
        "hi": "high", "pdx": "prior-day high", "pds": "prior-day low"},
    -1: {"side": "Buy-side", "lowhigh": "higher low", "block": "supply", "gap": "bearish", "alow": "Asian high",
         "ahigh": "Asian low", "beyond": "below", "turn": "down", "stack": "21 < 50 < 200 and falling",
         "hi": "low", "pdx": "prior-day low", "pds": "prior-day high"},
}


class View:
    """Long-only picture of the candles. s=+1 is the market; s=-1 is its mirror."""

    def __init__(self, ctx: Ctx, s: int):
        self.ctx, self.s = ctx, s
        n = ctx.n
        if s > 0:
            self.H, self.L, self.C, self.O = ctx.h, ctx.l, ctx.c, ctx.o
            hi_idx, hi_px = ctx.sh_idx, ctx.h[ctx.sh_idx]
            lo_idx, lo_px = ctx.sl_idx, ctx.l[ctx.sl_idx]
            self.asia_hi, self.asia_lo = ctx.asia_hi, ctx.asia_lo
            self.pdh, self.pdl, self.day_hi = ctx.pdh, ctx.pdl, ctx.day_hi
            self.bb_up = ctx.bb_up
        else:
            self.H, self.L, self.C, self.O = -ctx.l, -ctx.h, -ctx.c, -ctx.o
            hi_idx, hi_px = ctx.sl_idx, -ctx.l[ctx.sl_idx]
            lo_idx, lo_px = ctx.sh_idx, -ctx.h[ctx.sh_idx]
            self.asia_hi, self.asia_lo = -ctx.asia_lo, -ctx.asia_hi
            self.pdh, self.pdl, self.day_hi = -ctx.pdl, -ctx.pdh, -ctx.day_lo
            self.bb_up = -ctx.bb_lo
        self.hi_idx, self.hi_px, self.lo_idx, self.lo_px = hi_idx, hi_px, lo_idx, lo_px
        self.hi_conf, self.lo_conf = hi_idx + SWING_LB, lo_idx + SWING_LB
        self.atr = ctx.atr
        self.t = _WORDS[s]
        self.e21, self.e50, self.e200 = s * ctx.e21, s * ctx.e50, s * ctx.e200
        self.sma = s * ctx.sma20
        self.rsi = 50 + s * (ctx.rsi - 50)
        self.eq = s * ctx.eq
        self.bias = (s * ctx.b4, s * ctx.b1, s * ctx.b15)
        H, L, C, O, a = self.H, self.L, self.C, self.O, self.atr
        self.last_hi = _last_confirmed(self.hi_conf, hi_px, n)
        prev = np.r_[NAN, self.last_hi[:-1]]
        with np.errstate(invalid="ignore"):
            self.bos = C > prev
        body = C - O
        self.impulse = (body >= 1.1 * a) & (body >= 0.55 * (H - L))
        self.fvg = np.zeros(n, bool)
        if n > 3:
            gap = L[2:] - H[:-2]
            self.fvg[1:-1] = (gap >= 0.2 * a[1:-1]) & (body[1:-1] >= 0.8 * a[1:-1])

    def px(self, x: float) -> float:
        """View-space price back to the real market price (for human-readable notes)."""
        return self.s * float(x)

    def swings_before(self, kind: str, i: int, count: int, max_age: int):
        conf, idx, px = ((self.hi_conf, self.hi_idx, self.hi_px) if kind == "hi"
                         else (self.lo_conf, self.lo_idx, self.lo_px))
        k = int(np.searchsorted(conf, i, side="right"))
        out = []
        for j in range(k - 1, max(-1, k - 1 - count), -1):
            if i - idx[j] > max_age:
                break
            out.append((int(idx[j]), float(px[j])))
        return out


# --------------------------------------------------------------------------- #
# Direction filter. Causal (reads bar i and earlier only), so the live scan and the
# walk-forward statistics apply exactly the same rule.
# --------------------------------------------------------------------------- #

def trend_alignment(v: View, i: int) -> tuple[float, int, int]:
    """(-1..+1, 4h vote, 1h vote) in view space: +1 means the trend is with this side."""
    b4, b1, b15 = (int(x[i]) for x in v.bias)
    stack = 1 if v.e21[i] > v.e50[i] > v.e200[i] else -1 if v.e21[i] < v.e50[i] < v.e200[i] else 0
    ema = stack if stack else (0.5 if v.C[i] > v.e200[i] else -0.5)
    return (2 * b4 + 2 * b1 + b15 + ema) / 6.0, b4, b1


def direction_gate(v: View, i: int, key: str) -> tuple[bool, Optional[str], float]:
    """Is this side allowed to trade here? Returns (ok, reason_if_not, alignment)."""
    al, b4, b1 = trend_alignment(v, i)
    b15 = int(v.bias[2][i])
    if b4 < 0 and b1 < 0:
        return False, "Fights both the 4h and 1h structure", al
    if al >= ALIGN_OK:
        return True, None, al
    if key in REVERSAL_KEYS:
        if al > ALIGN_FLOOR and v.C[i] < v.eq[i] and b15 > 0:
            return True, None, al        # counter-trend reversal: only from the extreme, with 15m already flipped
        return False, "Counter-trend reversal without an extreme entry and a 15m flip", al
    return False, "Continuation setup against the higher-timeframe trend", al


def _rejection(v: View, i: int) -> bool:
    """Signal candle closed with this side in control (green, upper half of its range)."""
    rng = v.H[i] - v.L[i]
    return bool(rng > 0 and v.C[i] > v.O[i] and (v.C[i] - v.L[i]) / rng >= 0.5)


# --------------------------------------------------------------------------- #
# Detectors (long logic, view space). Each returns a candidate dict or None.
# --------------------------------------------------------------------------- #

STRATEGIES: dict[str, dict] = {}


def strategy(key, name, family, desc):
    def deco(fn):
        STRATEGIES[key] = {"key": key, "name": name, "family": family, "desc": desc, "fn": fn}
        return fn
    return deco


def _cand(entry, stop_ref, zone, etype, notes, base, expire=16, **extra):
    return {"entry": entry, "stop_ref": stop_ref, "zone": zone, "etype": etype,
            "notes": notes, "base": base, "expire": expire, **extra}


@strategy("sweep_choch", "Sweep + CHoCH", "SMC / ICT",
          "Stops run below a swing low, price reclaims it and breaks the last lower high. Entry on the 50-79% retrace of the reversal leg.")
def d_sweep_choch(v: View, i: int):
    a = v.atr[i]
    if i < 60:
        return None
    for s_idx, s_px in v.swings_before("lo", i, 4, 90):
        below = np.nonzero(v.L[s_idx + 1:i + 1] < s_px)[0]
        if not len(below):
            continue
        k = s_idx + 1 + int(below[0])
        if i - k > 30 or k + 1 > i:
            continue
        if not (v.C[k:min(k + 3, i + 1)] > s_px).any():
            continue
        ext = float(v.L[k:i + 1].min())
        refs = [(ix, px) for ix, px in v.swings_before("hi", k, 6, 60) if ix < k]
        if not refs:
            continue
        ref_hi = refs[0][1]
        hit = np.nonzero(v.C[k + 1:i + 1] > ref_hi)[0]
        if not len(hit):
            continue
        m = k + 1 + int(hit[0])
        if i - m > 14 or (v.C[m] - v.O[m]) < 0.4 * v.atr[m]:
            continue
        leg_hi = float(v.H[m:i + 1].max())
        R = leg_hi - ext
        if R < 1.2 * a or R > 6 * a:
            continue
        retr = (leg_hi - v.C[i]) / R
        if retr > 0.85 or v.C[i] <= ext:
            continue
        z_lo, z_hi = leg_hi - 0.79 * R, leg_hi - 0.5 * R
        in_zone = v.C[i] <= z_hi
        t = v.t
        notes = [f"{t['side']} liquidity at {v.px(s_px):.5g} swept and reclaimed {i - k} bars ago",
                 f"CHoCH: closed {t['beyond']} the last {t['lowhigh']} ({v.px(ref_hi):.5g}) {i - m} bars ago"]
        if np.isfinite(v.pdl[i]) and abs(s_px - v.pdl[i]) < 0.35 * a:
            notes.append(f"Swept level coincides with the {t['pds']}")
        return _cand(leg_hi - 0.62 * R if not in_zone else float(v.C[i]), ext, (z_lo, z_hi),
                     "market" if in_zone else "limit", notes, 34, sweep_depth=(s_px - ext) / a)
    return None


@strategy("order_block", "Order Block Retest", "SMC / ICT",
          "Last down-candle before a displacement that broke structure. Unmitigated; entry at its mean threshold.")
def d_order_block(v: View, i: int):
    lo_i = max(25, i - 45)
    for m in np.nonzero(v.impulse[lo_i:i] & v.bos[lo_i:i])[0][::-1]:
        m = lo_i + int(m)
        j = next((q for q in range(m - 1, max(m - 7, 1), -1) if v.C[q] < v.O[q]), None)
        if j is None:
            continue
        a = v.atr[i]
        zl, zh = float(v.L[j]), float(v.O[j])
        if zh - zl > 2.5 * a:
            zl = float(v.C[j])
        if zh - zl < 0.15 * a:
            continue
        if m + 1 <= i - 1 and v.L[m + 1:i].min() <= zh:
            continue                                   # already mitigated
        if v.C[i] < zl:
            continue
        touching = v.L[i] <= zh
        if not touching and (v.C[i] - zh) / a > 2.5:
            continue
        z = sorted((v.px(zl), v.px(zh)))
        notes = [f"Unmitigated {v.t['block']} block ({z[0]:.5g}-{z[1]:.5g}) left by a structure-breaking displacement {i - m} bars ago"]
        return _cand(float(v.C[i]) if touching else zl + 0.5 * (zh - zl), zl, (zl, zh),
                     "market" if touching else "limit", notes, 30,
                     disp=(v.C[m] - v.O[m]) / v.atr[m])
    return None


@strategy("fvg_retest", "FVG Retest", "SMC / ICT",
          "Retrace into the fair value gap left by displacement. Entry at the gap's consequent encroachment (50%).")
def d_fvg_retest(v: View, i: int):
    lo_i = max(3, i - 40)
    for k in np.nonzero(v.fvg[lo_i:i - 1])[0][::-1]:
        k = lo_i + int(k)
        zl, zh = float(v.H[k - 1]), float(v.L[k + 1])
        ce = zl + 0.5 * (zh - zl)
        if k + 2 <= i - 1 and v.L[k + 2:i].min() <= ce:
            continue
        if v.C[i] < zl:
            continue
        a = v.atr[i]
        touching = v.L[i] <= zh
        if not touching and (v.C[i] - zh) / a > 2.0:
            continue
        broke = bool(v.bos[k] or v.bos[k + 1])
        z = sorted((v.px(zl), v.px(zh)))
        notes = [f"Unfilled {v.t['gap']} gap ({z[0]:.5g}-{z[1]:.5g}), {i - k} bars old"]
        if broke:
            notes.append("Displacement that created it broke structure")
        return _cand(float(v.C[i]) if touching else ce, float(min(v.L[k - 1], v.L[k])), (zl, zh),
                     "market" if touching else "limit", notes, 26 + (6 if broke else 0))
    return None


@strategy("ote_pullback", "OTE Pullback", "SMC / ICT",
          "Trend leg that broke structure, then a 62-79% retracement (optimal trade entry) from the leg high.")
def d_ote_pullback(v: View, i: int):
    if i < 100:
        return None
    a = v.atr[i]
    w0 = i - 50
    bi = w0 + int(np.argmax(v.H[w0:i + 1]))
    if bi < i - 45:
        return None
    a0 = max(0, bi - 80)
    ai = a0 + int(np.argmin(v.L[a0:bi - 2])) if bi - 2 > a0 else None
    if ai is None or bi - ai < 4:
        return None
    leg_hi, leg_lo = float(v.H[bi]), float(v.L[ai])
    R = leg_hi - leg_lo
    if R < 2 * a or R > 8 * a or not v.bos[ai:bi + 1].any():
        return None
    retr = (leg_hi - v.C[i]) / R
    if not (0.0 <= retr <= 0.79) or v.C[i] <= leg_lo:
        return None
    entry = leg_hi - 0.705 * R
    in_zone = retr >= 0.62
    if not in_zone and (v.C[i] - entry) / a > 3.5:
        return None
    notes = [f"Impulse of {R / a:.1f} ATR broke structure; price has retraced {retr * 100:.0f}% of it"]
    return _cand(float(v.C[i]) if in_zone else entry, leg_lo, (leg_hi - 0.79 * R, leg_hi - 0.62 * R),
                 "market" if in_zone else "limit", notes, 30)


@strategy("asia_sweep", "Asian Range Sweep", "Session",
          "London takes the Asian low, closes back inside the range, and the move reverses toward the opposite side.")
def d_asia_sweep(v: View, i: int):
    ctx = v.ctx
    if not np.isfinite(v.asia_lo[i]) or ctx.hour[i] >= 17:
        return None
    A_lo, A_hi, a = float(v.asia_lo[i]), float(v.asia_hi[i]), v.atr[i]
    if not (0.6 * a <= A_hi - A_lo <= 8 * a):
        return None
    e = int(np.searchsorted(ctx.ep, ctx.uday[i] * 86400 + 7 * 3600))
    if e >= i:
        return None
    below = np.nonzero(v.L[e:i + 1] < A_lo)[0]
    if not len(below):
        return None
    k = e + int(below[0])
    if i - k > 16 or v.C[i] < A_lo or not (v.C[k:min(k + 3, i + 1)] > A_lo).any():
        return None
    ext = float(v.L[k:i + 1].min())
    z_hi = A_lo + 0.6 * a
    touching = v.L[i] <= z_hi
    kz = 7 <= ctx.hour[k] < 11
    t = v.t
    notes = [f"{t['alow']} {v.px(A_lo):.5g} swept{' inside the London killzone' if kz else ''} and reclaimed",
             f"Asian range is {(A_hi - A_lo) / a:.1f} ATR wide; the {t['ahigh']} ({v.px(A_hi):.5g}) is the draw on liquidity"]
    return _cand(float(v.C[i]) if touching else A_lo + 0.3 * a, ext, (A_lo, z_hi),
                 "market" if touching else "limit", notes, 32 + (4 if kz else 0), expire=12)


@strategy("pdhl_raid", "Prior-Day Level Raid", "Session",
          "Wick through the prior day's high/low that closes back inside; trade the rejection toward the other side of yesterday's range.")
def d_pdhl_raid(v: View, i: int):
    if not np.isfinite(v.pdl[i]) or i < 30:
        return None
    lvl, a = float(v.pdl[i]), v.atr[i]
    lo_i = max(0, i - 14)
    below = np.nonzero(v.L[lo_i:i + 1] < lvl)[0]
    if not len(below):
        return None
    k = lo_i + int(below[0])
    if v.C[i] <= lvl or (v.C[i] - lvl) / a > 2.5 or not (v.C[k:min(k + 3, i + 1)] > lvl).any():
        return None
    ext = float(v.L[k:i + 1].min())
    z_hi = lvl + 0.3 * a
    touching = v.L[i] <= z_hi
    notes = [f"Prior-day extreme {v.px(lvl):.5g} raided {i - k} bars ago and reclaimed"]
    return _cand(float(v.C[i]) if touching else z_hi, ext, (lvl, z_hi),
                 "market" if touching else "limit", notes, 30, expire=12)


@strategy("ema_pullback", "EMA Trend Pullback", "Trend",
          "EMA 21/50/200 stacked, price tags the 21 EMA with an RSI reset and a rejection candle; enter on the close.")
def d_ema_pullback(v: View, i: int):
    if i < WARMUP:
        return None
    a = v.atr[i]
    if not (v.e21[i] > v.e50[i] > v.e200[i]) or v.e50[i] - v.e50[i - 10] < 0.1 * a:
        return None
    if v.L[i - 4:i + 1].min() > v.e21[i] + 0.15 * a or v.C[i] <= v.e50[i]:
        return None
    rng = v.H[i] - v.L[i]
    wick = (min(v.O[i], v.C[i]) - v.L[i]) / rng if rng > 0 else 0
    if not (v.C[i] > v.O[i] and (v.C[i] > v.H[i - 1] or wick >= 0.4)):
        return None
    if not (38 <= v.rsi[i] <= 64 and v.rsi[i] > v.rsi[i - 1]):
        return None
    return _cand(float(v.C[i]), float(v.L[i - 6:i + 1].min()), (float(v.e21[i] - 0.3 * a), float(v.e21[i] + 0.3 * a)),
                 "market", [f"EMA stack {v.t['stack']}", f"Pullback to the 21 EMA rejected, RSI turning {v.t['turn']}"], 28,
                 expire=4)


@strategy("squeeze_breakout", "Squeeze Breakout", "Volatility",
          "Bollinger Bands inside Keltner Channels for 8+ bars, then a momentum close beyond the band as the squeeze releases.")
def d_squeeze(v: View, i: int):
    ctx = v.ctx
    if i < 60 or ctx.sq_on[i] or not ctx.sq_on[i - 3:i].any() or ctx.sq_on[i - 12:i].sum() < 7:
        return None
    a = v.atr[i]
    if not (v.C[i] > v.bb_up[i] and v.C[i] - v.O[i] >= 0.7 * a and v.C[i] > v.sma[i]):
        return None
    return _cand(float(v.C[i]), float(v.L[i - 2:i + 1].min()), (float(v.sma[i]), float(v.C[i])), "market",
                 ["Volatility squeeze released with a momentum close beyond the band"], 26, expire=3)


# --------------------------------------------------------------------------- #
# Stops, targets, simulation
# --------------------------------------------------------------------------- #

def _finalize(v: View, i: int, cand: dict) -> Optional[dict]:
    """Long-space stop tiers + staged targets. None if the structure is not tradeable intraday."""
    a, E, ref = float(v.atr[i]), cand["entry"], cand["stop_ref"]
    stops = {k: min(ref - m * a, E - MIN_RISK_ATR * a) for k, m in STOP_TIERS.items()}
    risk = E - stops["standard"]
    if risk <= 0 or risk > MAX_RISK_ATR * a:
        return None
    lo = max(0, i - 400)
    levels = []
    k = int(np.searchsorted(v.hi_conf, i, side="right"))
    for j in range(k):
        if v.hi_idx[j] >= lo:
            levels.append((float(v.hi_px[j]), f"swing {v.t['hi']}"))
    for arr, label in ((v.pdh, v.t["pdx"]), (v.asia_hi, v.t["ahigh"]), (v.day_hi, f"session {v.t['hi']}")):
        if np.isfinite(arr[i]):
            levels.append((float(arr[i]), label))
    levels.append((float(v.H[max(0, i - 96):i + 1].max()), f"24h {v.t['hi']}"))
    levels = sorted({round(p, 8): s for p, s in levels if p > E}.items())

    def pick(min_rr, max_rr, after):
        for px, src in levels:
            rr = (px - E) / risk
            if min_rr <= rr <= max_rr and px > after + 0.3 * risk:
                return px, src
        return None

    t1 = pick(1.0, 2.3, E) or (E + 1.5 * risk, "1.5R projection")
    r1 = (t1[0] - E) / risk
    t2 = pick(max(2.0, r1 + 0.7), 4.5, t1[0]) or (E + max(2.5, r1 + 1.0) * risk, "2.5R projection")
    r2 = (t2[0] - E) / risk
    t3 = pick(max(3.5, r2 + 0.8), 8.0, t2[0]) or (E + max(4.0, r2 + 1.2) * risk, "4R projection")
    return {"stops": stops, "risk": risk, "targets": [t1, t2, t3]}


def _simulate(ctx: Ctx, i: int, s: int, etype: str, entry: float, stop: float, tps: list, expire: int):
    """Result in R for one setup, or None if it never filled / is unresolved. Conservative bar logic."""
    h, l, c, n = ctx.h, ctx.l, ctx.c, ctx.n
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    if etype == "market":
        j0 = i
    else:
        j0 = None
        for j in range(i + 1, min(n, i + 1 + expire)):
            if (s > 0 and l[j] <= entry) or (s < 0 and h[j] >= entry):
                j0 = j
                break
        if j0 is None:
            return None
    if (s > 0 and l[j0] <= stop and j0 != i) or (s < 0 and h[j0] >= stop and j0 != i):
        return -1.0
    weights = list(SCALE_OUT)
    cur_stop, realised, hit = stop, 0.0, 0
    end = min(n, j0 + 1 + MAX_HOLD_BARS)
    for j in range(j0 + 1, end):
        if (s > 0 and l[j] <= cur_stop) or (s < 0 and h[j] >= cur_stop):
            return realised + sum(weights) * ((cur_stop - entry) * s / risk)
        while hit < 3 and ((s > 0 and h[j] >= tps[hit]) or (s < 0 and l[j] <= tps[hit])):
            realised += weights[hit] * (abs(tps[hit] - entry) / risk)
            weights[hit] = 0.0
            hit += 1
            if hit == 1:
                cur_stop = entry
        if hit == 3:
            return realised
    if end - (j0 + 1) < MAX_HOLD_BARS:
        return None                                     # ran out of data before it resolved
    return realised + sum(weights) * ((c[end - 1] - entry) * s / risk)


def backtest(ctx: Ctx, keys: list[str], start: int = WARMUP, cooldown: int = 6, strict: bool = True) -> dict:
    """Walk-forward stats per strategy and side over the candles we hold."""
    out = {}
    last = ctx.n - 2
    for key in keys:
        fn = STRATEGIES[key]["fn"]
        for s in (1, -1):
            v = ctx.views[s]
            rs = []
            i = start
            while i < last:
                cand = fn(v, i)
                if cand and strict and not direction_gate(v, i, key)[0]:
                    cand = None
                fin = _finalize(v, i, cand) if cand else None
                if fin:
                    tps = [s * t[0] for t in fin["targets"]]
                    r = _simulate(ctx, i, s, cand["etype"], s * cand["entry"], s * fin["stops"]["standard"],
                                  tps, cand["expire"])
                    if r is not None:
                        rs.append(r)
                    i += cooldown
                i += 1
            out[(key, s)] = _stats(rs)
    return out


def _stats(rs: list) -> dict:
    n = len(rs)
    if not n:
        return {"n": 0}
    wins = [r for r in rs if r > 0]
    loss = [-r for r in rs if r <= 0]
    return {"n": n, "win_rate": round(len(wins) / n * 100, 1), "avg_r": round(float(sum(rs) / n), 2),
            "profit_factor": round(float(sum(wins) / sum(loss)), 2) if loss and sum(loss) > 0 else None}


# --------------------------------------------------------------------------- #
# Session labels, scoring, public entry point
# --------------------------------------------------------------------------- #

def session_label(hour: float) -> tuple[str, bool]:
    if 7 <= hour < 10:
        return "London killzone", True
    if 12 <= hour < 15:
        return "New York killzone", True
    if hour < 7:
        return "Asia", False
    if 10 <= hour < 12:
        return "London midday", False
    if 15 <= hour < 17:
        return "London close", False
    if 17 <= hour < 21:
        return "NY afternoon", False
    return "Rollover", False


def _name(b: int) -> str:
    return "bullish" if b > 0 else "bearish" if b < 0 else "neutral"


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _build(ctx: Ctx, key: str, s: int, i: int, cand: dict, fin: dict, external: dict, strict: bool = True) -> dict:
    v = ctx.views[s]
    sym = ctx.symbol
    a = float(ctx.atr[i])
    E = s * cand["entry"]
    rd = lambda p: round_price(sym, p)
    stops = {k: s * p for k, p in fin["stops"].items()}
    targets = [{"price": rd(s * t[0]), "rr": round((t[0] - cand["entry"]) / fin["risk"], 2),
                "pips": to_pips(sym, t[0] - cand["entry"]), "source": t[1]} for t in fin["targets"]]
    direction = "bullish" if s > 0 else "bearish"
    hour = float(ctx.hour[i])
    session, kz = session_label(hour)
    b4, b1, b15 = (int(arr[i]) for arr in v.bias)
    discount = bool(v.C[i] < v.eq[i])
    reasons = list(cand["notes"])
    warns = []
    score = float(cand["base"])

    for label, b, pts in (("4h", b4, 12), ("1h", b1, 12)):
        if b > 0:
            score += pts
            reasons.append(f"{label} structure is {direction}")
        elif b < 0:
            score -= pts + 2
            warns.append(f"Against {label} structure ({_name(s * b)} trend)")
    if b15 > 0:
        score += 6
        reasons.append("15m structure agrees")
    if discount:
        score += 6
        reasons.append("Entry from discount of the 24h range" if s > 0 else "Entry from premium of the 24h range")
    else:
        score -= 8
        warns.append("Buying in the premium half of the 24h range" if s > 0 else "Selling in the discount half of the 24h range")
    if kz and key != "asia_sweep":
        score += 8
        reasons.append(f"Inside the {session}")
    elif not kz and key != "asia_sweep" and session in ("Asia", "Rollover"):
        score -= 4
    if session in ("Rollover",):
        warns.append("Rollover hours: spreads are wide, avoid new entries")
    rr2 = targets[1]["rr"]
    score += 6 if rr2 >= 2.5 else 0
    if targets[0]["rr"] < 1.2:
        score -= 6
    risk_atr = fin["risk"] / a
    if 0.8 <= risk_atr <= 2.2:
        score += 4
    if risk_atr > 2.4:
        warns.append(f"Stop is {risk_atr:.1f} ATR: wide for intraday, size down")

    al = trend_alignment(v, i)[0]
    label = "with trend" if al >= 0.34 else "mixed" if al > ALIGN_OK else "counter-trend"
    if strict:
        who = "buyers" if s > 0 else "sellers"
        if al >= 0.5:
            score += 6
            reasons.append("Trading with the higher-timeframe trend")
        elif label == "counter-trend":
            score -= 6
            warns.append("Counter-trend reversal from an extreme: size down and take TP1 early")
        if cand["etype"] == "market":
            if _rejection(v, i):
                score += 4
                reasons.append(f"Signal candle closed with {who} in control")
            else:
                score -= 8
                warns.append(f"No rejection candle yet: wait for a 15m close with {who} in control before entering")

    bonus, note, agrees = external["align"](direction)
    score += bonus
    if agrees is True:
        reasons.append(note)
    elif agrees is False:
        warns.append(note)
    if external.get("blackout"):
        score -= 30
        warns.append(external["blackout_note"])
    elif external.get("news_soon"):
        warns.append(external["news_soon"])
        score -= 6

    adr = float(ctx.adr[i]) if np.isfinite(ctx.adr[i]) else None
    used = None
    if adr:
        used = round((float(ctx.day_hi[i]) - float(ctx.day_lo[i])) / adr * 100)
        if used >= 90:
            score -= 6
            warns.append(f"Today's range is already {used}% of the average daily range")
        elif (fin["targets"][2][0] - cand["entry"]) > max(adr - (ctx.day_hi[i] - ctx.day_lo[i]), 0) + 0.3 * adr:
            warns.append("TP3 sits beyond the range left in today's average daily move")

    bar_close = float(ctx.ep[i] + BAR_SEC)
    live = ctx.live_price
    chase = abs(live - E) / fin["risk"] if cand["etype"] == "market" else None
    if chase is not None and chase > 0.5:
        warns.append(f"Price has moved {chase:.1f}R from the signal close; re-check before entering")
    zone = sorted((s * cand["zone"][0], s * cand["zone"][1]))
    return {
        "id": f"{sym}-{key}-{direction}-{int(bar_close)}", "symbol": sym, "strategy": key,
        "strategy_name": STRATEGIES[key]["name"], "family": STRATEGIES[key]["family"],
        "direction": direction, "side": "BUY" if s > 0 else "SELL",
        "status": "ready_now" if cand["etype"] == "market" else "pending",
        "entry_type": cand["etype"], "entry": rd(E), "zone": [rd(zone[0]), rd(zone[1])],
        "stops": {k: {"price": rd(p), "pips": to_pips(sym, E - p), "atr": round(abs(E - p) / a, 2)} for k, p in stops.items()},
        "stop": rd(stops["standard"]), "targets": targets,
        "risk_pips": to_pips(sym, fin["risk"]), "atr_pips": to_pips(sym, a),
        "breakeven": rd(E + s * fin["risk"]),
        "trail": "Close 50% at TP1 and move the stop to entry; close 30% at TP2; trail the last 20% under the newest 15m swing.",
        "valid_until": _iso(min(bar_close + cand["expire"] * BAR_SEC, bar_close + 6 * 3600)),
        "close_by": _iso(bar_close + 48 * BAR_SEC),
        "confluence": reasons, "warnings": warns,
        "htf": {"h4": _name(s * b4), "h1": _name(s * b1), "m15": _name(s * b15)},
        "session": session, "killzone": kz, "zone_pd": "discount" if ctx.c[i] < ctx.eq[i] else "premium",
        "adr_used_pct": used, "live_price": rd(live),
        "trend": {"alignment": round(al, 2), "label": label},
        "bar_time": _iso(float(ctx.ep[i])), "age_bars": int(ctx.n - 1 - i),
        "_score": score,
    }


def scan_symbol(symbol: str, candles: dict, external: dict, strategies: Optional[list[str]] = None,
                with_backtest: bool = True, strict: bool = True) -> dict:
    """Run every selected strategy on one symbol. `external` carries the DXY and news context."""
    keys = [k for k in (strategies or list(STRATEGIES)) if k in STRATEGIES]
    if candles.get("15m") is None or len(candles["15m"]) < WARMUP + 40:
        return {"symbol": symbol.upper(), "setups": [], "note": "Not enough 15m history to scan this market yet."}
    ctx = Ctx(symbol, candles)
    if ctx.n < WARMUP + 40:
        return {"symbol": symbol.upper(), "setups": [], "note": "Not enough closed 15m bars to scan this market yet."}
    i = ctx.n - 1
    stats = backtest(ctx, keys, strict=strict) if with_backtest else {}
    setups = []
    suppressed = []

    def _drop(key, s, reason):
        suppressed.append({"strategy": key, "strategy_name": STRATEGIES[key]["name"],
                           "direction": "bullish" if s > 0 else "bearish", "reason": reason})

    for key in keys:
        for s in (1, -1):
            v = ctx.views[s]
            cand = STRATEGIES[key]["fn"](v, i)
            if cand and strict:
                ok, why, _ = direction_gate(v, i, key)
                if not ok:
                    _drop(key, s, why)
                    continue
            fin = _finalize(v, i, cand) if cand else None
            if not fin:
                continue
            setup = _build(ctx, key, s, i, cand, fin, external, strict)
            st = stats.get((key, s))
            if st and st.get("n", 0) >= 1:
                both = stats.get((key, -s)) or {"n": 0}
                pooled = _pool(st, both)
                use = st if st["n"] >= 6 else pooled
                setup["stats"] = {**use, "scope": "this side" if use is st else "both sides"}
                if strict and use["n"] >= 10 and use["avg_r"] < -0.15:
                    _drop(key, s, f"Strategy has been losing on this market ({use['avg_r']:+.2f}R over {use['n']} trades)")
                    continue
                if use["n"] >= 8:
                    if use["avg_r"] >= 0.25:
                        setup["_score"] += 8
                        setup["confluence"].append(f"Walk-forward edge: {use['win_rate']}% wins, {use['avg_r']:+.2f}R average over {use['n']} trades")
                    elif use["avg_r"] > 0:
                        setup["_score"] += 3
                    elif use["avg_r"] < -0.1:
                        setup["_score"] -= 10
                        setup["warnings"].append(f"This strategy has been losing on this market recently ({use['avg_r']:+.2f}R over {use['n']} trades)")
                else:
                    setup["warnings"].append(f"Thin sample ({use['n']} trades) behind the statistics")
            else:
                setup["stats"] = None
            setups.append(setup)

    # Stacking: several independent strategies agreeing on the same side is the strongest tell.
    for side in ("bullish", "bearish"):
        group = [x for x in setups if x["direction"] == side]
        if len(group) > 1:
            bump = min(12, 5 * (len(group) - 1))
            for x in group:
                x["_score"] += bump
                x["confluence"].append(f"{len(group)} independent strategies agree on {side} here")

    # Both directions live on one market is the classic "it went the other way" trap: keep the side the
    # trend and the scoring clearly favour, and stand aside when neither is clearly stronger.
    if strict:
        bull = [x for x in setups if x["direction"] == "bullish"]
        bear = [x for x in setups if x["direction"] == "bearish"]
        if bull and bear:
            def strength(g):
                return max(x["_score"] + 15 * x["trend"]["alignment"] for x in g)
            sb, ss = strength(bull), strength(bear)
            if abs(sb - ss) >= CONFLICT_GAP:
                loser, reason = (bear, f"Bullish side is stronger here ({sb:.0f} vs {ss:.0f})") if sb > ss \
                    else (bull, f"Bearish side is stronger here ({ss:.0f} vs {sb:.0f})")
            else:
                loser, reason = bull + bear, "Both directions are live with no clear edge: standing aside"
            for x in loser:
                suppressed.append({"strategy": x["strategy"], "strategy_name": x["strategy_name"],
                                   "direction": x["direction"], "reason": reason})
            setups = [x for x in setups if x not in loser]

    kept = []
    for x in setups:
        x["score"] = round(max(0.0, min(100.0, x.pop("_score"))), 1)
        x["grade"] = grade_for(x["score"])
        if strict and x["score"] < STRICT_MIN_SCORE:
            suppressed.append({"strategy": x["strategy"], "strategy_name": x["strategy_name"],
                               "direction": x["direction"], "reason": "Below the minimum quality score"})
            continue
        kept.append(x)
    setups = kept
    setups.sort(key=lambda x: x["score"], reverse=True)
    return {"symbol": ctx.symbol, "setups": setups, "bars": ctx.n,
            "last_bar": _iso(float(ctx.ep[-1])), "live_price": round_price(ctx.symbol, ctx.live_price),
            "filter": {"strict": strict, "count": len(suppressed), "suppressed": suppressed[:12]}}


def _pool(a: dict, b: dict) -> dict:
    n = a.get("n", 0) + b.get("n", 0)
    if not n:
        return {"n": 0}
    avg = (a.get("avg_r", 0) * a.get("n", 0) + b.get("avg_r", 0) * b.get("n", 0)) / n
    win = (a.get("win_rate", 0) * a.get("n", 0) + b.get("win_rate", 0) * b.get("n", 0)) / n
    return {"n": n, "win_rate": round(win, 1), "avg_r": round(avg, 2), "profit_factor": None}


def catalog() -> list[dict]:
    return [{k: v[k] for k in ("key", "name", "family", "desc")} for v in STRATEGIES.values()]
