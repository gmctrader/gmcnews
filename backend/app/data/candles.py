"""
backend/app/data/candles.py  —  drop-in replacement for the yfinance-only feed.

Provider chain (first that works wins):
  1. Twelve Data  (official API, key required: TWELVEDATA_API_KEY)  FX, metals, crypto
  2. Yahoo Finance via yfinance (no key)                            safety net only
  3. Stale cache (<= MAX_STALE seconds), labelled in df.attrs["source"]

DXY: built from the six ICE component pairs (official weights) on Twelve Data,
falling back to Yahoo's DX-Y.NYB. No fake/synthetic price series is ever invented;
if every provider fails, CandleUnavailable is raised so the engine can withhold a plan.

Interface (unchanged):
    get_candles(symbol)  -> {"1h": DataFrame, "15m": DataFrame}
    get_dxy_candles()    -> {"1h": DataFrame, "15m": DataFrame}
DataFrame: UTC DatetimeIndex, columns open/high/low/close/volume (see COLUMN_STYLE).
df.attrs["source"] tells you which provider produced it.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

import pandas as pd
import requests

log = logging.getLogger("candles")

# ── config ────────────────────────────────────────────────────────────────
COLUMN_STYLE = os.getenv("CANDLE_COLUMNS", "lower")   # "lower" -> close ; "title" -> Close (yfinance style)
TD_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
TD_URL = "https://api.twelvedata.com/time_series"
TD_PER_MIN = int(os.getenv("TD_CREDITS_PER_MIN", "8"))     # free plan: 8/min
TD_PER_DAY = int(os.getenv("TD_CREDITS_PER_DAY", "780"))   # free plan: 800/day, keep headroom
HTTP_TIMEOUT = 8
MAX_STALE = int(os.getenv("CANDLE_MAX_STALE", "7200"))     # serve cached bars up to 2h old if all providers fail

TTL = {"15m": 300, "1h": 1200}      # cache seconds per timeframe
DXY_TTL = 900
BARS = {"15m": 500, "1h": 300}
TD_INTERVAL = {"15m": "15min", "1h": "1h"}
YF_PERIOD = {"15m": "30d", "1h": "60d"}

FIAT = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "MXN",
        "ZAR", "SGD", "HKD", "TRY", "PLN", "CNH"}
METALS = {"XAU", "XAG", "XPT", "XPD"}
CRYPTO = {"BTC", "ETH", "SOL", "XRP", "LTC", "BNB", "ADA", "DOGE"}

# ICE DXY: 50.14348112 * EURUSD^-0.576 * USDJPY^0.136 * GBPUSD^-0.119 * USDCAD^0.091 * USDSEK^0.042 * USDCHF^0.036
DXY_K = 50.14348112
DXY_W = {"EUR/USD": -0.576, "USD/JPY": 0.136, "GBP/USD": -0.119,
         "USD/CAD": 0.091, "USD/SEK": 0.042, "USD/CHF": 0.036}


class CandleUnavailable(RuntimeError):
    """Every provider failed and no usable cache exists."""


# ── symbol mapping ────────────────────────────────────────────────────────
def _norm(sym: str) -> str:
    return sym.upper().replace("/", "").replace("_", "").replace("=X", "").replace("-", "").strip()


def to_td_symbol(sym: str) -> str | None:
    s = _norm(sym)
    if len(s) == 6 and s[:3] in (FIAT | METALS | CRYPTO) and s[3:] in FIAT:
        return f"{s[:3]}/{s[3:]}"
    return None


def to_yahoo_symbol(sym: str) -> str:
    s = _norm(sym)
    if s in ("DXY", "USDX"):
        return "DX-Y.NYB"
    if s == "XAUUSD":
        return "GC=F"
    if s == "XAGUSD":
        return "SI=F"
    if len(s) == 6 and s[:3] in CRYPTO:
        return f"{s[:3]}-{s[3:]}"
    if len(s) == 6 and s[:3] in FIAT and s[3:] in FIAT:
        return f"{s}=X"
    return sym


# ── Twelve Data credit governor (never blows the free quota) ──────────────
class _Budget:
    def __init__(self):
        self.lock = threading.Lock()
        self.day = None
        self.used = 0
        self.window: list[float] = []

    def take(self, n: int) -> bool:
        now = time.time()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.lock:
            if self.day != today:
                self.day, self.used = today, 0
            self.window = [t for t in self.window if now - t < 60]
            if self.used + n > TD_PER_DAY or len(self.window) + n > TD_PER_MIN:
                return False
            self.used += n
            self.window.extend([now] * n)
            return True


_budget = _Budget()
_cache: dict[tuple, tuple[float, pd.DataFrame]] = {}
_cache_lock = threading.Lock()


# ── Twelve Data ───────────────────────────────────────────────────────────
def _parse_td_block(block) -> pd.DataFrame:
    if not isinstance(block, dict) or block.get("status") == "error" or not block.get("values"):
        msg = block.get("message") if isinstance(block, dict) else "bad payload"
        raise CandleUnavailable(f"twelvedata: {msg}")
    df = pd.DataFrame(block["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) if "volume" in df else 0.0
    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
    df.index.name = None
    return df


def _td_request(symbols: list[str], tf: str) -> dict[str, pd.DataFrame]:
    if not TD_KEY:
        raise CandleUnavailable("twelvedata: no TWELVEDATA_API_KEY set")
    if not _budget.take(len(symbols)):
        raise CandleUnavailable("twelvedata: local credit budget exhausted")
    r = requests.get(
        TD_URL,
        params={"symbol": ",".join(symbols), "interval": TD_INTERVAL[tf], "outputsize": BARS[tf],
                "timezone": "UTC", "order": "ASC", "apikey": TD_KEY},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error" and "values" not in data \
            and not any("/" in k for k in data):
        raise CandleUnavailable(f"twelvedata: {data.get('message', data)}")
    blocks = {symbols[0]: data} if len(symbols) == 1 else data
    return {s: _parse_td_block(blocks.get(s)) for s in symbols}


def _td_single(td_sym: str, tf: str) -> pd.DataFrame:
    return _td_request([td_sym], tf)[td_sym]


def _dxy_from_components(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = {}
    for field in ("open", "high", "low", "close"):
        cols[field] = pd.concat({s: frames[s][field] for s in DXY_W}, axis=1).ffill(limit=2).dropna()
    idx = cols["close"].index.intersection(cols["open"].index)
    out = pd.DataFrame(index=idx)
    for field in ("open", "high", "low", "close"):
        c = cols[field].loc[idx]
        v = DXY_K
        for s, w in DXY_W.items():
            v = v * (c[s] ** w)
        out[field] = v
    # components never peak simultaneously: keep OHLC internally consistent
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    out["volume"] = 0.0
    return out


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()


def _td_dxy_15m(_tf=None) -> pd.DataFrame:
    return _dxy_from_components(_td_request(list(DXY_W), "15m"))


def _td_dxy(tf: str) -> pd.DataFrame:
    m15 = _get("DXY-TD", "15m", [("twelvedata-dxy", _td_dxy_15m)], ttl=DXY_TTL)
    return m15 if tf == "15m" else _resample(m15, "1h")


# ── Yahoo fallback ────────────────────────────────────────────────────────
def _yahoo(ticker: str, tf: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(ticker, period=YF_PERIOD[tf], interval="15m" if tf == "15m" else "1h",
                      progress=False, auto_adjust=False, threads=False)
    if raw is None or raw.empty:
        raise CandleUnavailable(f"yahoo: empty for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    df = raw[["open", "high", "low", "close"]].copy()
    df["volume"] = raw["volume"] if "volume" in raw else 0.0
    df.index = pd.to_datetime(df.index, utc=True) if df.index.tz is None else df.index.tz_convert("UTC")
    df.index.name = None
    return df.dropna(subset=["open", "high", "low", "close"])


# ── generic cached loader with fallback + stale-serve ─────────────────────
def _get(key: str, tf: str, loaders: list, ttl: int | None = None) -> pd.DataFrame:
    ck = (key, tf)
    ttl = TTL[tf] if ttl is None else ttl
    with _cache_lock:
        hit = _cache.get(ck)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    errors = []
    for name, fn in loaders:
        try:
            df = fn(tf)
            if df is not None and len(df) >= 30:
                df = df.copy()
                df.attrs["source"] = name
                with _cache_lock:
                    _cache[ck] = (time.time(), df)
                return df
            errors.append(f"{name}: too few bars")
        except Exception as e:  # noqa: BLE001 — any provider failure falls through
            errors.append(f"{name}: {e}")
            log.warning("candles %s %s via %s failed: %s", key, tf, name, e)
    if hit and time.time() - hit[0] < MAX_STALE:
        stale = hit[1].copy()
        stale.attrs["source"] = str(hit[1].attrs.get("source", "?")) + " (stale)"
        return stale
    raise CandleUnavailable(f"{key} {tf}: " + " | ".join(errors))


def _style(df: pd.DataFrame) -> pd.DataFrame:
    if COLUMN_STYLE == "title":
        src = df.attrs.get("source")
        df = df.rename(columns=str.capitalize)
        df.attrs["source"] = src
    return df


# ── public API ────────────────────────────────────────────────────────────
def get_candles(symbol: str) -> dict[str, pd.DataFrame]:
    td_sym, y_sym = to_td_symbol(symbol), to_yahoo_symbol(symbol)
    key = _norm(symbol)
    out = {}
    for tf in ("1h", "15m"):
        loaders = []
        if td_sym and TD_KEY:
            loaders.append(("twelvedata", lambda t, s=td_sym: _td_single(s, t)))
        loaders.append(("yahoo", lambda t, s=y_sym: _yahoo(s, t)))
        out[tf] = _style(_get(key, tf, loaders))
    return out


def get_dxy_candles() -> dict[str, pd.DataFrame]:
    out = {}
    for tf in ("1h", "15m"):
        loaders = []
        if TD_KEY:
            loaders.append(("twelvedata-dxy", _td_dxy))
        loaders.append(("yahoo", lambda t: _yahoo("DX-Y.NYB", t)))
        out[tf] = _style(_get("DXY", tf, loaders, ttl=DXY_TTL))
    return out
