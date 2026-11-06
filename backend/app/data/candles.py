"""
Candle data provider.

This engine was originally MT5-side (broker-connected). For the web
app, we need *some* live price source with zero setup, so this module
uses yfinance (free, no API key) as a fallback provider. When you're
ready to go live with real MT5/broker candles, drop a new provider in
here with the same function signature and swap PROVIDER below -- every
other module (fundamental.py, market_structure.py, confluence.py)
already just consumes a plain {"1h": df, "15m": df} dict and doesn't
care where it came from.

DXY itself has no clean free ticker on yfinance, so we use "DX-Y.NYB"
where available and fall back to a synthetic index built from the six
basket currencies, matching the "synthetic" path fundamental.py's
docstring already anticipates.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

import httpx
import pandas as pd
import yfinance as yf

log = logging.getLogger("candles")

# yfinance interval/period pairs that keep us under its lookback limits.
_INTERVALS = {
    "1h": ("60m", "1mo"),
    "15m": ("15m", "5d"),
}

# --------------------------------------------------------------------
# In-memory TTL cache for raw per-ticker downloads.
#
# Every page load used to trigger a fresh, sequential yfinance download
# for every ticker/interval combo -- for the default 9-symbol watchlist
# plus the DXY basket fallback, that's 20-30+ blocking network calls on
# every single request, which is what made the app feel like it hung on
# "Loading market data...". This cache means each distinct ticker/interval
# is only ever actually downloaded once per `_TTL_SECONDS` window; every
# request in between (including the 9 symbols reusing shared basket
# tickers, and repeat page loads) is served from memory instantly.
#
# Not persisted to disk on purpose -- candle data goes stale fast enough
# that surviving a restart isn't useful, and keeping it in-process avoids
# any serialization cost for DataFrames.
# --------------------------------------------------------------------
_TTL_SECONDS = 90
_cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}


def _cached_download(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    key = (ticker, interval)
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < _TTL_SECONDS:
        return hit[1]

    try:
        raw = yf.download(ticker, interval=interval, period=period,
                           progress=False, auto_adjust=True)
    except Exception:
        return hit[1] if hit else None  # serve stale data over a hard failure if we have it

    if raw is None or raw.empty:
        return hit[1] if hit else None

    df = _normalize(raw)
    _cache[key] = (now, df)
    return df

# Maps our internal symbol names to Yahoo Finance tickers.
_YF_TICKERS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X", "USDJPY": "USDJPY=X", "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X", "XAUUSD": "GC=F", "XAGUSD": "SI=F",
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD",
    "USOIL": "CL=F", "UKOIL": "BZ=F",
    # Broad equity indices (Yahoo symbols; kept distinct from USD FX pairs).
    "US500": "^GSPC", "SPX500": "^GSPC", "NAS100": "^NDX",
    "US100": "^NDX", "DJ30": "^DJI", "US30": "^DJI",
    "UK100": "^FTSE", "GER40": "^GDAXI", "DE40": "^GDAXI",
    "JP225": "^N225",
}

DXY_TICKER = "DX-Y.NYB"

# DXY basket weights (ICE formula), used only if the direct ticker
# is unavailable and we need to build a synthetic index.
_DXY_BASKET = {
    "EURUSD=X": -0.576, "USDJPY=X": 0.136, "GBPUSD=X": -0.119,
    "USDCAD=X": 0.091, "USDSEK=X": 0.042, "USDCHF=X": 0.036,
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Recent yfinance versions return MultiIndex columns like
    # ("High", "DX-Y.NYB") even for a single ticker. Flatten to the
    # plain field name first, or the rename below silently no-ops and
    # df["high"] ends up returning a DataFrame instead of a Series --
    # which is what produces "truth value of a Series is ambiguous"
    # several layers downstream in market_structure.py.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    time_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={
        time_col: "time", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df[["time", "open", "high", "low", "close"]]



# ====================================================================
# Twelve Data layer (primary). Yahoo above stays as the fallback.
#
# Free plan = 8 credits/min, 800/day, 1 credit per symbol per request.
# To stay inside that we fetch ONLY 15m bars (1000 of them) per symbol
# and derive the 1h frame by resampling, cache for 10 minutes, share
# the cache between the watchlist and the DXY basket, and count credits
# locally so we drop to Yahoo *before* Twelve Data would ever 429.
# ====================================================================
TD_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
_TD_URL = "https://api.twelvedata.com/time_series"
_TD_PER_MIN = int(os.getenv("TD_CREDITS_PER_MIN", "8"))
_TD_PER_DAY = int(os.getenv("TD_CREDITS_PER_DAY", "780"))
_TD_TTL = int(os.getenv("TD_CACHE_SECONDS", "600"))
_TD_BARS = 1000
_MIN_BARS = 200
_MAX_STALE = 7200

_FIAT = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK",
         "MXN", "ZAR", "SGD", "HKD", "TRY", "PLN"}
_METALS = {"XAU", "XAG", "XPT", "XPD"}
_CRYPTO = {"BTC", "ETH", "SOL", "XRP", "LTC", "BNB", "ADA", "DOGE"}
_DXY_TD = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CAD", "USD/SEK", "USD/CHF"]


def _to_td(symbol: str) -> str | None:
    s = symbol.upper().replace("/", "")
    if len(s) == 6 and s[:3] in (_FIAT | _METALS | _CRYPTO) and s[3:] in _FIAT:
        return f"{s[:3]}/{s[3:]}"
    return None


class _Budget:
    def __init__(self):
        self.lock = threading.Lock()
        self.day, self.used, self.window = None, 0, []

    def take(self, n: int) -> bool:
        now = time.time()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.lock:
            if self.day != today:
                self.day, self.used = today, 0
            self.window = [t for t in self.window if now - t < 60]
            if self.used + n > _TD_PER_DAY or len(self.window) + n > _TD_PER_MIN:
                return False
            self.used += n
            self.window.extend([now] * n)
            return True


_budget = _Budget()
_td_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_last_good: dict[str, tuple[float, dict, str]] = {}
_source_of: dict[str, str] = {}   # symbol -> provider name actually used last


def provider_name(symbol: str) -> str:
    """Provider that produced the most recent candles for this symbol."""
    return _source_of.get(symbol.upper(), "Yahoo Finance")


def _parse_td_block(block) -> pd.DataFrame | None:
    if not isinstance(block, dict) or block.get("status") == "error" or not block.get("values"):
        return None
    df = pd.DataFrame(block["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["open", "high", "low", "close"]].dropna()
    df.index.name = None
    return df


def _td_fetch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    if not TD_KEY:
        raise RuntimeError("no TWELVEDATA_API_KEY")
    if not _budget.take(len(symbols)):
        raise RuntimeError("local Twelve Data credit budget exhausted")
    r = httpx.get(_TD_URL, timeout=8.0, params={
        "symbol": ",".join(symbols), "interval": "15min", "outputsize": _TD_BARS,
        "timezone": "UTC", "order": "ASC", "apikey": TD_KEY})
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("status") == "error" and "values" not in data \
            and not any("/" in k for k in data):
        raise RuntimeError(str(data.get("message", data)))
    blocks = {symbols[0]: data} if len(symbols) == 1 else data
    out = {}
    for s in symbols:
        df = _parse_td_block(blocks.get(s))
        if df is not None and len(df) >= _MIN_BARS:
            out[s] = df
    if not out:
        raise RuntimeError("Twelve Data returned no usable bars")
    return out


def _td_base(symbols: list[str]) -> dict[str, pd.DataFrame]:
    now = time.time()
    out, missing = {}, []
    for s in symbols:
        hit = _td_cache.get(s)
        if hit and now - hit[0] < _TD_TTL:
            out[s] = hit[1]
        else:
            missing.append(s)
    if missing:
        try:
            for s, df in _td_fetch(missing).items():
                _td_cache[s] = (now, df)
                out[s] = df
        except Exception as e:  # noqa: BLE001
            # Quota/timeout: prefer slightly stale Twelve Data bars (<= 2x TTL)
            # over switching provider mid-scan; they refresh on the next request.
            log.warning("Twelve Data refresh skipped (%s); using recent cache", e)
            for s in missing:
                hit = _td_cache.get(s)
                if hit and now - hit[0] < 2 * _TD_TTL:
                    out[s] = hit[1]
            if any(s not in out for s in symbols):
                raise
    return out


def _resample_1h(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last"}).dropna()


def _to_engine(df: pd.DataFrame) -> pd.DataFrame:
    """DatetimeIndex frame -> the engine's format: RangeIndex + 'time' column."""
    return df[["open", "high", "low", "close"]].rename_axis("time").reset_index()


def _both(base: pd.DataFrame) -> dict:
    return {"1h": _to_engine(_resample_1h(base)), "15m": _to_engine(base)}


def _dxy_from_components(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    weights = {"EUR/USD": -0.576, "USD/JPY": 0.136, "GBP/USD": -0.119,
               "USD/CAD": 0.091, "USD/SEK": 0.042, "USD/CHF": 0.036}
    fields = {}
    for f in ("open", "high", "low", "close"):
        fields[f] = pd.concat({s: frames[s][f] for s in weights}, axis=1).ffill(limit=2).dropna()
    idx = fields["close"].index.intersection(fields["open"].index)
    out = pd.DataFrame(index=idx)
    for f in ("open", "high", "low", "close"):
        v = 50.14348112
        for s, w in weights.items():
            v = v * (fields[f].loc[idx][s] ** w)
        out[f] = v
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    return out


def get_candles(symbol: str) -> dict:
    """Returns {"1h": df, "15m": df} for a tradeable symbol ({} if unknown/unavailable)."""
    key = symbol.upper()
    ticker = _YF_TICKERS.get(key)
    td_sym = _to_td(key)
    if not ticker and not td_sym:
        return {}

    # 1) Twelve Data
    if td_sym and TD_KEY:
        try:
            base = _td_base([td_sym])[td_sym]
            out = _both(base)
            _source_of[key] = "Twelve Data"
            _last_good[key] = (time.time(), out, "Twelve Data")
            return out
        except Exception as e:  # noqa: BLE001 - any failure falls through to Yahoo
            log.warning("Twelve Data failed for %s: %s", key, e)

    # 2) Yahoo Finance (original path)
    if ticker:
        out = {}
        for label, (interval, period) in _INTERVALS.items():
            df = _cached_download(ticker, interval, period)
            if df is not None:
                out[label] = df
        if out:
            _source_of[key] = "Yahoo Finance"
            _last_good[key] = (time.time(), out, "Yahoo Finance")
            return out

    # 3) Stale (labelled) rather than nothing
    hit = _last_good.get(key)
    if hit and time.time() - hit[0] < _MAX_STALE:
        _source_of[key] = hit[2] + " (cached)"
        return hit[1]
    return {}


def get_dxy_candles() -> tuple[dict, str]:
    """
    Returns ({"1h": df, "15m": df}, source) where source is
    "computed" (ICE-formula DXY from live Twelve Data components),
    "direct" (Yahoo DX-Y.NYB) or "synthetic" (Yahoo basket proxy).
    """
    # 1) Twelve Data components -> ICE-formula DXY
    if TD_KEY:
        try:
            frames = _td_base(_DXY_TD)
            if len(frames) == len(_DXY_TD):
                base = _dxy_from_components(frames)
                if len(base) >= _MIN_BARS:
                    out = _both(base)
                    _source_of["DXY"] = "Twelve Data"
                    _last_good["DXY"] = (time.time(), out, "computed")
                    return out, "computed"
        except Exception as e:  # noqa: BLE001
            log.warning("Twelve Data DXY failed: %s", e)

    # 2) Yahoo direct ticker (original path)
    out = {}
    for label, (interval, period) in _INTERVALS.items():
        df = _cached_download(DXY_TICKER, interval, period)
        if df is not None:
            out[label] = df
    if out:
        _source_of["DXY"] = "Yahoo Finance"
        _last_good["DXY"] = (time.time(), out, "direct")
        return out, "direct"

    # 3) Stale TD/Yahoo DXY, labelled
    hit = _last_good.get("DXY")
    if hit and time.time() - hit[0] < _MAX_STALE:
        _source_of["DXY"] = "Yahoo Finance"
        return hit[1], hit[2]

    # 4) Synthetic basket proxy (original fallback, unchanged)
    out = {}
    for label, (interval, period) in _INTERVALS.items():
        frames = {}
        for tkr in _DXY_BASKET:
            df = _cached_download(tkr, interval, period)
            if df is not None:
                frames[tkr] = df
        if not frames:
            continue
        base = next(iter(frames.values()))["time"]
        synth = pd.Series(100.0, index=range(len(base)))
        for tkr, weight in _DXY_BASKET.items():
            if tkr not in frames:
                continue
            closes = frames[tkr]["close"].reset_index(drop=True)
            returns = closes.pct_change().fillna(0.0)
            synth = synth * (1 + weight * returns).cumprod()
        df = pd.DataFrame({
            "time": base, "open": synth, "high": synth, "low": synth, "close": synth,
        })
        out[label] = df
    _source_of["DXY"] = "Yahoo Finance"
    return out, "synthetic"
