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

import time

import pandas as pd
import yfinance as yf

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


def get_candles(symbol: str) -> dict:
    """Returns {"1h": df, "15m": df} for a tradeable symbol."""
    ticker = _YF_TICKERS.get(symbol.upper())
    if not ticker:
        return {}
    out = {}
    for label, (interval, period) in _INTERVALS.items():
        df = _cached_download(ticker, interval, period)
        if df is not None:
            out[label] = df
    return out


def get_dxy_candles() -> tuple[dict, str]:
    """
    Returns ({"1h": df, "15m": df}, source) where source is
    "broker"/"direct" or "synthetic".
    """
    out = {}
    for label, (interval, period) in _INTERVALS.items():
        df = _cached_download(DXY_TICKER, interval, period)
        if df is not None:
            out[label] = df
    if out:
        return out, "direct"

    # Fallback: synthetic index from the basket, weighted log-returns
    # accumulated into an index series (same shape as a real price df).
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
    return out, "synthetic"
