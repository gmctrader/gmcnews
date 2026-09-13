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

import pandas as pd
import yfinance as yf

# yfinance interval/period pairs that keep us under its lookback limits.
_INTERVALS = {
    "1h": ("60m", "1mo"),
    "15m": ("15m", "5d"),
}

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
        try:
            raw = yf.download(ticker, interval=interval, period=period,
                               progress=False, auto_adjust=True)
            if raw is not None and not raw.empty:
                out[label] = _normalize(raw)
        except Exception:
            continue
    return out


def get_dxy_candles() -> tuple[dict, str]:
    """
    Returns ({"1h": df, "15m": df}, source) where source is
    "broker"/"direct" or "synthetic".
    """
    out = {}
    for label, (interval, period) in _INTERVALS.items():
        try:
            raw = yf.download(DXY_TICKER, interval=interval, period=period,
                               progress=False, auto_adjust=True)
            if raw is not None and not raw.empty:
                out[label] = _normalize(raw)
        except Exception:
            continue
    if out:
        return out, "direct"

    # Fallback: synthetic index from the basket, weighted log-returns
    # accumulated into an index series (same shape as a real price df).
    out = {}
    for label, (interval, period) in _INTERVALS.items():
        frames = {}
        for tkr in _DXY_BASKET:
            try:
                raw = yf.download(tkr, interval=interval, period=period,
                                   progress=False, auto_adjust=True)
                if raw is not None and not raw.empty:
                    frames[tkr] = _normalize(raw)
            except Exception:
                continue
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
