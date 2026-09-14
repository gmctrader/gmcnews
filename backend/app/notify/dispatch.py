"""
Signal dispatch -- takes a Signal.to_dict() payload (from
analysis/confluence.py) and sends it out as a normal, human-readable
trading signal to whichever channel(s) are configured via env vars:

  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID   -> Telegram bot message
  DISCORD_WEBHOOK_URL                      -> Discord embed
  GENERIC_WEBHOOK_URL                      -> raw JSON POST (for Zapier,
                                              your own server, MT5 EA
                                              bridge, etc.)

None are required -- if nothing is configured, dispatch_signal() is a
no-op that reports which channels it skipped, so the API endpoint
that calls this never breaks a deployment that hasn't set anything up
yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


def _fmt_price(v) -> str:
    return "—" if v is None else f"{v:g}"


def format_signal_text(signal: dict) -> str:
    """Plain-text message body shared by Telegram/Discord/generic channels."""
    direction = (signal.get("direction") or "").upper()
    arrow = "🟢 LONG" if signal.get("direction") == "bullish" else "🔴 SHORT" if signal.get("direction") == "bearish" else direction
    lvl = signal.get("levels") or {}
    targets = lvl.get("targets") or []
    rr = lvl.get("risk_reward") or []
    target_lines = "\n".join(
        f"   TP{i+1}: {_fmt_price(t)}  ({rr[i]:.1f}R)" if i < len(rr) else f"   TP{i+1}: {_fmt_price(t)}"
        for i, t in enumerate(targets)
    ) or "   (withheld -- no valid stop structure yet)"

    lines = [
        f"{arrow}  {signal.get('symbol', '?')}",
        f"Grade: {signal.get('grade', '—').upper()}  ·  Score: {signal.get('score', '—')}/100",
        "",
        f"Entry: {_fmt_price(lvl.get('entry'))}",
        f"Stop:  {_fmt_price(lvl.get('stop'))}"
        + (f"  ({lvl.get('risk_pips')} pips risk)" if lvl.get("risk_pips") else ""),
        "Targets:",
        target_lines,
        "",
        f"Why: {'; '.join(signal.get('reasons', [])[:3])}" if signal.get("reasons") else "",
        f"News: {signal.get('news_note', '')}",
    ]
    return "\n".join(l for l in lines if l is not None)


@dataclass
class DispatchResult:
    channel: str
    sent: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"channel": self.channel, "sent": self.sent, "detail": self.detail}


async def _send_telegram(text: str) -> DispatchResult:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return DispatchResult("telegram", False, "not configured")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json={"chat_id": chat_id, "text": text})
            r.raise_for_status()
        return DispatchResult("telegram", True, "sent")
    except Exception as e:
        return DispatchResult("telegram", False, str(e))


async def _send_discord(text: str) -> DispatchResult:
    webhook = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook:
        return DispatchResult("discord", False, "not configured")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(webhook, json={"content": f"```{text}```"})
            r.raise_for_status()
        return DispatchResult("discord", True, "sent")
    except Exception as e:
        return DispatchResult("discord", False, str(e))


async def _send_generic_webhook(signal: dict, text: str) -> DispatchResult:
    webhook = os.getenv("GENERIC_WEBHOOK_URL")
    if not webhook:
        return DispatchResult("webhook", False, "not configured")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(webhook, json={"signal": signal, "message": text})
            r.raise_for_status()
        return DispatchResult("webhook", True, "sent")
    except Exception as e:
        return DispatchResult("webhook", False, str(e))


async def dispatch_signal(signal: dict) -> dict:
    """
    Sends `signal` (a Signal.to_dict() payload) to every configured
    channel. Returns a per-channel report; channels with no env vars
    set are reported as skipped, not treated as errors.
    """
    text = format_signal_text(signal)
    results = [
        await _send_telegram(text),
        await _send_discord(text),
        await _send_generic_webhook(signal, text),
    ]
    return {
        "message": text,
        "results": [r.to_dict() for r in results],
        "any_sent": any(r.sent for r in results),
        "any_configured": any(r.detail != "not configured" for r in results),
    }
