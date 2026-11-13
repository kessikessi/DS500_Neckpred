"""Pure helper logic for the dashboard (no Dash imports, easy to test)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app import config


@dataclass(frozen=True)
class RowingStatus:
    key: str  # "ok", "caution" or "stop"; used as CSS class
    label: str  # short German label for the UI
    detail: str  # one sentence explanation


def classify_level(level_cm: float) -> RowingStatus:
    """Map a water level to a traffic light status for rowers."""
    low, high, margin = config.LEVEL_LOW_CM, config.LEVEL_HIGH_CM, config.WARN_MARGIN_CM
    if level_cm < low:
        return RowingStatus("stop", "Zu wenig Wasser", "Der Steg liegt voraussichtlich trocken.")
    if level_cm > high:
        return RowingStatus("stop", "Zu viel Wasser", "Die Strömung ist voraussichtlich zu stark.")
    if level_cm < low + margin:
        return RowingStatus("caution", "Grenzwertig niedrig", "Nahe an der Untergrenze, bitte vor Ort prüfen.")
    if level_cm > high - margin:
        return RowingStatus("caution", "Grenzwertig hoch", "Nahe an der Obergrenze, bitte vor Ort prüfen.")
    return RowingStatus("ok", "Rudern möglich", "Der Wasserstand liegt im üblichen Bereich.")


def trend(current_cm: float, future_cm: float, tolerance_cm: float = 2.0) -> tuple[str, str]:
    """Return (arrow, German text) describing the change from now to later."""
    diff = future_cm - current_cm
    if diff > tolerance_cm:
        return "↗", "steigend"
    if diff < -tolerance_cm:
        return "↘", "fallend"
    return "→", "gleichbleibend"


def value_at_lead(forecast: pd.DataFrame, issued_at: pd.Timestamp, hours: float) -> pd.Series | None:
    """Forecast row closest to ``issued_at + hours``; None if out of range."""
    if forecast.empty:
        return None
    target = issued_at + pd.Timedelta(hours=hours)
    idx = (forecast["time"] - target).abs().idxmin()
    row = forecast.loc[idx]
    if abs(row["time"] - target) > pd.Timedelta(minutes=config.FORECAST_INTERVAL_MIN):
        return None
    return row


def is_stale(issued_at: pd.Timestamp, now: pd.Timestamp | None = None) -> bool:
    """True if the newest forecast is older than the configured limit."""
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    return now - issued_at > pd.Timedelta(minutes=config.STALE_AFTER_MIN)


def to_local(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a UTC timestamp to the display timezone."""
    return ts.tz_convert(config.DISPLAY_TZ)
