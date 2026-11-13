"""Central configuration for the dashboard.

All values can be overridden with environment variables so that the same
container image runs locally and on the bwCloud VM without code changes.
"""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to ``default``."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


# Location of the forecast point (Tübingen-Lustnau, between the two weirs).
SITE_NAME = "Neckar Tübingen-Lustnau"
SITE_LAT = 48.522883
SITE_LON = 9.083089

# Timezone used for display only. All data is stored and passed around in UTC.
DISPLAY_TZ = "Europe/Berlin"

# Which data provider to use. "dummy" produces synthetic placeholder data.
# Later: e.g. "database" once the forecast job writes to PostgreSQL.
DATA_PROVIDER = os.getenv("NECKAR_DATA_PROVIDER", "dummy")

# Rowing thresholds in cm. PLACEHOLDERS: to be agreed with the rowing club
# (see docs/05-offene-punkte.md). Below LOW the jetty is dry, above HIGH the
# current is too strong. Within WARN_MARGIN of a threshold we show "caution".
LEVEL_LOW_CM = _env_float("NECKAR_LEVEL_LOW_CM", 90.0)
LEVEL_HIGH_CM = _env_float("NECKAR_LEVEL_HIGH_CM", 180.0)
WARN_MARGIN_CM = _env_float("NECKAR_WARN_MARGIN_CM", 10.0)

# The forecast job is expected to run every 15 minutes. If the newest forecast
# is older than this, the dashboard shows a staleness warning.
FORECAST_INTERVAL_MIN = 15
STALE_AFTER_MIN = _env_float("NECKAR_STALE_AFTER_MIN", 45.0)

# How often the browser asks the server for fresh data (milliseconds).
REFRESH_INTERVAL_MS = int(_env_float("NECKAR_REFRESH_SECONDS", 60.0) * 1000)

# Lead times shown as summary tiles (hours ahead of the forecast issue time).
SUMMARY_HORIZONS_H = (1, 3, 6, 12, 24)

# Selectable chart horizons (hours).
CHART_HORIZONS_H = (24, 48, 72)
HISTORY_HOURS = 48
