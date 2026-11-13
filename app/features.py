from __future__ import annotations

from datetime import datetime, timedelta
from typing import Mapping, Sequence


def build_lag_features(
    observations: Mapping[datetime, float],
    observed_at: datetime,
    lag_hours: int = 6,
) -> list[float]:
    """Return recent water levels oldest-to-newest for model inference."""
    values = []
    for offset in range(lag_hours, 0, -1):
        timestamp = observed_at - timedelta(hours=offset)
        try:
            values.append(float(observations[timestamp]))
        except KeyError as exc:
            raise ValueError(f"Missing water-level observation at {timestamp.isoformat()}") from exc
    return values


def hourly_timestamps(start: datetime, hours: int) -> list[datetime]:
    return [start + timedelta(hours=index) for index in range(1, hours + 1)]

