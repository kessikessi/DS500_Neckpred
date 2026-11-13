"""Data interface between the dashboard and the forecasting pipeline.

The dashboard never talks to a database, an API or a model directly. It only
calls ``get_provider().get_latest()`` and renders the returned
``ForecastBundle``. To connect the real pipeline later, implement a new class
with the same ``get_latest`` method (e.g. reading from PostgreSQL) and register
it in ``get_provider``. Nothing in ``main.py`` has to change.

Conventions
-----------
* All timestamps are timezone-aware and in UTC.
* Water levels are in cm.
* ``observed`` has columns ``time, level_cm``.
* ``forecast`` has columns ``time, level_cm, lower_cm, upper_cm`` where
  ``lower_cm``/``upper_cm`` describe an uncertainty band (e.g. 10 % / 90 %).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from app import config


@dataclass(frozen=True)
class ForecastBundle:
    """Everything the dashboard needs for one render."""

    issued_at: pd.Timestamp  # when the forecast was produced (UTC)
    observed: pd.DataFrame  # recent measurements
    forecast: pd.DataFrame  # predicted levels with uncertainty band
    source: str  # human readable description of the data source
    is_placeholder: bool  # True while no real model is connected


class ForecastProvider(Protocol):
    """Anything that can deliver the latest forecast."""

    def get_latest(self) -> ForecastBundle: ...


class DummyProvider:
    """Synthetic data so the dashboard can be developed without a model.

    The curve is deterministic for a given 15 minute slot, so every reload
    within the same slot shows the same numbers, and a new "forecast" appears
    every 15 minutes just like the real job will produce it.
    """

    base_level_cm = 130.0

    def get_latest(self) -> ForecastBundle:
        now = pd.Timestamp.now(tz="UTC")
        issued_at = now.floor(f"{config.FORECAST_INTERVAL_MIN}min")

        step = pd.Timedelta(minutes=config.FORECAST_INTERVAL_MIN)
        hist_times = pd.date_range(
            end=issued_at, periods=config.HISTORY_HOURS * 4 + 1, freq=step, tz="UTC"
        )
        fc_hours = max(config.CHART_HORIZONS_H)
        fc_times = pd.date_range(
            start=issued_at + step, periods=fc_hours * 4, freq=step, tz="UTC"
        )

        observed = pd.DataFrame(
            {"time": hist_times, "level_cm": self._curve(hist_times, noise_seed=int(issued_at.timestamp()))}
        )

        level = self._curve(fc_times, noise_seed=None)
        lead_h = (fc_times - issued_at).total_seconds().to_numpy() / 3600.0
        spread = 3.0 + 0.6 * lead_h  # uncertainty grows with lead time
        forecast = pd.DataFrame(
            {
                "time": fc_times,
                "level_cm": level,
                "lower_cm": level - spread,
                "upper_cm": level + spread,
            }
        )

        return ForecastBundle(
            issued_at=issued_at,
            observed=observed,
            forecast=forecast,
            source="Platzhalterdaten (synthetisch)",
            is_placeholder=True,
        )

    def _curve(self, times: pd.DatetimeIndex, noise_seed: int | None) -> np.ndarray:
        """Slow wave plus a daily cycle, roughly resembling a river level."""
        t_h = times.asi8 / 3.6e12  # hours since epoch
        level = (
            self.base_level_cm
            + 35.0 * np.sin(2 * np.pi * t_h / 120.0)
            + 4.0 * np.sin(2 * np.pi * t_h / 24.0)
        )
        if noise_seed is not None:
            rng = np.random.default_rng(noise_seed)
            level = level + rng.normal(0.0, 1.0, size=len(times))
        return np.round(level, 1)


_PROVIDERS: dict[str, type] = {
    "dummy": DummyProvider,
    # "database": DatabaseProvider,  # TODO: add once the forecast job exists
}


def get_provider() -> ForecastProvider:
    """Return the provider selected via ``NECKAR_DATA_PROVIDER``."""
    try:
        return _PROVIDERS[config.DATA_PROVIDER]()
    except KeyError as exc:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown data provider {config.DATA_PROVIDER!r}. Known providers: {known}"
        ) from exc
