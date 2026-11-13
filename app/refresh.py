from __future__ import annotations

from datetime import datetime
from typing import Mapping

from .model import ForecastModel
from .pipeline import generate_forecast
from .schemas import Forecast
from .storage import ForecastStore


def refresh_once(
    observations: Mapping[datetime, float],
    observed_at: datetime,
    model: ForecastModel,
    store: ForecastStore,
    station_id: str = "00086",
    station_name: str = "Kirchentellinsfurt",
) -> Forecast:
    """Generate and atomically persist one successful hourly forecast."""
    forecast = generate_forecast(
        observations=observations,
        observed_at=observed_at,
        model=model,
        station_id=station_id,
        station_name=station_name,
    )
    store.save(forecast)
    return forecast


def refresh_with_fallback(
    observations: Mapping[datetime, float],
    observed_at: datetime,
    model: ForecastModel,
    store: ForecastStore,
    station_id: str = "00086",
    station_name: str = "Kirchentellinsfurt",
) -> Forecast | None:
    """Return a fresh forecast, or the last forecast marked stale after an error."""
    try:
        return refresh_once(
            observations, observed_at, model, store, station_id, station_name
        )
    except (OSError, ValueError, RuntimeError) as error:
        return store.load_for_serving(str(error))

