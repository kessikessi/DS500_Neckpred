from __future__ import annotations

from datetime import datetime
from typing import Mapping

from .features import build_lag_features, hourly_timestamps
from .model import ForecastModel
from .schemas import Forecast, ForecastPoint


def generate_forecast(
    observations: Mapping[datetime, float],
    observed_at: datetime,
    model: ForecastModel,
    station_id: str,
    station_name: str,
    lag_hours: int = 6,
    forecast_hours: int = 72,
) -> Forecast:
    features = build_lag_features(observations, observed_at, lag_hours)
    predictions = model.predict(features, forecast_hours)
    timestamps = hourly_timestamps(observed_at, forecast_hours)
    if len(predictions) != len(timestamps):
        raise ValueError("Model output length does not match forecast horizon")
    return Forecast(
        station_id=station_id,
        station_name=station_name,
        generated_at=datetime.now(observed_at.tzinfo),
        input_observed_at=observed_at,
        model_version=model.version,
        stale=False,
        points=tuple(ForecastPoint(timestamp, float(value)) for timestamp, value in zip(timestamps, predictions)),
    )

