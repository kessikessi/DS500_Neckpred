from __future__ import annotations

import json
import logging
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import Settings
from .ingestion import (
    fetch_live_pegel,
    fetch_recent_dwd_weather,
    load_historical_weather,
    load_scraped_pegel,
    pegel_observation_time,
)
from .model import ForecastModel, XGBoostModel
from .refresh import refresh_with_fallback
from .schemas import Forecast
from .storage import ForecastStore

LOGGER = logging.getLogger("neckpred.scheduler")


@dataclass(frozen=True)
class HourlyRunResult:
    forecast: Forecast | None
    pegel_fetched: bool
    weather_rows: int
    error: str | None = None


class NoNewDataError(ValueError):
    """Raised when a source has not published an observation for this cycle."""


def _load_observation_history(path: Path, seed_path: Path) -> dict[datetime, float]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {datetime.fromisoformat(key): float(value) for key, value in payload.items()}
    return load_scraped_pegel(seed_path)


def _save_observation_history(path: Path, observations: dict[datetime, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = {timestamp.isoformat(): value for timestamp, value in sorted(observations.items())}
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_weather_snapshot(path: Path, weather: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    weather.to_json(temporary, orient="records", date_format="iso")
    temporary.replace(path)


def _load_weather_history(path: Path, seed_path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_json(path, orient="records", convert_dates=["timestamp"])
    return load_historical_weather(seed_path)


def _merge_weather_history(history: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([history, latest], ignore_index=True)
    return (
        combined.dropna(subset=["timestamp", "value"])
        .assign(
            station_id=lambda frame: frame["station_id"].astype(str).str.zfill(5),
            timestamp=lambda frame: pd.to_datetime(frame["timestamp"]).dt.floor("h"),
        )
        .drop_duplicates(["station_id", "timestamp", "parameter"], keep="last")
        .sort_values(["timestamp", "station_id", "parameter"])
        .reset_index(drop=True)
    )


def _next_full_hour(now: datetime) -> datetime:
    return now.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)


def _is_new_pegel_observation(
    observed_at: datetime,
    observations: dict[datetime, float],
) -> bool:
    normalized = observed_at.replace(minute=0, second=0, microsecond=0)
    return normalized > max(observations, default=datetime.min)


def _save_scheduler_status(path: Path, status: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _latest_complete_observed_at(
    observations: dict[datetime, float],
    latest: datetime,
    lag_hours: int,
) -> datetime:
    candidates = sorted(
        (timestamp for timestamp in observations if timestamp <= latest),
        reverse=True,
    )
    for candidate in candidates:
        required = {
            candidate.replace(minute=0, second=0, microsecond=0)
            for candidate in (
                candidate - pd.Timedelta(hours=offset)
                for offset in range(1, lag_hours + 1)
            )
        }
        if required.issubset(observations):
            return candidate
    raise ValueError(f"No complete {lag_hours}-hour water-level lag window is available")


def run_hourly_once(
    settings: Settings,
    model: ForecastModel | None = None,
    *,
    fetch_pegel: Callable[[], dict[str, object]] = fetch_live_pegel,
    fetch_weather: Callable[[], pd.DataFrame] = fetch_recent_dwd_weather,
    require_weather: bool = False,
    require_new_data: bool = False,
) -> HourlyRunResult:
    """Run one hourly fetch/predict cycle.

    ``require_weather`` can be enabled once the promoted model consumes DWD
    features. The current placeholder model only needs water-level lags.
    """
    store = ForecastStore(settings.state_dir / "forecast.json")
    history_path = settings.state_dir / "pegel_observations.json"
    weather_path = settings.state_dir / "weather_latest.json"
    weather_history_path = settings.state_dir / "weather_observations.json"
    status_path = settings.state_dir / "scheduler_status.json"
    try:
        attempted_at = datetime.now().isoformat()
        pegel = fetch_pegel()
        observed_at = pegel_observation_time(pegel["water_level_time"])
        water_level = float(pegel["water_level"])
        observations = _load_observation_history(
            history_path,
            settings.data_dir / "pegel_kirchentellinsfurt.csv",
        )
        new_pegel = _is_new_pegel_observation(observed_at, observations)
        if require_new_data and not new_pegel:
            raise NoNewDataError(
                f"No new Pegel observation after {max(observations).isoformat()}"
            )
        observations[observed_at.replace(minute=0, second=0, microsecond=0)] = water_level

        latest_weather = fetch_weather()
        weather_history = _load_weather_history(
            weather_history_path,
            settings.data_dir / "weather_data_hourly_1968_2026.csv",
        )
        if require_new_data:
            if latest_weather.empty:
                raise NoNewDataError("DWD returned no new observations")
            latest_timestamps = pd.to_datetime(
                latest_weather["timestamp"], errors="coerce"
            ).dropna()
            history_max = pd.to_datetime(
                weather_history["timestamp"], errors="coerce"
            ).max()
            if latest_timestamps.empty or latest_timestamps.max() <= history_max:
                raise NoNewDataError(
                    f"No new DWD observation after {history_max.isoformat()}"
                )
        weather = _merge_weather_history(weather_history, latest_weather)
        weather_rows = len(latest_weather)
        _save_observation_history(history_path, observations)
        _save_weather_snapshot(weather_history_path, weather)
        _save_weather_snapshot(weather_path, latest_weather)
        if require_weather and weather.empty:
            raise ValueError("No historical or recent DWD observations are available")
        inference_time = _latest_complete_observed_at(
            observations,
            max(observations),
            settings.lag_hours,
        )
        forecast = refresh_with_fallback(
            observations=observations,
            observed_at=inference_time,
            model=model or XGBoostModel(settings.model_artifact),
            store=store,
            station_id=settings.station_id,
            station_name=settings.station_name,
        )
        if forecast is None:
            raise RuntimeError("Forecast generation failed and no previous forecast is available")
        _save_scheduler_status(
            status_path,
            {
                "status": "ok",
                "last_attempt_at": attempted_at,
                "last_success_at": datetime.now().isoformat(),
                "weather_rows": weather_rows,
                "error": None,
            },
        )
        return HourlyRunResult(forecast, True, weather_rows)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        LOGGER.exception("Hourly refresh failed")
        _save_scheduler_status(
            status_path,
            {
                "status": "error",
                "last_attempt_at": datetime.now().isoformat(),
                "last_success_at": None,
                "weather_rows": 0,
                "error": str(error),
            },
        )
        return HourlyRunResult(
            forecast=store.load_for_serving(str(error)),
            pegel_fetched=False,
            weather_rows=0,
            error=str(error),
        )


def run_hourly_forever(
    settings: Settings,
    model: ForecastModel | None = None,
    *,
    interval_seconds: int = 3600,
    retry_interval_seconds: int = 60,
    max_wait_minutes: int = 15,
    **kwargs: object,
) -> None:
    """Run after each full hour, retrying until fresh data or the deadline."""

    stop_event = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        LOGGER.info("Scheduler shutdown requested")
        stop_event.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    LOGGER.info(
        "Scheduler started; retry interval=%s seconds, max wait=%s minutes",
        retry_interval_seconds,
        max_wait_minutes,
    )
    while not stop_event.is_set():
        now = datetime.now()
        next_hour = _next_full_hour(now)
        wait_seconds = max(0.0, (next_hour - now).total_seconds())
        LOGGER.info("Next hourly fetch scheduled for %s", next_hour.isoformat())
        if stop_event.wait(wait_seconds):
            break

        deadline = next_hour + pd.Timedelta(minutes=max_wait_minutes)
        while not stop_event.is_set():
            result = run_hourly_once(
                settings,
                model,
                require_new_data=True,
                **kwargs,
            )
            if result.error is None:
                LOGGER.info("Hourly refresh succeeded")
                break
            now = datetime.now()
            if now >= deadline:
                LOGGER.error("Hourly refresh deadline reached: %s", result.error)
                break
            retry_wait = min(
                retry_interval_seconds,
                max(0.0, (deadline - now).total_seconds()),
            )
            LOGGER.warning(
                "Hourly refresh did not receive new data; retrying in %s seconds: %s",
                retry_wait,
                result.error,
            )
            if stop_event.wait(retry_wait):
                break
    LOGGER.info("Scheduler stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_hourly_forever(Settings())
