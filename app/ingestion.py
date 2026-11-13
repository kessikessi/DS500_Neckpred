from __future__ import annotations

import ast
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from datetime import timedelta, timezone
from typing import Protocol, Sequence

import pandas as pd


def load_scraped_pegel(path: Path) -> dict[datetime, float]:
    """Load the assumed-successfully-scraped gauge history in the project format."""
    frame = pd.read_csv(path, sep=";", usecols=["Messzeitpunkt", "Messwert"], dtype="string")
    timestamps = pd.to_datetime(frame["Messzeitpunkt"], format="%d.%m.%Y %H:%M", errors="coerce")
    levels = pd.to_numeric(frame["Messwert"].str.replace(",", ".", regex=False), errors="coerce")
    valid = timestamps.notna() & levels.notna()
    observations: dict[datetime, float] = {}
    for timestamp, level in zip(timestamps[valid], levels[valid]):
        observations[timestamp.to_pydatetime()] = float(level)
    if not observations:
        raise ValueError(f"No valid Pegel observations found in {path}")
    return observations


def load_historical_weather(path: Path) -> pd.DataFrame:
    """Load the notebook-produced hourly weather history in normalized long form."""
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {
        "station_id",
        "timestamp",
        "humidity_relative",
        "pressure_vapor",
        "temperature_air_2m",
        "precipitation_amount",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Historical weather file is missing columns: {sorted(missing)}")
    long = frame.melt(
        id_vars=["station_id", "timestamp"],
        value_vars=sorted(required - {"station_id", "timestamp"}),
        var_name="parameter",
        value_name="value",
    )
    long["station_id"] = long["station_id"].astype(str).str.zfill(5)
    long["timestamp"] = pd.to_datetime(long["timestamp"], errors="coerce").dt.floor("h")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["timestamp", "value"]).sort_values(
        ["timestamp", "station_id", "parameter"]
    ).reset_index(drop=True)


class ObservationSource(Protocol):
    def fetch(self) -> dict[datetime, float]:
        """Return normalized hourly observations."""


PEGEL_URL = "https://www.hvz.baden-wuerttemberg.de/js/hvz_peg_stmn.js"
DEFAULT_DWD_STATIONS = ("03278", "04931", "02074", "06275", "03402")
DEFAULT_DWD_PARAMETERS = (
    ("hourly", "moisture", "temperature_air_2m"),
    ("hourly", "moisture", "pressure_vapor"),
    ("hourly", "moisture", "humidity_relative"),
    ("hourly", "precipitation", "precipitation_amount"),
)


def parse_pegel_station_row(javascript: str, station_id: str) -> dict[str, object]:
    """Parse one station row from the HVZ station JavaScript file."""
    match = re.search(
        rf"^\s*\['{re.escape(station_id)}'.*$",
        javascript,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Station {station_id} was not found in the Pegel response")
    try:
        row = ast.literal_eval(match.group(0).strip().rstrip(","))
        return {
            "station_id": row[0],
            "station": row[1],
            "river": row[2],
            "water_level": float(row[4]),
            "water_level_unit": row[5],
            "water_level_time": row[6],
            "discharge": float(row[7]),
            "discharge_unit": row[8],
            "discharge_time": row[9],
        }
    except (SyntaxError, ValueError, IndexError, TypeError) as exc:
        raise ValueError(f"Invalid Pegel row for station {station_id}") from exc


def fetch_live_pegel(
    station_id: str = "00086",
    url: str = PEGEL_URL,
    timeout: float = 15,
) -> dict[str, object]:
    """Fetch the current Kirchentellinsfurt reading from the HVZ station feed."""
    request = urllib.request.Request(url, headers={"User-Agent": "Neckpred/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            javascript = response.read().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Could not fetch live Pegel data: {exc}") from exc
    return parse_pegel_station_row(javascript, station_id)


def pegel_observation_time(value: object) -> datetime:
    """Convert the HVZ local time string into a naive hourly timestamp."""
    text = str(value).replace(" MEZ", "").replace(" MESZ", "")
    return datetime.strptime(text, "%d.%m.%Y %H:%M")


def fetch_dwd_weather(
    start: datetime,
    end: datetime,
    station_ids: Sequence[str] = DEFAULT_DWD_STATIONS,
    parameters: Sequence[tuple[str, str, str]] = DEFAULT_DWD_PARAMETERS,
) -> pd.DataFrame:
    """Fetch recent hourly DWD observations using wetterdienst.

    The returned frame has one row per station/timestamp/parameter and the
    normalized columns ``station_id``, ``timestamp``, ``parameter``, ``value``.
    """
    try:
        from wetterdienst.provider.dwd.observation import DwdObservationRequest
    except ImportError as exc:
        raise RuntimeError(
            "wetterdienst is required for live DWD ingestion; install the data dependencies"
        ) from exc

    request = DwdObservationRequest(
        parameters=list(parameters),
        start=start,
        end=end,
    ).filter_by_station_id(station_id=list(station_ids))
    try:
        frame = request.values().all().df
        frame = frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Could not fetch DWD weather data: {exc}") from exc
    required = {"station_id", "timestamp", "parameter", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"DWD response is missing columns: {sorted(missing)}")
    normalized = frame.loc[:, list(required)].copy()
    normalized["station_id"] = normalized["station_id"].astype(str).str.zfill(5)
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce").dt.floor("h")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp", "value"])
    return normalized.sort_values(["timestamp", "station_id", "parameter"]).reset_index(drop=True)


def fetch_recent_dwd_weather(
    lookback_hours: int = 6,
    now: datetime | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Fetch the recent window needed to assemble hourly model inputs."""
    end = now or datetime.now(timezone.utc).replace(tzinfo=None)
    return fetch_dwd_weather(
        start=end - timedelta(hours=lookback_hours),
        end=end,
        **kwargs,
    )
