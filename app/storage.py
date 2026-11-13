from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .schemas import Forecast


class ForecastStore:
    def __init__(self, path: Path):
        self.path = path

    def save(self, forecast: Forecast) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(forecast.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def load(self) -> Forecast | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        from datetime import datetime
        from .schemas import ForecastPoint

        return Forecast(
            station_id=data["station_id"],
            station_name=data["station_name"],
            generated_at=datetime.fromisoformat(data["generated_at"]),
            input_observed_at=datetime.fromisoformat(data["input_observed_at"]),
            model_version=data["model_version"],
            stale=bool(data["stale"]),
            points=tuple(
                ForecastPoint(datetime.fromisoformat(point["timestamp"]), float(point["water_level_cm"]))
                for point in data["points"]
            ),
            error=data.get("error"),
        )

    def load_for_serving(self, error: str | None = None) -> Forecast | None:
        forecast = self.load()
        return replace(forecast, stale=True, error=error) if forecast else None

