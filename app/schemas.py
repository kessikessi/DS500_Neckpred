from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ForecastPoint:
    timestamp: datetime
    water_level_cm: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "water_level_cm": self.water_level_cm,
        }


@dataclass(frozen=True)
class Forecast:
    station_id: str
    station_name: str
    generated_at: datetime
    input_observed_at: datetime
    model_version: str
    stale: bool
    points: tuple[ForecastPoint, ...]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["generated_at"] = self.generated_at.isoformat()
        result["input_observed_at"] = self.input_observed_at.isoformat()
        result["points"] = [point.to_dict() for point in self.points]
        return result

