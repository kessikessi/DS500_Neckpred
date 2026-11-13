from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    station_id: str = "00086"
    station_name: str = "Kirchentellinsfurt"
    data_dir: Path = Path("data/intermediate")
    state_dir: Path = Path("runtime")
    model_dir: Path = Path("runtime/models")
    model_artifact: Path = Path("out/models/best_xgb_model.joblib")
    lag_hours: int = 6
    forecast_hours: int = 72
