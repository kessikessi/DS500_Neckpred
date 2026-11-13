from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import pandas as pd


class ForecastModel(Protocol):
    version: str

    def predict(self, features: Sequence[float], horizon: int) -> Sequence[float]:
        """Predict one value per future hour."""


@dataclass(frozen=True)
class PersistenceModel:
    """Safe placeholder model used until a trained artifact is promoted."""

    version: str = "persistence-placeholder"

    def predict(self, features: Sequence[float], horizon: int) -> list[float]:
        if not features:
            raise ValueError("At least one feature is required")
        return [float(features[-1])] * horizon


class XGBoostModel:
    """Adapter for the repository's trained six-lag, 72-step XGBoost pipeline."""

    FEATURE_NAMES = tuple(f"water_level_lag_{hours}h" for hours in range(1, 7))
    DEFAULT_ARTIFACT = Path("out/models/best_xgb_model.joblib")

    def __init__(
        self,
        artifact_path: Path | str = DEFAULT_ARTIFACT,
        version: str = "xgboost-best",
    ):
        import joblib

        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"XGBoost model artifact not found: {self.artifact_path}")
        self._model = joblib.load(self.artifact_path)
        expected_names = getattr(self._model, "feature_names_in_", None)
        if expected_names is not None and tuple(expected_names) != self.FEATURE_NAMES:
            raise ValueError(
                "XGBoost artifact feature schema does not match the application contract"
            )
        if getattr(self._model, "n_features_in_", len(self.FEATURE_NAMES)) != len(
            self.FEATURE_NAMES
        ):
            raise ValueError("XGBoost artifact must accept exactly six lag features")
        self.version = version

    def predict(self, features: Sequence[float], horizon: int) -> list[float]:
        if len(features) != len(self.FEATURE_NAMES):
            raise ValueError(
                f"XGBoost requires {len(self.FEATURE_NAMES)} features, got {len(features)}"
            )
        # Feature construction is oldest-to-newest; the trained artifact expects
        # water_level_lag_1h through water_level_lag_6h.
        frame = pd.DataFrame([list(reversed(features))], columns=self.FEATURE_NAMES)
        values = self._model.predict(frame)
        flattened = list(values[0])
        if len(flattened) != horizon:
            raise ValueError(f"Model returned {len(flattened)} values; expected {horizon}")
        return [float(value) for value in flattened]


class JoblibModel:
    """Adapter for a promoted sklearn-compatible multi-output model."""

    def __init__(self, artifact_path: str, version: str):
        import joblib

        self._model = joblib.load(artifact_path)
        self.version = version

    def predict(self, features: Sequence[float], horizon: int) -> list[float]:
        values = self._model.predict([list(features)])
        flattened = list(values[0])
        if len(flattened) != horizon:
            raise ValueError(f"Model returned {len(flattened)} values; expected {horizon}")
        return [float(value) for value in flattened]
