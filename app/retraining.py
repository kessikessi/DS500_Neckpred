from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence


class ModelTrainer(Protocol):
    def train(self, rows: Sequence[dict[str, Any]], output_dir: Path, version: str) -> Path:
        """Train and write a candidate artifact."""


@dataclass(frozen=True)
class RetrainingResult:
    version: str
    promoted: bool
    reason: str
    artifact: Path | None = None


def calendar_month_start(day: date) -> bool:
    return day.day == 1


def retrain_and_promote(
    trainer: ModelTrainer,
    rows: Sequence[dict[str, Any]],
    model_dir: Path,
    version: str,
    quality_gate: Callable[[Path], bool],
) -> RetrainingResult:
    """Train a candidate, promote it only after the caller's validation gate passes."""
    candidate_dir = model_dir / "candidates" / version
    artifact = trainer.train(rows, candidate_dir, version)
    if not quality_gate(artifact):
        return RetrainingResult(version, False, "quality gate failed", artifact)
    model_dir.mkdir(parents=True, exist_ok=True)
    promoted = model_dir / "active"
    temporary = model_dir / f".active-{version}"
    temporary.write_text(str(artifact), encoding="utf-8")
    temporary.replace(promoted)
    return RetrainingResult(version, True, "promoted", artifact)
