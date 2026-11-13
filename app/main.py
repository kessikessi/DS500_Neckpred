from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import Settings
from .ingestion import load_scraped_pegel
from .model import XGBoostModel
from .refresh import refresh_once
from .storage import ForecastStore


def create_local_app():
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    settings = Settings()
    store = ForecastStore(settings.state_dir / "forecast.json")
    if store.load() is None:
        observations = load_scraped_pegel(
            settings.data_dir / "pegel_kirchentellinsfurt.csv"
        )
        observed_at = max(observations)
        refresh_once(
            observations,
            observed_at,
            XGBoostModel(settings.model_artifact),
            store,
            settings.station_id,
            settings.station_name,
        )

    app = FastAPI(title="Neckar water-level forecast")

    @app.get("/api/forecast")
    def forecast():
        result = store.load()
        if result is None:
            return {"error": "No forecast is available"}
        return result.to_dict()

    @app.get("/api/health")
    def health():
        scheduler_status_path = settings.state_dir / "scheduler_status.json"
        scheduler_status = None
        if scheduler_status_path.exists():
            import json

            scheduler_status = json.loads(scheduler_status_path.read_text(encoding="utf-8"))
        return {
            "status": "ok",
            "forecast_available": store.load() is not None,
            "scheduler": scheduler_status,
        }

    frontend = Path(__file__).parent.parent / "frontend"
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_local_app()
