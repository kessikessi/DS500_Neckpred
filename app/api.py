from __future__ import annotations

from .storage import ForecastStore


def create_app(store: ForecastStore):
    """Create the optional FastAPI adapter without making FastAPI a core dependency."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("Install the optional 'api' dependencies to run the web server") from exc

    app = FastAPI(title="Neckar water-level forecast")

    @app.get("/api/forecast")
    def forecast():
        result = store.load()
        if result is None:
            raise HTTPException(status_code=503, detail="No forecast is available")
        return result.to_dict()

    @app.get("/api/health")
    def health():
        return {"status": "ok", "forecast_available": store.load() is not None}

    return app

