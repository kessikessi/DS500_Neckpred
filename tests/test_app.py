from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.features import build_lag_features
from app.ingestion import (
    load_historical_weather,
    load_scraped_pegel,
    parse_pegel_station_row,
    pegel_observation_time,
)
from app.model import PersistenceModel
from app.model import XGBoostModel
from app.pipeline import generate_forecast
from app.refresh import refresh_with_fallback
from app.schemas import ForecastPoint
from app.storage import ForecastStore


class ForecastAppTests(unittest.TestCase):
    def test_features_are_oldest_to_newest(self):
        observed_at = datetime(2026, 1, 1, 6)
        observations = {
            observed_at - timedelta(hours=offset): float(offset)
            for offset in range(1, 7)
        }
        self.assertEqual(build_lag_features(observations, observed_at), [6, 5, 4, 3, 2, 1])

    def test_placeholder_generates_72_points_and_persists(self):
        observed_at = datetime(2026, 1, 1, 6)
        observations = {
            observed_at - timedelta(hours=offset): 100.0
            for offset in range(1, 7)
        }
        forecast = generate_forecast(
            observations, observed_at, PersistenceModel(), "00086", "Kirchentellinsfurt"
        )
        self.assertEqual(len(forecast.points), 72)
        self.assertEqual(forecast.points[-1], ForecastPoint(datetime(2026, 1, 4, 6), 100.0))
        with TemporaryDirectory() as directory:
            store = ForecastStore(Path(directory) / "forecast.json")
            store.save(forecast)
            loaded = store.load()
            self.assertEqual(loaded, forecast)
            self.assertTrue(store.load_for_serving("DWD timeout").stale)

    def test_scraped_pegel_file_is_normalized(self):
        path = Path("data/intermediate/pegel_kirchentellinsfurt.csv")
        observations = load_scraped_pegel(path)
        self.assertGreater(len(observations), 100)
        self.assertTrue(all(isinstance(value, float) for value in observations.values()))

    def test_historical_weather_file_is_normalized(self):
        weather = load_historical_weather(
            Path("data/intermediate/weather_data_hourly_1968_2026.csv")
        )
        self.assertGreater(len(weather), 100)
        self.assertEqual(
            set(weather.columns),
            {"station_id", "timestamp", "parameter", "value"},
        )
        self.assertTrue(weather["station_id"].str.len().eq(5).all())

    def test_missing_lag_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Missing water-level observation"):
            build_lag_features({}, datetime(2026, 1, 1, 6))

    def test_xgboost_artifact_generates_72_predictions(self):
        model = XGBoostModel()
        predictions = model.predict([100.0] * 6, 72)
        self.assertEqual(model.version, "xgboost-best")
        self.assertEqual(len(predictions), 72)
        self.assertTrue(all(isinstance(value, float) for value in predictions))

    def test_xgboost_uses_newest_to_oldest_artifact_order(self):
        model = XGBoostModel()
        captured = {}

        class CaptureModel:
            def predict(self, frame):
                captured["columns"] = list(frame.columns)
                captured["values"] = frame.iloc[0].tolist()
                return [[0.0] * 72]

        model._model = CaptureModel()
        model.predict([1, 2, 3, 4, 5, 6], 72)
        self.assertEqual(captured["columns"], list(model.FEATURE_NAMES))
        self.assertEqual(captured["values"], [6, 5, 4, 3, 2, 1])

    def test_live_pegel_row_is_parsed(self):
        javascript = "['00086', 'Kirchentellinsfurt', 'Neckar', 'W', 126, 'cm', '13.11.2026 11:55 MEZ', 11.9, 'm³/s', '13.11.2026 11:55 MEZ'],\n"
        result = parse_pegel_station_row(javascript, "00086")
        self.assertEqual(result["water_level"], 126.0)
        self.assertEqual(pegel_observation_time(result["water_level_time"]), datetime(2026, 11, 13, 11, 55))

    def test_unknown_live_pegel_station_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "was not found"):
            parse_pegel_station_row("['00086', 'station']", "99999")

    def test_refresh_falls_back_to_last_success(self):
        observed_at = datetime(2026, 1, 1, 6)
        observations = {
            observed_at - timedelta(hours=offset): 100.0
            for offset in range(1, 7)
        }
        with TemporaryDirectory() as directory:
            store = ForecastStore(Path(directory) / "forecast.json")
            first = refresh_with_fallback(
                observations, observed_at, PersistenceModel(), store
            )
            fallback = refresh_with_fallback(
                {}, observed_at, PersistenceModel(), store
            )
            self.assertFalse(first.stale)
            self.assertTrue(fallback.stale)
            self.assertIn("Missing water-level observation", fallback.error)


if __name__ == "__main__":
    unittest.main()
