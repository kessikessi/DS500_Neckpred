from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from app.config import Settings
from app.scheduler import _next_full_hour, run_hourly_once


class SchedulerTests(unittest.TestCase):
    def test_hourly_run_fetches_and_persists_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(
                data_dir=Path("data/intermediate"),
                state_dir=root,
                model_dir=root / "models",
            )
            history = pd.read_csv(
                "data/intermediate/pegel_kirchentellinsfurt.csv",
                sep=";",
                usecols=["Messzeitpunkt"],
            )
            latest = pd.to_datetime(
                history["Messzeitpunkt"], format="%d.%m.%Y %H:%M"
            ).max()
            live_time = (latest + pd.Timedelta(hours=1)).strftime("%d.%m.%Y %H:%M")
            result = run_hourly_once(
                settings,
                fetch_pegel=lambda: {
                    "water_level": 128,
                    "water_level_time": f"{live_time} MEZ",
                },
                fetch_weather=lambda: pd.DataFrame(
                    [{"station_id": "03278", "timestamp": "2026-11-13T11:00:00", "parameter": "temperature_air_2m", "value": 10.0}]
                ),
            )
            self.assertIsNotNone(result.forecast)
            self.assertTrue((root / "forecast.json").exists())
            self.assertTrue((root / "pegel_observations.json").exists())
            self.assertTrue((root / "weather_latest.json").exists())
            self.assertTrue((root / "weather_observations.json").exists())
            self.assertEqual(result.weather_rows, 1)

    def test_weather_requirement_accepts_historical_seed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(data_dir=Path("data/intermediate"), state_dir=root)
            result = run_hourly_once(
                settings,
                fetch_pegel=lambda: {
                    "water_level": 128,
                    "water_level_time": "13.11.2026 11:55 MEZ",
                },
                fetch_weather=lambda: pd.DataFrame(),
                require_weather=True,
            )
            self.assertIsNotNone(result.forecast)
            self.assertIsNone(result.error)

    def test_empty_live_weather_uses_historical_seed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(data_dir=Path("data/intermediate"), state_dir=root)
            result = run_hourly_once(
                settings,
                fetch_pegel=lambda: {
                    "water_level": 128,
                    "water_level_time": "13.11.2026 11:55 MEZ",
                },
                fetch_weather=lambda: pd.DataFrame(
                    columns=["station_id", "timestamp", "parameter", "value"]
                ),
                require_weather=True,
            )
            self.assertIsNotNone(result.forecast)
            self.assertEqual(result.weather_rows, 0)
            self.assertTrue((root / "weather_observations.json").exists())

    def test_next_full_hour(self):
        self.assertEqual(
            _next_full_hour(datetime(2026, 11, 13, 5, 20, 10)),
            datetime(2026, 11, 13, 6, 0),
        )

    def test_unchanged_pegel_can_be_reported_as_no_new_data(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(data_dir=Path("data/intermediate"), state_dir=root)
            result = run_hourly_once(
                settings,
                fetch_pegel=lambda: {
                    "water_level": 128,
                    "water_level_time": "13.11.2026 11:55 MEZ",
                },
                fetch_weather=lambda: pd.DataFrame(),
                require_new_data=True,
            )
            self.assertIsNotNone(result.error)
            self.assertIn("no new", result.error.lower())


if __name__ == "__main__":
    unittest.main()
