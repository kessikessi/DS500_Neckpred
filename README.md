# DS500_Neckpred
Repository for the group Data-Science Project DS500. Topic: Predict the water level of the Neckar

## Forecast application

The `app/` package provides the model-agnostic core for a 72-hour Kirchentellinsfurt
forecast. It consumes the assumed-successfully-scraped historical Pegel file,
constructs six hourly water-level lags, and uses a replaceable `ForecastModel`
interface. The running scheduler uses the trained artifact
`out/models/best_xgb_model.joblib` through `XGBoostModel`.

The forecast store uses atomic JSON replacement so the last successful forecast
can remain available when an hourly refresh fails. Monthly retraining should call
`retrain_and_promote` at the start of each calendar month using all validated
history. Candidates are promoted only after the caller's quality gate passes;
otherwise the active model remains unchanged and the failure should be alerted.

### Run the core checks

```bash
python3 -m unittest discover -s tests -v
```

### Optional API

Install `requirements-api.txt` and start the local demo server:

```bash
python3 -m pip install -r requirements-api.txt
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000/>. The demo creates an initial forecast from the
scraped Pegel history if no forecast exists, using the trained XGBoost model.
Check the API at <http://127.0.0.1:8000/api/forecast>
and the health endpoint at <http://127.0.0.1:8000/api/health>.

This command runs only the web process. To run both processes locally with one
command, use:

```bash
python3 run_local.py
```

This starts Uvicorn and the scheduler as separate child processes. They share
the `runtime/` directory, and the launcher terminates both processes together.
The API health endpoint includes the latest scheduler status from
`runtime/scheduler_status.json`.

For production, run the same two commands under a process supervisor rather
than using `--reload`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
python3 -m app.scheduler
```

The supervisor should restart either process independently and ensure both use
the same working directory and `runtime/` volume. Monthly retraining remains a
separate job; observed lags do not provide future weather information.

### Run the hourly scheduler

Run one cycle manually:

```bash
python3 -c "from app.scheduler import run_hourly_once; from app.config import Settings; print(run_hourly_once(Settings()))"
```

Run it continuously in a dedicated worker process:

```bash
python3 -m app.scheduler
```

The continuous worker waits for the next full hour before its first fetch. For
example, if it starts at 05:20, the first attempt is at 06:00. Each cycle
fetches the live Pegel reading, fetches recent DWD observations,
appends the Pegel value to `runtime/pegel_observations.json`, stores the latest
DWD response in `runtime/weather_latest.json`, merges it with the historical
`data/intermediate/weather_data_hourly_1968_2026.csv` seed, and stores the
combined weather history in `runtime/weather_observations.json`. If a fetch or
validation fails, the previous forecast is returned as stale. DWD completeness is optional for the current water-level-only XGBoost model;
pass `require_weather=True` when a future promoted model depends on weather
features.

If the Pegel source has not published a new observation, the worker retries
every 60 seconds by default until 15 minutes after the full hour. Both values
are configurable through `run_hourly_forever`; after the deadline, it waits
for the next full hour. A direct `run_hourly_once` call still performs one
immediate cycle and does not wait.

## Live data fetchers

The notebook-derived fetchers are in [`app/ingestion.py`](./app/ingestion.py):

```python
from app.ingestion import fetch_live_pegel, fetch_recent_dwd_weather

pegel = fetch_live_pegel("00086")
weather = fetch_recent_dwd_weather(lookback_hours=6)
```

`fetch_live_pegel` reads the HVZ station feed used by `pegel_scrape.ipynb` and
returns the current water level, discharge, station metadata, and source time.
`fetch_recent_dwd_weather` uses the hourly `wetterdienst` request from
`dwd_api.ipynb` and returns normalized `station_id`, `timestamp`, `parameter`,
and `value` columns. Network and malformed-response errors are raised
explicitly so the hourly refresh can retain the previous forecast.
