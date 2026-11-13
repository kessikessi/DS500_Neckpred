# DS500_Neckpred

Repository for the group Data Science Project DS500 (University of Tübingen). Topic: predict the water level of the Neckar in Tübingen-Lustnau and present the forecast to rowers in an interactive dashboard.

## Repository layout

| Path | Content |
|---|---|
| `app/` | Dash web dashboard (layout, callbacks, data interface) |
| `tests/` | Unit tests for the dashboard |
| `code/` | Exploratory notebooks (data download, checks, correlations, training data) |
| `data/` | Intermediate and final datasets of the exploration phase |
| `out/` | Figures from the exploration phase |
| `neckar_ridge_experiment.py` | Ridge pilot experiment with persistence baseline |

## Dashboard

The dashboard currently runs on **synthetic placeholder data**. No model or live data source is connected yet. A banner on the page says so.

### Design

```
forecast job (every 15 min, later)  ──writes──▶  storage (later)
                                                    │
                                                    ▼
                         app/data_provider.py  get_provider().get_latest()
                                                    │  ForecastBundle (UTC)
                                                    ▼
                                    app/main.py  Dash layout + callback
```

* `app/data_provider.py` defines the only interface between the dashboard and the pipeline. It returns a `ForecastBundle` with recent observations, the forecast and an uncertainty band. All timestamps are UTC, levels are in cm. To connect the real pipeline, add a new provider class (e.g. reading from a database) and select it with `NECKAR_DATA_PROVIDER`. The dashboard itself does not change.
* The dashboard does not compute forecasts. It reloads the newest forecast every 60 seconds and warns if it is older than 45 minutes.
* `app/status.py` contains the traffic light logic for rowers. It has no Dash dependency and is unit tested.
* `app/config.py` holds all settings. Each can be overridden with an environment variable.

| Variable | Default | Meaning |
|---|---|---|
| `NECKAR_DATA_PROVIDER` | `dummy` | Data source for the dashboard |
| `NECKAR_LEVEL_LOW_CM` | `90` | Below: jetty dry (placeholder value) |
| `NECKAR_LEVEL_HIGH_CM` | `180` | Above: current too strong (placeholder value) |
| `NECKAR_WARN_MARGIN_CM` | `10` | Distance to a threshold that triggers "caution" |
| `NECKAR_STALE_AFTER_MIN` | `45` | Age after which a forecast counts as outdated |
| `NECKAR_REFRESH_SECONDS` | `60` | Browser refresh interval |

The thresholds are placeholders and still have to be agreed with the rowing club.

### Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m app.main
```

Then open http://127.0.0.1:8050. Run the tests with `pytest`.

### Run with Docker (as on the server)

```bash
docker build -t neckar-dashboard .
docker run --rm -p 8050:8050 neckar-dashboard
```

The container serves the app with gunicorn on port 8050 and exposes `/health` for health checks. On the bwCloud VM a reverse proxy (e.g. Caddy) in front of the container can add HTTPS.
