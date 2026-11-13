"""Dash web app showing the Neckar water level forecast for rowers.

Run locally:
    python -m app.main

In production (Docker) the WSGI server imports ``app.main:server``.
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from app import config
from app.data_provider import ForecastBundle, get_provider
from app.status import classify_level, is_stale, to_local, trend, value_at_lead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Dash(
    __name__,
    title="Neckar Pegel Tübingen",
    # Headings in PT Serif, body text in Raleway (style inspired by local rowing club sites)
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=PT+Serif:wght@400;700&family=Raleway:wght@400;600;700&display=swap"
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # WSGI entry point for gunicorn

CHART_HEIGHT_PX = 380


@server.route("/health")
def health() -> tuple[str, int]:
    """Lightweight endpoint for uptime checks and the Docker healthcheck."""
    return "ok", 200


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


def _card(*children, class_name: str = "") -> html.Div:
    return html.Div(children, className=f"card {class_name}".strip())


app.layout = html.Div([
    html.Header(
        className="hero",
        children=html.Div(
            className="hero-inner",
            children=[
                html.P(config.SITE_NAME, className="hero-kicker"),
                html.H1("Wasserstand Neckar"),
                html.Div("Vorhersage für Ruderer", className="hero-banner"),
            ],
        ),
    ),
    html.Div(
        className="page",
        children=[
            html.Div(id="alerts"),
            html.Main(
                className="grid",
                children=[
                    _card(html.Div(id="status-content"), class_name="status-card"),
                    _card(
                        html.H2("Die nächsten Stunden"),
                        html.Div(id="tiles", className="tiles"),
                        class_name="tiles-card",
                    ),
                    _card(
                        html.Div(
                            className="chart-head",
                            children=[
                                html.H2("Verlauf und Vorhersage"),
                                dcc.RadioItems(
                                    id="horizon",
                                    options=[{"label": f"{h} h", "value": h} for h in config.CHART_HORIZONS_H],
                                    value=config.CHART_HORIZONS_H[0],
                                    inline=True,
                                    className="horizon-select",
                                ),
                            ],
                        ),
                        # Fixed container height: with a relative height the responsive
                        # graph resizes to its own container and grows without end.
                        dcc.Graph(
                            id="chart",
                            config={"displayModeBar": False, "locale": "de"},
                            responsive=True,
                            style={"height": f"{CHART_HEIGHT_PX}px"},
                        ),
                        class_name="chart-card",
                    ),
                    # Placeholder for a later extra (see docs: "Extras für später").
                    _card(
                        html.H2("Wetter und Wind"),
                        html.P("Kommt später: aktuelles Wetter, Windstärke und Windrichtung relativ zum Fluss.",
                               className="muted"),
                        class_name="weather-card",
                    ),
                ],
            ),
            html.Footer(
                className="footer",
                children=[
                    html.P(id="source-info", className="muted"),
                    html.P(
                        "Alle Angaben ohne Gewähr. Vor dem Ablegen die Lage am Steg selbst prüfen.",
                        className="muted",
                    ),
                ],
            ),
            dcc.Interval(id="refresh", interval=config.REFRESH_INTERVAL_MS),
        ],
    ),
])


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def _fmt_time(ts: pd.Timestamp) -> str:
    return to_local(ts).strftime("%H:%M")


def _fmt_datetime(ts: pd.Timestamp) -> str:
    return to_local(ts).strftime("%d.%m.%Y, %H:%M Uhr")


def _alerts(bundle: ForecastBundle) -> list:
    alerts = []
    if bundle.is_placeholder:
        alerts.append(html.Div(
            "Demo-Modus: Es ist noch kein Modell angeschlossen, alle Werte sind Platzhalter.",
            className="alert alert-info",
        ))
    if is_stale(bundle.issued_at):
        alerts.append(html.Div(
            f"Die letzte Vorhersage ist vom {_fmt_datetime(bundle.issued_at)} und damit veraltet.",
            className="alert alert-warn",
        ))
    return alerts


def _status(bundle: ForecastBundle) -> list:
    current = float(bundle.observed["level_cm"].iloc[-1])
    status = classify_level(current)
    later = value_at_lead(bundle.forecast, bundle.issued_at, 3)
    arrow, trend_text = trend(current, float(later["level_cm"])) if later is not None else ("", "unbekannt")
    return [
        html.Div(className=f"light light-{status.key}"),
        html.Div(
            className="status-text",
            children=[
                html.Div(status.label, className=f"status-label status-{status.key}"),
                html.Div([html.Span(f"{current:.0f}", className="level"), html.Span(" cm", className="unit")]),
                html.Div(f"{arrow} {trend_text} in den nächsten 3 Stunden", className="trend"),
                html.Div(status.detail, className="muted"),
                html.Div(f"Stand: {_fmt_datetime(bundle.observed['time'].iloc[-1])}", className="muted small"),
            ],
        ),
    ]


def _tiles(bundle: ForecastBundle) -> list:
    tiles = []
    for hours in config.SUMMARY_HORIZONS_H:
        row = value_at_lead(bundle.forecast, bundle.issued_at, hours)
        if row is None:
            continue
        level = float(row["level_cm"])
        status = classify_level(level)
        tiles.append(html.Div(
            className=f"tile tile-{status.key}",
            children=[
                html.Div(f"in {hours} h", className="tile-lead"),
                html.Div(f"{level:.0f} cm", className="tile-level"),
                html.Div(f"{_fmt_time(row['time'])} Uhr", className="tile-time"),
                html.Div(status.label, className="tile-status"),
            ],
        ))
    return tiles


def _local_naive(times: pd.Series) -> pd.Series:
    """Plotly shows naive timestamps as-is, so convert UTC to local wall time."""
    return times.dt.tz_convert(config.DISPLAY_TZ).dt.tz_localize(None)


def _figure(bundle: ForecastBundle, horizon_h: int) -> go.Figure:
    fc = bundle.forecast[bundle.forecast["time"] <= bundle.issued_at + pd.Timedelta(hours=horizon_h)]
    # Show as much history as forecast (capped), so short horizons stay readable.
    history_h = min(config.HISTORY_HOURS, horizon_h)
    obs = bundle.observed[bundle.observed["time"] >= bundle.issued_at - pd.Timedelta(hours=history_h)]
    fc_t, obs_t = _local_naive(fc["time"]), _local_naive(obs["time"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc_t, y=fc["upper_cm"], line={"width": 0}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=fc_t, y=fc["lower_cm"], line={"width": 0}, fill="tonexty",
        fillcolor="rgba(174, 218, 245, 0.6)", name="Unsicherheit", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=obs_t, y=obs["level_cm"], name="Gemessen", line={"color": "#222222", "width": 2},
        hovertemplate="%{y:.0f} cm<extra>Gemessen</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fc_t, y=fc["level_cm"], name="Vorhersage", line={"color": "#127fc5", "width": 2.5, "dash": "dash"},
        hovertemplate="%{y:.0f} cm<extra>Vorhersage</extra>",
    ))

    for y, text in ((config.LEVEL_LOW_CM, "Untergrenze"), (config.LEVEL_HIGH_CM, "Obergrenze")):
        fig.add_hline(y=y, line={"color": "#dc2626", "width": 1, "dash": "dot"},
                      annotation_text=text, annotation_position="top left")
    now_local = to_local(bundle.issued_at).tz_localize(None)
    fig.add_vline(x=now_local, line={"color": "#6b7280", "width": 1})
    fig.add_annotation(x=now_local, y=1, yref="paper", text="jetzt", showarrow=False, yanchor="bottom")

    fig.update_layout(
        margin={"l": 40, "r": 10, "t": 30, "b": 30},
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.15},
        yaxis_title="Wasserstand (cm)",
        separators=",.",
        font={"family": "Raleway, Helvetica Neue, Verdana, sans-serif", "color": "#222222"},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f3", tickformat="%d.%m. %H:%M")
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f3")
    return fig


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, plot_bgcolor="white")
    return fig


# --------------------------------------------------------------------------
# Callback
# --------------------------------------------------------------------------


@app.callback(
    Output("alerts", "children"),
    Output("status-content", "children"),
    Output("tiles", "children"),
    Output("chart", "figure"),
    Output("source-info", "children"),
    Input("refresh", "n_intervals"),
    Input("horizon", "value"),
)
def update(_n_intervals: int | None, horizon_h: int):
    """Fetch the latest forecast and re-render all dynamic parts.

    Any failure in the data layer is caught here, so the page stays usable and
    shows a clear message instead of a stack trace.
    """
    try:
        bundle = get_provider().get_latest()
        if bundle.observed.empty or bundle.forecast.empty:
            raise ValueError("Provider returned no data")
    except Exception:  # noqa: BLE001  (the UI must never crash)
        logger.exception("Could not load forecast data")
        error = html.Div("Die Daten können gerade nicht geladen werden. Bitte später erneut versuchen.",
                         className="alert alert-error")
        return [error], html.P("Keine Daten", className="muted"), [], _empty_figure("Keine Daten"), ""

    source = f"Quelle: {bundle.source} · Vorhersage erstellt {_fmt_datetime(bundle.issued_at)}"
    return _alerts(bundle), _status(bundle), _tiles(bundle), _figure(bundle, horizon_h), source


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
