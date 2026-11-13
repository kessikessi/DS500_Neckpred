"""Basic tests for the dashboard helpers and the placeholder data provider."""

import pandas as pd

from app import config
from app.data_provider import DummyProvider
from app.status import classify_level, is_stale, trend, value_at_lead


def test_classify_level_covers_all_zones():
    assert classify_level(config.LEVEL_LOW_CM - 1).key == "stop"
    assert classify_level(config.LEVEL_LOW_CM + 1).key == "caution"
    assert classify_level((config.LEVEL_LOW_CM + config.LEVEL_HIGH_CM) / 2).key == "ok"
    assert classify_level(config.LEVEL_HIGH_CM - 1).key == "caution"
    assert classify_level(config.LEVEL_HIGH_CM + 1).key == "stop"


def test_trend_direction():
    assert trend(100, 110)[1] == "steigend"
    assert trend(100, 90)[1] == "fallend"
    assert trend(100, 101)[1] == "gleichbleibend"


def test_dummy_provider_contract():
    bundle = DummyProvider().get_latest()
    assert bundle.issued_at.tz is not None and str(bundle.issued_at.tz) == "UTC"
    assert {"time", "level_cm"} <= set(bundle.observed.columns)
    assert {"time", "level_cm", "lower_cm", "upper_cm"} <= set(bundle.forecast.columns)
    assert (bundle.forecast["lower_cm"] <= bundle.forecast["upper_cm"]).all()
    assert bundle.forecast["time"].min() > bundle.issued_at
    assert value_at_lead(bundle.forecast, bundle.issued_at, max(config.CHART_HORIZONS_H)) is not None


def test_is_stale():
    now = pd.Timestamp("2026-11-13 12:00", tz="UTC")
    assert not is_stale(now - pd.Timedelta(minutes=10), now)
    assert is_stale(now - pd.Timedelta(hours=2), now)
