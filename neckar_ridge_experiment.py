#!/usr/bin/env python3
"""DS500 – eigener Ridge-Prognoseversuch fuer den Pegel Kirchentellinsfurt.

Eigenstaendige Python-Datei fuer ein BESTEHENDES GitHub-Projekt.
Keine Imports aus anderen Projektmodulen erforderlich.

Installation:
    pip install numpy pandas scikit-learn joblib

Ausfuehren (CSV-Dateien lokal in data/ ablegen):
    python neckar_ridge_experiment.py --data-dir data --output-dir results/ridge_experiment

Prognosehorizonte: 1, 3, 8, 24, 48 und 72 Stunden.
Train: 2020–2023; Validation: 2024; Test: 2025–01.08.2026.

WICHTIG: Pilotversuch, keine validierte operative Wasserstandsvorhersage.
Der Zeitbezug (LUBW 'MEZ' / Wetterdaten) und die tatsaechliche
Verfuegbarkeit der Stundenmittelwerte muessen vor dem Livebetrieb
geprueft werden. Ausschliesslich vergangene Wetterbeobachtungen,
keine kuenftigen tatsaechlich gemessenen Wetterwerte, werden benutzt.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HORIZONS = (1, 3, 8, 24, 48, 72)

# Exakt die Datenquellen des bisherigen Ridge-Pilotversuchs.
HYDRO_FILES = {
    "Horb Wasserstand(1).csv": "horb_w",
    "Steinlach Wasserstand(1).csv": "steinlach_w",
    "Ammer Wasserstand(1).csv": "ammer_w",
    "Horb Abfluss(1).csv": "horb_q",
    "Steinlach Abfluss(1).csv": "steinlach_q",
}


def read_hydro(path: Path, column: str) -> pd.DataFrame:
    """LUBW CSV: Semikolon, dd.mm.yyyy HH:MM, Dezimalkomma, -999=fehlend."""
    raw = pd.read_csv(
        path, sep=";", encoding="utf-8-sig",
        usecols=["Messzeitpunkt", "Messwert"], dtype="string",
    )
    raw["time"] = pd.to_datetime(
        raw["Messzeitpunkt"], format="%d.%m.%Y %H:%M", errors="raise"
    )
    raw[column] = pd.to_numeric(
        raw["Messwert"].str.strip().str.replace(",", ".", regex=False),
        errors="coerce",
    ).replace(-999, np.nan)
    if raw["time"].duplicated().any():
        warnings.warn(f"Doppelte Zeitstempel in {path.name}; letzter Wert gilt")
    return (
        raw[["time", column]]
        .drop_duplicates("time", keep="last")
        .set_index("time")
        .sort_index()
    )


def read_weather(path: Path) -> pd.DataFrame:
    """Stundenwetter pro Station in breite Tabelle umwandeln."""
    raw = pd.read_csv(path, dtype={"station_id": str}, parse_dates=["timestamp"])
    columns = [
        "precipitation_amount", "temperature_air_2m",
        "humidity_relative", "pressure_vapor",
    ]
    long = raw.set_index(["timestamp", "station_id"])[columns]
    if long.index.duplicated().any():
        raise ValueError("Doppelte Kombinationen aus Zeitstempel und Wetterstation")
    wide = long.unstack("station_id")
    wide.columns = [f"{variable}_{sid}" for variable, sid in wide.columns]
    return wide.sort_index()


def load_observations(data_dir: Path, target_file: str, end_date: str) -> pd.DataFrame:
    """Einheitliche Stundenzeitreihe – diese Schicht kann spaeter API-Daten nutzen."""
    target = data_dir / target_file
    if not target.exists():
        alternatives = sorted(data_dir.glob("Kirchentellinsfurt Wasserstand 1h*.csv"))
        if len(alternatives) == 1:
            target = alternatives[0]
        else:
            raise FileNotFoundError(f"Zieldatei fehlt: {target}")
    obs = read_hydro(target, "target_w").loc["2019-12-01":]
    for filename, col in HYDRO_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Eingabedatei fehlt: {path}")
        obs = obs.join(read_hydro(path, col).loc["2019-12-01":], how="left")

    weather_file = data_dir / "weather_data_hourly_2000_2026.csv"
    if not weather_file.exists():
        raise FileNotFoundError(f"Wetterdatei fehlt: {weather_file}")
    weather = read_weather(weather_file).loc["2019-12-01":]
    obs = obs.join(weather, how="left").asfreq("h")
    # Gleiches Datenfenster wie im Pilot; kein kuenftiges gemessenes Wetter.
    return obs.loc[: min(pd.Timestamp(end_date), weather.index.max())]


def build_features(obs: pd.DataFrame) -> pd.DataFrame:
    """Alle Features zu Zeitpunkt t bestehen aus Beobachtungen bis maximal t."""
    feats: dict[str, pd.Series | np.ndarray] = {}
    for col in ["target_w", *HYDRO_FILES.values()]:
        s = obs[col]
        for lag in (0, 1, 2, 3, 6, 12, 24, 48, 72):
            feats[f"{col}_lag{lag}"] = s.shift(lag)
        for lag in (1, 3, 6, 12, 24):
            feats[f"{col}_change{lag}"] = s - s.shift(lag)
        feats[f"{col}_mean6"] = s.rolling(6, min_periods=6).mean()
        feats[f"{col}_mean24"] = s.rolling(24, min_periods=24).mean()
    for col in obs.columns:
        if col.startswith("precipitation_amount_"):
            for lag in (3, 6, 12, 24, 72, 168):
                feats[f"{col}_rain{lag}"] = obs[col].rolling(lag, min_periods=lag).sum()
        elif col.startswith("temperature_air_2m_"):
            feats[f"{col}_last"] = obs[col]
            feats[f"{col}_mean24"] = obs[col].rolling(24, min_periods=24).mean()
    idx = obs.index
    feats["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    feats["year_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    feats["year_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    return pd.DataFrame(feats, index=idx, dtype="float32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ridge_experiment"))
    parser.add_argument("--target-file", default="Kirchentellinsfurt Wasserstand 1h(2).csv")
    parser.add_argument("--end-date", default="2026-08-01 00:00")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Lade und kombiniere historische Pegel- und Wetterdaten ...", flush=True)
    obs = load_observations(args.data_dir, args.target_file, args.end_date)
    X = build_features(obs).loc["2020-01-01":]
    current = obs["target_w"].reindex(X.index)
    rows = []
    predictions = []
    val_start = pd.Timestamp("2024-01-01")
    test_start = pd.Timestamp("2025-01-01")

    for h in HORIZONS:
        # Punktprognose: W(t+h), NICHT Mittelwert der naechsten h Stunden.
        future = obs["target_w"].shift(-h).reindex(X.index)
        origin = X.index
        target_time = origin + pd.Timedelta(hours=h)
        valid = future.notna() & current.notna()

        train = (origin < val_start) & (target_time < val_start) & valid
        validation = (
            (origin >= val_start) & (origin < test_start)
            & (target_time < test_start) & valid
        )
        train_val = (origin < test_start) & (target_time < test_start) & valid
        test = (origin >= test_start) & (target_time <= origin.max()) & valid
        if min(train.sum(), validation.sum(), test.sum()) == 0:
            raise ValueError(f"Kein ausreichend ueberlappender Datensatz fuer {h} Stunden")

        ridge = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=100, solver="lsqr", tol=0.002),
        )
        ridge.fit(X.loc[train], future.loc[train])
        val_forecast = ridge.predict(X.loc[validation])
        val_ridge = float(mean_absolute_error(future.loc[validation], val_forecast))
        val_baseline = float(mean_absolute_error(future.loc[validation], current.loc[validation]))

        # Auswahl NUR anhand der Validation, nicht anhand der Testdaten.
        selected = "Ridge" if val_ridge < val_baseline else "Persistence"
        model_file = ""
        if selected == "Ridge":
            ridge.fit(X.loc[train_val], future.loc[train_val])
            forecast = ridge.predict(X.loc[test])
            model_file = f"ridge_{h}h.joblib"
            dump(ridge, args.output_dir / model_file)
        else:
            forecast = current.loc[test].to_numpy()

        actual = future.loc[test].to_numpy()
        baseline = current.loc[test].to_numpy()
        threshold = float(future.loc[train].quantile(0.95))
        high_water = actual >= threshold
        high_water_mae = (
            float(mean_absolute_error(actual[high_water], np.asarray(forecast)[high_water]))
            if high_water.any() else None
        )
        summary = {
            "horizon_h": h,
            "selected_model": selected,
            "train_n": int(train.sum()),
            "validation_n": int(validation.sum()),
            "test_n": int(test.sum()),
            "validation_ridge_MAE_cm": round(val_ridge, 3),
            "validation_persistence_MAE_cm": round(val_baseline, 3),
            "test_selected_MAE_cm": round(float(mean_absolute_error(actual, forecast)), 3),
            "test_persistence_MAE_cm": round(float(mean_absolute_error(actual, baseline)), 3),
            "test_highwater_n": int(high_water.sum()),
            "test_highwater_MAE_cm": round(high_water_mae, 3) if high_water_mae is not None else None,
            "model_file": model_file,
        }
        rows.append(summary)
        predictions.append(pd.DataFrame({
            "issue_at": origin[test], "target_at": target_time[test],
            "horizon_h": h, "actual_cm": actual,
            "forecast_cm": forecast, "persistence_cm": baseline,
        }))
        print(f"{h:>2}h: {selected}, MAE={summary['test_selected_MAE_cm']} cm", flush=True)

    pd.DataFrame(rows).to_csv(args.output_dir / "ridge_scores.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / "ridge_predictions.csv", index=False
    )
    (args.output_dir / "ridge_feature_manifest.json").write_text(
        json.dumps({
            "target": "Kirchentellinsfurt water level (cm)",
            "horizons_h": list(HORIZONS),
            "feature_columns": X.columns.tolist(),
            "data_sources": ["target_w", *HYDRO_FILES.values(), "weather_hourly"],
            "timebase_warning": "LUBW MEZ versus weather timestamps: verify alignment before live use.",
            "availability_warning": "Hour-mean t may be released after t: respect publication delays.",
            "api_integration": "Replace load_observations() with a provider that returns the same column names and hourly index.",
        }, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Fertig. Ergebnisse liegen in: {args.output_dir}")


if __name__ == "__main__":
    main()
