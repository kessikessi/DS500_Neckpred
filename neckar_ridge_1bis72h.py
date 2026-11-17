"""Neckar Kirchentellinsfurt: forecast every horizon from 1 to 72 hours.

Three matched variants:
  - persistence: W(t+h) = W(t)
  - Ridge (hydrological + weather + annual seasonality)
  - Ridge + clock/weekday/Baden-Wuerttemberg holiday features

Training: 2020-2024; chronological test: 2025 through 2026-08-01.
The SAME issue timestamps and training samples are used for every horizon,
so each comparison is aligned. Historical future measured rainfall is NEVER
used as a predictor; calendar values at the forecast target time are known.

IMPORTANT: LUBW station timestamps say MEZ. Weather time convention and
real-time publication delays still need verification before production/API use.
Calendar features do not establish actual weir operations or causality.

Usage:
  pip install pandas numpy scikit-learn python-dateutil matplotlib
  python neckar_ridge_1bis72h.py --data-dir path/to/data --output-dir results

Expected CSV filenames are listed in HYDRO and WEATHER_FILE below.
Outputs:
  ridge_1bis72h_metrics.csv: one row per horizon and model (72 x 3)
  ridge_1bis72h_comparison.png: MAE across all horizons
  ridge_1bis72h_example_forecast.csv: one full 72-point forecast issued at
      the last common test timestamp
  ridge_1bis72h_example_forecast.png: example prediction trajectory
  ridge_1bis72h_model_params.npz: parameters for transformed feature matrices
  ridge_1bis72h_feature_metadata.json: fitted transforms and column metadata

No remote API is called. Later replace load_observations() with an API-backed
reader returning the same hourly, typed, datetime-indexed columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.easter import easter
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

HORIZONS = tuple(range(1, 73))
HYDRO = {
    'Kirchentellinsfurt Wasserstand 1h(2).csv': 'target_w',
    'Horb Wasserstand(1).csv': 'horb_w',
    'Steinlach Wasserstand(1).csv': 'steinlach_w',
    'Ammer Wasserstand(1).csv': 'ammer_w',
    'Horb Abfluss(1).csv': 'horb_q',
    'Steinlach Abfluss(1).csv': 'steinlach_q',
}
WEATHER_FILE = 'weather_data_hourly_2000_2026.csv'
TRAIN_START = pd.Timestamp('2020-01-01 00:00')
TRAIN_END = pd.Timestamp('2025-01-01 00:00')
TEST_START = TRAIN_END
TEST_END = pd.Timestamp('2026-08-01 00:00')
ALPHA = 100.0


def read_hydro(path: Path, column: str) -> pd.DataFrame:
    f = pd.read_csv(
        path, sep=';', encoding='utf-8-sig',
        usecols=['Messzeitpunkt', 'Messwert'], dtype='string',
    )
    f['timestamp'] = pd.to_datetime(f['Messzeitpunkt'], format='%d.%m.%Y %H:%M', errors='coerce')
    f[column] = (pd.to_numeric(f['Messwert'].str.replace(',', '.', regex=False), errors='coerce')
                 .replace(-999, np.nan))
    return (f[['timestamp', column]].dropna(subset=['timestamp'])
            .drop_duplicates('timestamp', keep='last')
            .set_index('timestamp').sort_index().loc['2019-12-01':])


def read_weather(path: Path) -> pd.DataFrame:
    f = pd.read_csv(
        path, usecols=['station_id', 'timestamp', 'precipitation_amount', 'temperature_air_2m'],
        dtype={'station_id': 'string'}, parse_dates=['timestamp'],
    )
    f = f.loc[f.timestamp >= '2019-12-01']
    if f.duplicated(['timestamp', 'station_id']).any():
        raise ValueError('Duplicate weather station/timestamp combinations')
    w = f.set_index(['timestamp', 'station_id'])[['precipitation_amount', 'temperature_air_2m']].unstack('station_id')
    w.columns = [f'{var}_{station}' for var, station in w.columns]
    return w.sort_index()


def load_observations(data_dir: Path) -> pd.DataFrame:
    obs = None
    for filename, col in HYDRO.items():
        path = data_dir / filename
        if not path.exists():
            # Accept earlier uploaded alternative file version for the target.
            if col == 'target_w' and (data_dir / 'Kirchentellinsfurt Wasserstand 1h(1).csv').exists():
                path = data_dir / 'Kirchentellinsfurt Wasserstand 1h(1).csv'
            else:
                raise FileNotFoundError(f'Missing: {path}')
        frame = read_hydro(path, col)
        obs = frame if obs is None else obs.join(frame, how='left')
    obs = obs.join(read_weather(data_dir / WEATHER_FILE), how='left')
    # Retain target after end of meteorological observations so we can evaluate
    # forecasts issued on 2026-08-01 against true +72h future target values.
    return obs.asfreq('h')


def feature_engineering(obs: pd.DataFrame) -> pd.DataFrame:
    out: dict[str, pd.Series | np.ndarray] = {}
    for c in HYDRO.values():
        s = obs[c]
        for lag in (0, 1, 2, 3, 6, 12, 24, 48, 72):
            out[f'{c}_lag{lag}'] = s.shift(lag)
        for lag in (1, 3, 6, 12, 24):
            out[f'{c}_delta{lag}'] = s - s.shift(lag)
        for w in (6, 24):
            out[f'{c}_rolling{w}'] = s.rolling(w, min_periods=w).mean()
    for c in obs.columns:
        if c.startswith('precipitation_amount_'):
            for window in (3, 6, 12, 24, 72, 168):
                out[f'{c}_rain{window}'] = obs[c].rolling(window, min_periods=window).sum()
        elif c.startswith('temperature_air_2m_'):
            out[f'{c}_last'] = obs[c]
            out[f'{c}_mean24'] = obs[c].rolling(24, min_periods=24).mean()
    t = obs.index
    out['annual_sin'] = np.sin(2 * np.pi * t.dayofyear / 365.25)
    out['annual_cos'] = np.cos(2 * np.pi * t.dayofyear / 365.25)
    return pd.DataFrame(out, index=t).astype('float32')


def bw_holidays(year: int) -> set[pd.Timestamp]:
    e = pd.Timestamp(easter(year))
    fixed = [(1, 1), (1, 6), (5, 1), (10, 3), (11, 1), (12, 25), (12, 26)]
    return ({pd.Timestamp(year=year, month=m, day=d) for m, d in fixed}
            | {e + pd.Timedelta(days=d) for d in (-2, 1, 39, 50, 60)})


def calendar_features(issue_idx: pd.DatetimeIndex, h: int, holiday_dates: set[pd.Timestamp]) -> pd.DataFrame:
    target_t = issue_idx + pd.Timedelta(hours=h)
    holiday = target_t.normalize().isin(holiday_dates)
    out: dict[str, np.ndarray] = {}
    for p, t in [('issue', issue_idx), ('target', target_t)]:
        out[f'{p}_hour_sin'] = np.sin(2 * np.pi * t.hour / 24)
        out[f'{p}_hour_cos'] = np.cos(2 * np.pi * t.hour / 24)
        out[f'{p}_dow_sin'] = np.sin(2 * np.pi * t.dayofweek / 7)
        out[f'{p}_dow_cos'] = np.cos(2 * np.pi * t.dayofweek / 7)
    out['target_is_weekend'] = (target_t.dayofweek >= 5).astype(float)
    out['target_is_BW_holiday'] = holiday.astype(float)
    out['target_weekday_morning'] = (
        (target_t.dayofweek < 5) & (target_t.hour >= 8) &
        (target_t.hour < 12) & (~holiday)
    ).astype(float)
    return pd.DataFrame(out, index=issue_idx, dtype='float32')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=Path('ridge_1bis72h_results'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print('Reading original weather and hydrology data ...', flush=True)
    obs = load_observations(args.data_dir)
    features = feature_engineering(obs)
    t = features.index
    target = obs['target_w']

    # All target horizons use an identical issue-time sample. This avoids a
    # confounding issue-time composition when comparing MAE across horizons.
    ymat = np.column_stack([target.shift(-h).to_numpy(dtype='float64') for h in HORIZONS])
    valid_all_targets = np.isfinite(ymat).all(axis=1) & target.notna().to_numpy()
    # Leave a 72h embargo at the end of training, so *no* training label is
    # located in the test period, even for the longest forecast horizon.
    train_mask = (t >= TRAIN_START) & (t + pd.Timedelta(hours=72) < TRAIN_END) & valid_all_targets
    test_mask = (t >= TEST_START) & (t <= TEST_END) & valid_all_targets
    if train_mask.sum() < 1000 or test_mask.sum() < 100:
        raise RuntimeError('Insufficient valid rows. Check input files and time boundaries.')
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    print(f'Common training timestamps: {n_train:,}; test timestamps: {n_test:,}', flush=True)

    # Fit preprocessing strictly on training data; also reuse it at inference.
    feature_names = features.columns.tolist()
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    x_train = imputer.fit_transform(features.loc[train_mask])
    x_test = imputer.transform(features.loc[test_mask])
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    y_train = ymat[train_mask]
    y_test = ymat[test_mask]
    base = Ridge(alpha=ALPHA, solver='cholesky')
    base.fit(x_train, y_train) # 72 targets in one multioutput Ridge fit
    pred_base = base.predict(x_test)
    print('Ridge without calendar fitted for all 72 horizons.', flush=True)

    # For the time model, fit a distinct ridge for every h, because target
    # weekday / holiday / working-hour indicators depend on the forecast target.
    holiday_dates = set().union(*(bw_holidays(year) for year in range(t.min().year, t.max().year + 2)))
    idx_train = t[train_mask]
    idx_test = t[test_mask]
    pred_calendar = np.empty_like(y_test)
    cal_coeffs = []
    cal_intercepts = []
    cal_means = []
    cal_scales = []
    cal_columns = None
    for j, h in enumerate(HORIZONS):
        c_train = calendar_features(idx_train, h, holiday_dates)
        c_test = calendar_features(idx_test, h, holiday_dates)
        if cal_columns is None:
            cal_columns = c_train.columns.tolist()
        cal_scaler = StandardScaler()
        c_train_trans = cal_scaler.fit_transform(c_train)
        c_test_trans = cal_scaler.transform(c_test)
        m = Ridge(alpha=ALPHA, solver='cholesky')
        m.fit(np.column_stack((x_train, c_train_trans)), y_train[:, j])
        pred_calendar[:, j] = m.predict(np.column_stack((x_test, c_test_trans)))
        cal_coeffs.append(m.coef_)
        cal_intercepts.append(m.intercept_)
        cal_means.append(cal_scaler.mean_)
        cal_scales.append(cal_scaler.scale_)
        if h in (1, 8, 24, 48, 72):
            print(f'Calendar +{h}h done', flush=True)

    current = target.to_numpy(dtype='float64')[test_mask]
    pred_persistence = np.broadcast_to(current[:, None], y_test.shape)
    rows = []
    for j, h in enumerate(HORIZONS):
        real = y_test[:, j]
        strong = np.abs(real - current) >= 10
        for name, pred in [('Persistence', pred_persistence[:, j]),
                           ('Ridge ohne Zeit', pred_base[:, j]),
                           ('Ridge mit Zeit', pred_calendar[:, j])]:
            abs_err = np.abs(real - pred)
            rows.append({
                'horizon_h': h,
                'model': name,
                'MAE_cm': float(mean_absolute_error(real, pred)),
                'RMSE_cm': float(np.sqrt(mean_squared_error(real, pred))),
                'bias_cm': float(np.mean(pred - real)),
                'MAE_starke_Aenderung_ab10cm': float(np.mean(abs_err[strong])) if strong.any() else np.nan,
                'n_starke_Aenderungen': int(strong.sum()),
                'n_train': n_train if name != 'Persistence' else 0,
                'n_test': n_test,
            })
    score = pd.DataFrame(rows)
    score.to_csv(args.output_dir / 'ridge_1bis72h_metrics.csv', index=False, float_format='%.6f')

    # Example: all predictions issued on the last test timestamp.
    issue = idx_test[-1]
    example = pd.DataFrame({
        'issue_time': issue,
        'forecast_for': [issue + pd.Timedelta(hours=h) for h in HORIZONS],
        'horizon_h': HORIZONS,
        'actual_water_level_cm': y_test[-1, :],
        'persistence_cm': pred_persistence[-1, :],
        'ridge_ohne_zeit_cm': pred_base[-1, :],
        'ridge_mit_zeit_cm': pred_calendar[-1, :],
    })
    example.to_csv(args.output_dir / 'ridge_1bis72h_example_forecast.csv', index=False, float_format='%.4f')

    # Export compact fitted coefficients and preprocessing metadata. API adapter
    # must create identical normalized features in the same feature order.
    np.savez_compressed(
        args.output_dir / 'ridge_1bis72h_model_params.npz',
        base_coefs=base.coef_, base_intercepts=base.intercept_,
        calendar_coefs=np.asarray(cal_coeffs),
        calendar_intercepts=np.asarray(cal_intercepts),
        calendar_means=np.asarray(cal_means), calendar_scales=np.asarray(cal_scales),
        imputing_values=imputer.statistics_,
        base_means=scaler.mean_, base_scales=scaler.scale_,
    )
    (args.output_dir / 'ridge_1bis72h_feature_metadata.json').write_text(
        json.dumps({
            'forecast_horizons_h': list(HORIZONS), 'alpha': ALPHA,
            'training_range': '2020-01-01 to 2024-12-28 inclusive (72h embargo)',
            'test_range': '2025-01-01 to 2026-08-01',
            'feature_columns': feature_names, 'calendar_columns': cal_columns,
            'timezone_warning': 'LUBW MEZ vs weather timestamp time basis unverified',
            'prediction_at_t_plus_h': 'water level in centimeters (point forecast)',
            'data_source_adapter': 'Any API adapter must return identical hourly columns before feature_engineering',
        }, indent=2, ensure_ascii=False), encoding='utf-8',
    )

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name in ('Persistence', 'Ridge ohne Zeit', 'Ridge mit Zeit'):
        z = score.loc[score.model == name]
        ax.plot(z.horizon_h, z.MAE_cm, label=name, linewidth=1.9)
    ax.set(xlabel='Prognosehorizont (Stunden)', ylabel='MAE (cm)',
           title='Neckar Kirchentellinsfurt: 1 bis 72 Stunden (Test 2025–2026)')
    ax.set_xlim(1, 72)
    ax.grid(alpha=.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / 'ridge_1bis72h_comparison.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(example.horizon_h, example.actual_water_level_cm, label='Tatsaechlich', linewidth=2.3)
    ax.plot(example.horizon_h, example.persistence_cm, label='Persistence')
    ax.plot(example.horizon_h, example.ridge_ohne_zeit_cm, label='Ridge ohne Zeit')
    ax.plot(example.horizon_h, example.ridge_mit_zeit_cm, label='Ridge mit Zeit')
    ax.set(xlabel='Prognosehorizont (Stunden)', ylabel='Wasserstand (cm)',
           title=f'Beispielprognose – Ausgabe {issue:%Y-%m-%d %H:%M}')
    ax.set_xlim(1, 72)
    ax.grid(alpha=.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / 'ridge_1bis72h_example_forecast.png', dpi=180)
    plt.close(fig)

    selected = score.loc[score.horizon_h.isin((1, 3, 8, 12, 24, 36, 48, 60, 72))]
    print('\nMAE for selected horizons (cm):')
    print(selected.pivot(index='horizon_h', columns='model', values='MAE_cm').round(3).to_string())
    print(f'\nAll results saved to {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
