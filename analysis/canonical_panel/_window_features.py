#!/usr/bin/env python3
"""
Window-level CGM features for the canonical lab-CGM panel.

Generalises `window_metrics` in analysis/reliability/_window_structure.py
(seven contiguous dense days) to a variable-length, possibly non-contiguous set of dense
participant-local calendar days:

- every dense day is a 96-cell fifteen-minute grid (cell mean of raw readings 20-500);
- whole bounded missing runs of <=2 cells (30 min of missing support; 45 min
  between endpoints) are linearly interpolated, only WITHIN consecutive-day runs;
  longer missing runs remain wholly missing, with no edge extrapolation;
- MODD is the mean absolute difference between same-cell values on consecutive-calendar-day
  pairs only;
- autocorrelation at lags 1-10 cells is computed over the concatenated dense-day series with
  lagged pairs that cross a non-adjacent-day boundary excluded (implemented by inserting a
  ten-cell NaN spacer between non-adjacent runs, so that `autocorrelation` drops the
  crossing pairs); every lag interval must also have finite support throughout;
  AC_Var is the sample variance across the ten lags - identical to
  analysis/reliability/_window_structure.py when the days are contiguous;
- GMI / TIR / TAR / TBR / J-index / skewness / kurtosis / fasting-window (04:00-07:59 local)
  / dawn effect (06:00-08:59 minus 02:00-04:59 local) mirror
  analysis/metabolic/cgm_features.py;
- MAGE is the mean supplied daily gap-bounded raw turning-amplitude proxy over dense days
  (the canonical raw-reading turning-point definition).

The numeric helpers and gate constants are imported from
analysis/reliability/_window_structure.py so the definitions are shared rather than re-derived.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, relative_path: str):
    """Import a numbered analysis script (not a package) as a module."""
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_window_structure_module = load_script(
    "window_structure_for_window_features", "analysis/reliability/_window_structure.py"
)
_trace_features = load_script(
    "canonical_cgm_trace_features", "cgmpaper/cgm_trace_features.py"
)

RAW_DAILY_GATE = _window_structure_module.RAW_DAILY_GATE      # 230 raw readings per local day
CELLS_PER_DAY = _window_structure_module.CELLS_PER_DAY        # 96 fifteen-minute cells
MIN_15M_CELLS = _window_structure_module.MIN_15M_CELLS        # 77 observed cells per day
LAB_SPECS = _window_structure_module.LAB_SPECS
DERIVED_SPECS = _window_structure_module.DERIVED_SPECS
_grid_metrics = load_script("canonical_cgm_grid_metrics", "cgmpaper/cgm_grid_metrics.py")
GRID_FEATURE_VERSION = _grid_metrics.GRID_FEATURE_VERSION
CORE_GRID_FEATURE_VERSION = GRID_FEATURE_VERSION
interpolate_short_gaps = _grid_metrics.interpolate_short_gaps
paired_difference = _grid_metrics.paired_difference
autocorrelation = _grid_metrics.autocorrelation
extract_rich_features = _trace_features.extract_rich_features
RICH_MIN_READINGS = _trace_features.MIN_READINGS
RICH_FEATURE_VERSION = _trace_features.FEATURE_VERSION

AC_LAGS = range(1, 11)
BLOCK_SPACER = len(AC_LAGS)          # NaN cells inserted between non-adjacent day runs
FASTING_CELLS = slice(4 * 4, 8 * 4)  # 04:00–07:59 local
DAWN_LATE_CELLS = slice(6 * 4, 9 * 4)   # 06:00–08:59 local
DAWN_EARLY_CELLS = slice(2 * 4, 5 * 4)  # 02:00–04:59 local

WINDOW_METRIC_COLUMNS = [
    "mean_glucose", "sd_glucose", "cv_glucose", "median_glucose",
    "p10_glucose", "p25_glucose", "p75_glucose", "p90_glucose", "iqr_glucose",
    "gmi", "tir_70_180", "tir_70_140", "tar_140", "tar_180", "tar_250", "tbr_70", "tbr_54",
    "mage", "modd", "j_index", "skewness", "kurtosis", "mean_fasting_glucose",
    "dawn_effect", "day_mean_sd",
] + [f"ac_lag_{lag}" for lag in AC_LAGS] + ["ac_mean", "ac_var"]
W30_COLUMNS = ["n_dense_days", "mean_glucose", "sd_glucose", "cv_glucose", "modd", "ac_var"]


def contiguous_blocks(day_numbers: np.ndarray) -> list[np.ndarray]:
    """Split sorted integer day numbers into runs of consecutive calendar days.

    Returns index arrays into the input, one per run."""
    day_numbers = np.asarray(day_numbers, dtype=np.int64)
    if len(day_numbers) == 0:
        return []
    breaks = np.flatnonzero(np.diff(day_numbers) != 1) + 1
    return np.split(np.arange(len(day_numbers)), breaks)


def window_metrics(day_numbers: np.ndarray, grid: np.ndarray, daily_mage: np.ndarray) -> dict:
    """CGM metrics for one participant window.

    day_numbers : sorted integer calendar days (days since epoch) of the dense days
    grid        : (n_days, 96) raw cell means, NaN where the cell is unobserved
    daily_mage  : (n_days,) ETL per-day MAGE for the same days (NaN allowed)
    """
    day_numbers = np.asarray(day_numbers, dtype=np.int64)
    grid = np.asarray(grid, dtype=float)
    n_days = len(day_numbers)
    assert grid.shape == (n_days, CELLS_PER_DAY)
    assert np.all(np.diff(day_numbers) > 0), "dense days must be sorted and unique"

    # Interpolate short gaps within each run of consecutive days (never across a
    # non-adjacent boundary); build the spaced series used for lagged statistics.
    blocks = contiguous_blocks(day_numbers)
    day_grid = np.full_like(grid, np.nan)
    spaced: list[np.ndarray] = []
    for block in blocks:
        flat = interpolate_short_gaps(grid[block].reshape(-1), limit=2)
        day_grid[block] = flat.reshape(len(block), CELLS_PER_DAY)
        if spaced:
            spaced.append(np.full(BLOCK_SPACER, np.nan))
        spaced.append(flat)
    series = np.concatenate(spaced)
    finite = series[np.isfinite(series)]

    ac = np.array([autocorrelation(series, lag) for lag in AC_LAGS], dtype=float)

    day_means = np.nanmean(day_grid, axis=1)
    day_pair_diffs: list[float] = []
    for index in range(n_days - 1):
        if day_numbers[index + 1] - day_numbers[index] != 1:
            continue
        left, right = day_grid[index], day_grid[index + 1]
        pair_ok = np.isfinite(left) & np.isfinite(right)
        day_pair_diffs.extend(np.abs(right[pair_ok] - left[pair_ok]).tolist())

    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1))
    fasting = day_grid[:, FASTING_CELLS]
    dawn_late = day_grid[:, DAWN_LATE_CELLS]
    dawn_early = day_grid[:, DAWN_EARLY_CELLS]
    record = {
        "n_dense_days": int(n_days),
        "n_contiguous_runs": int(len(blocks)),
        "n_15m_observed": int(np.isfinite(grid).sum()),
        "n_15m_interpolated": int(np.isfinite(day_grid).sum() - np.isfinite(grid).sum()),
        "mean_glucose": mean,
        "sd_glucose": sd,
        "cv_glucose": 100.0 * sd / mean if mean > 0 else np.nan,
        "median_glucose": float(np.median(finite)),
        "p10_glucose": float(np.quantile(finite, 0.10)),
        "p25_glucose": float(np.quantile(finite, 0.25)),
        "p75_glucose": float(np.quantile(finite, 0.75)),
        "p90_glucose": float(np.quantile(finite, 0.90)),
        "iqr_glucose": float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)),
        "gmi": 3.31 + 0.02392 * mean,
        "tir_70_180": 100.0 * float(np.mean((finite >= 70.0) & (finite <= 180.0))),
        "tir_70_140": 100.0 * float(np.mean((finite >= 70.0) & (finite <= 140.0))),
        "tar_140": 100.0 * float(np.mean(finite > 140.0)),
        "tar_180": 100.0 * float(np.mean(finite > 180.0)),
        "tar_250": 100.0 * float(np.mean(finite > 250.0)),
        "tbr_70": 100.0 * float(np.mean(finite < 70.0)),
        "tbr_54": 100.0 * float(np.mean(finite < 54.0)),
        "mage": float(np.nanmean(daily_mage)) if np.isfinite(daily_mage).any() else np.nan,
        "modd": float(np.mean(day_pair_diffs)) if day_pair_diffs else np.nan,
        "n_modd_day_pairs": int(sum(
            1 for index in range(n_days - 1) if day_numbers[index + 1] - day_numbers[index] == 1
        )),
        "j_index": 0.001 * (mean + sd) ** 2,
        "skewness": float(stats.skew(finite, bias=False)),
        "kurtosis": float(stats.kurtosis(finite, fisher=True, bias=False)),
        "mean_fasting_glucose": float(np.nanmean(fasting)) if np.isfinite(fasting).any() else np.nan,
        "dawn_effect": (
            float(np.nanmean(dawn_late) - np.nanmean(dawn_early))
            if np.isfinite(dawn_late).any() and np.isfinite(dawn_early).any() else np.nan
        ),
        "day_mean_sd": float(np.std(day_means, ddof=1)) if n_days > 1 else np.nan,
        "ac_mean": float(np.mean(ac)) if np.isfinite(ac).all() else np.nan,
        "ac_var": float(np.var(ac, ddof=1)) if np.isfinite(ac).all() else np.nan,
    }
    for lag, value in zip(AC_LAGS, ac):
        record[f"ac_lag_{lag}"] = float(value)
    return record


def rich_features(ts_minutes: np.ndarray, glucose: np.ndarray, tz_offset_hours: float) -> dict | None:
    """Shared gap-aware dynamics on the raw UTC readings of one window.

    Pass actual timestamps, never concatenated day arrays or fifteen-minute
    cell timestamps: the shared extractor detects missing five-minute cells.
    Its `n_readings` is renamed `rich_n_readings` because the panel already
    carries `n_readings` (raw readings on the dense days) as an identifier column.
    """
    feats = extract_rich_features(
        np.asarray(ts_minutes, dtype=float), np.asarray(glucose, dtype=float),
        tz_offset_hours=float(tz_offset_hours),
    )
    if feats is None:
        return None
    feats["rich_n_readings"] = feats.pop("n_readings")
    # The panel's existing n_contiguous_runs counts consecutive *calendar days*;
    # this new diagnostic counts uninterrupted observed five-minute cells.
    feats["rich_n_contiguous_runs"] = feats.pop("n_contiguous_runs")
    return feats
