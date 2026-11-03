"""Gap-aware CGM dynamics features computed on observed five-minute cells only.

Features are computed strictly within uninterrupted observed runs and never
interpolated or joined across missing cells. Input timestamps are UTC epoch
minutes and glucose is mg/dL. These features describe the supplied observation
window; callers choose the cohort and window.

Features include: mean/SD/CV of glucose, coverage and wear-time statistics,
lag autocorrelations (30m/1h/2h/6h/24h) within uninterrupted runs, a
window-count-weighted Welch spectrum over complete 24-hour windows, velocity
and acceleration statistics, diurnal (morning/midday/evening/night) means and
amplitude, day-to-day SD of local-time daily means, and peak/recovery
statistics (peaks defined as >=1 SD above the trace mean, with recovery to
the mean observed within 6 hours and before the next peak or run end).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

FEATURE_VERSION = "gap_aware_observed_5m_v1"
MIN_READINGS = 500
CELL_MINUTES = 5.0
VELOCITY_HIGH = 3.0
LAG_MINUTES = {"acf_30m": 30, "acf_1h": 60, "acf_2h": 120,
               "acf_6h": 360, "acf_24h": 1440}
SPECTRAL_WINDOW_CELLS = 288  # fixed 24-hour windows, never joined across gaps
SPECTRAL_OVERLAP_CELLS = 144


def _observed_grid(t, g):
    """Reject invalid values; average duplicate timestamps then observed cells."""
    t, g = np.asarray(t, dtype=float), np.asarray(g, dtype=float)
    if t.ndim != 1 or g.ndim != 1 or t.shape != g.shape:
        raise ValueError("t and g must be one-dimensional arrays of equal length")
    valid = np.isfinite(t) & np.isfinite(g) & (g >= 20) & (g <= 500)
    t, g = t[valid], g[valid]
    n_valid = len(t)
    if not n_valid:
        return np.array([], dtype=np.int64), np.array([]), t, n_valid
    # Sorting and collapsing duplicates prevents zero-duration derivatives and
    # repeated upload rows from changing the observation or eligibility weights.
    t, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    g = np.bincount(inverse, weights=g) / counts
    bins, inverse, counts = np.unique(np.floor(t / CELL_MINUTES).astype(np.int64),
                                      return_inverse=True, return_counts=True)
    glucose = np.bincount(inverse, weights=g) / counts
    return bins, glucose, t, n_valid


def _runs(bins):
    return np.split(np.arange(len(bins)), np.flatnonzero(np.diff(bins) != 1) + 1)


def _lag_correlation(glucose, runs, lag):
    """Pearson correlation of fixed-lag pairs inside uninterrupted runs."""
    eligible = [r for r in runs if len(r) > lag]
    if not eligible:
        return np.nan, 0
    a = np.concatenate([glucose[r[:-lag]] for r in eligible])
    b = np.concatenate([glucose[r[lag:]] for r in eligible])
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan, len(a)
    return float(np.corrcoef(a, b)[0, 1]), len(a)


def _spectrum(glucose, runs):
    """Window-count-weighted Welch spectrum of complete uninterrupted days."""
    total, n_windows = None, 0
    for run in runs:
        if len(run) < SPECTRAL_WINDOW_CELLS:
            continue
        count = 1 + (len(run) - SPECTRAL_WINDOW_CELLS) // (
            SPECTRAL_WINDOW_CELLS - SPECTRAL_OVERLAP_CELLS)
        frequencies, power = signal.welch(
            glucose[run], fs=1 / CELL_MINUTES, window="hann",
            nperseg=SPECTRAL_WINDOW_CELLS, noverlap=SPECTRAL_OVERLAP_CELLS,
            detrend="constant", scaling="density", average="mean",
        )
        total = power * count if total is None else total + power * count
        n_windows += count
    if total is None:
        return np.nan, np.nan, 0
    total[0] = 0.0
    if total.sum() <= 0:
        return np.nan, np.nan, n_windows
    peak = float(frequencies[np.argmax(total)])
    low = (frequencies > 0) & (frequencies < 1 / 240.0)
    return peak, float(total[low].sum() / total.sum()), n_windows


def extract_rich_features(t, g, tz_offset_hours=0.0, *, min_readings=MIN_READINGS):
    """Return gap-aware features, or None below the unique-reading gate.

    Five-minute UTC bins are half-open, [5*k, 5*(k+1)); only bins containing
    accepted readings exist. No short or long gap is interpolated. A cell
    represents five minutes of observed support (not known continuous wear).
    Time-of-day uses cell centers shifted by the explicitly supplied fixed
    offset. Missing offsets are not silently replaced with UTC.
    """
    if not isinstance(min_readings, (int, np.integer)) or min_readings < 1:
        raise ValueError("min_readings must be a positive integer")
    bins, glucose, timestamps, n_valid = _observed_grid(t, g)
    if len(timestamps) < min_readings or not len(bins):
        return None
    runs = _runs(bins)
    mean = float(glucose.mean())
    sd = float(glucose.std(ddof=0))
    observed_minutes = len(bins) * CELL_MINUTES
    wear_days = observed_minutes / 1440.0
    elapsed_cells = int(bins[-1] - bins[0] + 1)
    f = {
        "trace_mean_glucose": mean,
        "trace_sd_glucose": sd,
        "trace_cv_glucose": 100 * sd / mean,
        "wear_days": wear_days,
        "coverage": float(len(bins) / elapsed_cells),
        "elapsed_days": elapsed_cells * CELL_MINUTES / 1440.0,
        "n_readings": int(len(timestamps)),
        "n_valid_input_readings": int(n_valid),
        "n_duplicate_timestamps": int(n_valid - len(timestamps)),
        "n_observed_cells": int(len(bins)),
        "n_contiguous_runs": len(runs),
        "median_raw_interval_minutes": float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else np.nan,
        "n_interpolated_cells": 0,
    }
    for name, minutes in LAG_MINUTES.items():
        f[name], f[name + "_n_pairs"] = _lag_correlation(glucose, runs, int(minutes / CELL_MINUTES))
    f["peak_freq"], f["power_low_ratio"], f["n_spectral_windows"] = _spectrum(glucose, runs)

    velocities = [np.diff(glucose[r]) / CELL_MINUTES for r in runs if len(r) >= 2]
    accelerations = [np.diff(glucose[r], n=2) / CELL_MINUTES**2 for r in runs if len(r) >= 3]
    vel = np.concatenate(velocities) if velocities else np.array([])
    acc = np.concatenate(accelerations) if accelerations else np.array([])
    f.update(
        vel_abs_mean=float(np.mean(np.abs(vel))) if len(vel) else np.nan,
        vel_p95=float(np.quantile(np.abs(vel), .95)) if len(vel) else np.nan,
        vel_high_prop=float(np.mean(np.abs(vel) > VELOCITY_HIGH)) if len(vel) else np.nan,
        acc_abs_mean=float(np.mean(np.abs(acc))) if len(acc) else np.nan,
        n_velocity_pairs=int(len(vel)), n_acceleration_triples=int(len(acc)),
    )

    # A finite caller-supplied offset is required for local-time features.
    offset = np.nan if tz_offset_hours is None else float(tz_offset_hours)
    local_minutes = (bins.astype(float) + .5) * CELL_MINUTES + offset * 60
    hours = np.mod(local_minutes, 1440) / 60

    def local_mean(lo, hi):
        selected = (hours >= lo) & (hours < hi)
        return float(glucose[selected].mean()) if selected.any() else np.nan

    vals = [local_mean(6, 10), local_mean(11, 15), local_mean(18, 22), local_mean(0, 5)]
    f.update(zip(["mean_morning", "mean_midday", "mean_evening", "mean_night"], vals))
    f["diurnal_amp"] = float(max(vals) - min(vals)) if np.isfinite(vals).all() else np.nan
    f["morning_rise"] = local_mean(8, 9) - local_mean(4, 5)
    if np.isfinite(offset):
        days, inverse, counts = np.unique(np.floor(local_minutes / 1440).astype(np.int64),
                                          return_inverse=True, return_counts=True)
        daily_means = np.bincount(inverse, weights=glucose) / counts
        f["day_to_day_sd"] = float(np.std(daily_means, ddof=1)) if len(days) >= 2 else np.nan
        f["n_local_days"] = len(days)
    else:
        f["day_to_day_sd"], f["n_local_days"] = np.nan, 0

    peak_heights, recoveries = [], []
    n_peaks, n_search_cells, n_followed_peaks = 0, 0, 0
    for run in runs:
        x = glucose[run]
        n_search_cells += max(0, len(x) - 2)
        # Endpoints cannot be peaks: both flanks must be observed in this run.
        peaks, _ = signal.find_peaks(x, height=mean + sd, distance=6)
        n_peaks += len(peaks)
        peak_heights.extend((x[peaks] - mean).tolist())
        for i, peak in enumerate(peaks):
            next_peak = peaks[i + 1] if i + 1 < len(peaks) else len(x)
            # Recovery is observed at <=mean within 6 h, before another peak,
            # and before the run ends. Unobserved recoveries are censored.
            stop = min(peak + 73, next_peak, len(x))
            n_followed_peaks += int(stop > peak + 1)
            recovered = np.flatnonzero(x[peak + 1:stop] <= mean)
            if len(recovered):
                recoveries.append(float((recovered[0] + 1) * CELL_MINUTES))
    peak_height = float(np.mean(peak_heights)) if peak_heights else np.nan
    f.update(
        peaks_per_day=float(n_peaks / wear_days) if n_search_cells else np.nan,
        peak_prom_mean=peak_height,  # name kept from the shared extractor; this is the peak height, not prominence
        peak_height_above_mean=peak_height,
        recovery_mean=float(np.mean(recoveries)) if recoveries else np.nan,
        n_peaks=int(n_peaks), n_peak_search_cells=int(n_search_cells),
        n_followed_peaks=int(n_followed_peaks), n_observed_recoveries=len(recoveries),
    )
    return f
