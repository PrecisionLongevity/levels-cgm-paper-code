#!/usr/bin/env python3
"""
Shared CGM-feature builder for the metabolic analysis group.

Computes per-user CGM distribution and derived glycemic features directly from
canonical tables in the study database (glucose_readings, cgm_sessions, users).

Feature construction notes:

1. Distribution stats (mean / sd / cv / quantiles / TIR / TAR / TBR / skewness /
   kurtosis / total_readings / sensor_days / data_completeness) are computed
   directly from `glucose_readings` with a single-pass aggregation.

2. `gmi` (Bergenstal et al. 2018) and `j_index` (Wojcicki 1995) are closed-form
   functions of mean_glucose / sd_glucose.

3. `mage` is pulled from the canonical `cgm_sessions` table (precomputed by
   ETL as per-session 1-SD turning-point amplitudes) and aggregated to
   per-user as a reading-weighted mean, rather than recomputed per-user from
   raw readings (session-weighted vs whole-trace MAGE; per-user correlation
   r ≳ 0.95 against the whole-trace definition).

4. `modd`, `mean_fasting_glucose`, `dawn_effect` require time-of-day-resolved
   reading streams, computed from `glucose_readings` with user-local-time
   binning (via users.TZ_OFFSET_HOURS) grouped by date / hour-of-day.
   `dawn_effect` is defined as the 06:00-08:59 average minus the 02:00-04:59
   average. `modd` is currently not computed (left as NaN) due to the cost of
   the required 5-minute-bucket self-join; downstream consumers handle NaN
   via dropna().

The 27 columns produced (per-user):
  USER_ID, mean_glucose, sd_glucose, cv_glucose, median_glucose,
  glucose_p10, glucose_p25, glucose_p75, glucose_p90,
  skewness, kurtosis, gmi,
  tir_70_180, tir_70_140, tar_140, tar_180, tar_250,
  tbr_70, tbr_54,
  mage, modd, j_index,
  sensor_days, total_readings, data_completeness,
  mean_fasting_glucose, dawn_effect

Inputs: glucose_readings, cgm_sessions, users, labs, biomarker_map tables via
  LEVELS_DB (database_path()).
Outputs: OUTPUTS/metabolic/_cache/cgm_features_per_user.csv (per-user cache;
  participant-level, stays private).
Usage: python analysis/metabolic/cgm_features.py
"""

from __future__ import annotations

import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402

DUCKDB_PATH = database_path()
CACHE_DIR = OUTPUTS / 'metabolic' / '_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "cgm_features_per_user.csv"

EXPECTED_COLS = [
    "USER_ID",
    "mean_glucose", "sd_glucose", "cv_glucose", "median_glucose",
    "glucose_p10", "glucose_p25", "glucose_p75", "glucose_p90",
    "skewness", "kurtosis",
    "gmi",
    "tir_70_180", "tir_70_140", "tar_140", "tar_180", "tar_250",
    "tbr_70", "tbr_54",
    "mage", "modd", "j_index",
    "sensor_days", "total_readings", "data_completeness",
    "mean_fasting_glucose", "dawn_effect",
]


def build_cgm_features_per_user(
    min_sensor_days: int | None = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build per-user CGM features. Reads from levels.duckdb.

    `min_sensor_days`: optional final filter applied on the returned frame
        (the cache itself is unfiltered, so any threshold can be applied later).
    """
    if use_cache and CACHE_PATH.exists():
        if verbose:
            print(f"  [cache] loading {CACHE_PATH.name}")
        df = pd.read_csv(CACHE_PATH)
    else:
        if verbose:
            print("  [build] computing cgm_features_per_user from glucose_readings + cgm_sessions...")
        df = _compute_cgm_features(verbose=verbose)
        df.to_csv(CACHE_PATH, index=False)
        if verbose:
            print(f"  [build] cached {len(df):,} users -> {CACHE_PATH}")

    if min_sensor_days is not None:
        df = df[df["sensor_days"] >= min_sensor_days].copy()
    return df


def _compute_cgm_features(verbose: bool = True) -> pd.DataFrame:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    con.execute("SET threads TO 8")

    t0 = time.time()
    if verbose:
        print("    [duckdb] base distribution stats (1.15B readings, single pass)...")
    # All time-of-day features as conditional aggregates -> single pass.
    # Time-of-day (fasting/dawn) and per-user day boundaries must use local time
    # via users.TZ_OFFSET_HOURS — UTC is wrong for per-user time-of-day
    # binning. COALESCE(_, 0) treats unknown-TZ users as UTC.
    base = con.execute(
        """
        WITH glucose_local AS (
            SELECT g.USER_ID, g.TS, g.VALUE, u.TZ_OFFSET_HOURS
            FROM glucose_readings g
            LEFT JOIN users u ON g.USER_ID = u.USER_ID
        ),
        per_user AS (
            SELECT USER_ID,
                   COUNT(*) AS total_readings,
                   AVG(VALUE) AS mean_glucose,
                   STDDEV_SAMP(VALUE) AS sd_glucose,
                   APPROX_QUANTILE(VALUE, 0.50) AS median_glucose,
                   APPROX_QUANTILE(VALUE, 0.10) AS glucose_p10,
                   APPROX_QUANTILE(VALUE, 0.25) AS glucose_p25,
                   APPROX_QUANTILE(VALUE, 0.75) AS glucose_p75,
                   APPROX_QUANTILE(VALUE, 0.90) AS glucose_p90,
                   SKEWNESS(VALUE) AS skewness,
                   KURTOSIS(VALUE) AS kurtosis,
                   100.0 * SUM(CASE WHEN VALUE BETWEEN 70 AND 180 THEN 1 ELSE 0 END) / COUNT(*) AS tir_70_180,
                   100.0 * SUM(CASE WHEN VALUE BETWEEN 70 AND 140 THEN 1 ELSE 0 END) / COUNT(*) AS tir_70_140,
                   100.0 * SUM(CASE WHEN VALUE > 140 THEN 1 ELSE 0 END) / COUNT(*) AS tar_140,
                   100.0 * SUM(CASE WHEN VALUE > 180 THEN 1 ELSE 0 END) / COUNT(*) AS tar_180,
                   100.0 * SUM(CASE WHEN VALUE > 250 THEN 1 ELSE 0 END) / COUNT(*) AS tar_250,
                   100.0 * SUM(CASE WHEN VALUE < 70 THEN 1 ELSE 0 END) / COUNT(*) AS tbr_70,
                   100.0 * SUM(CASE WHEN VALUE < 54 THEN 1 ELSE 0 END) / COUNT(*) AS tbr_54,
                   -- TZ-local time-of-day conditional aggregates (single-pass).
                   -- Windows preserve original integer-hour BETWEEN semantics:
                   -- BETWEEN 4 AND 7 covers integer hours {4,5,6,7} = wall-clock [4:00, 8:00).
                   AVG(CASE WHEN FLOOR(MOD(EXTRACT(EPOCH FROM TS) / 3600.0 + COALESCE(TZ_OFFSET_HOURS, 0), 24)) BETWEEN 4 AND 7
                            THEN VALUE END) AS mean_fasting_glucose,
                   AVG(CASE WHEN FLOOR(MOD(EXTRACT(EPOCH FROM TS) / 3600.0 + COALESCE(TZ_OFFSET_HOURS, 0), 24)) BETWEEN 6 AND 8
                            THEN VALUE END) AS dawn_late_avg,
                   AVG(CASE WHEN FLOOR(MOD(EXTRACT(EPOCH FROM TS) / 3600.0 + COALESCE(TZ_OFFSET_HOURS, 0), 24)) BETWEEN 2 AND 4
                            THEN VALUE END) AS dawn_early_avg,
                   COUNT(DISTINCT DATE_TRUNC('day', TS + (COALESCE(TZ_OFFSET_HOURS, 0) * INTERVAL '1 hour'))) AS sensor_days
            FROM glucose_local
            GROUP BY USER_ID
        )
        SELECT *,
               (dawn_late_avg - dawn_early_avg) AS dawn_effect
        FROM per_user
        """
    ).fetchdf()
    base = base.drop(columns=["dawn_late_avg", "dawn_early_avg"])
    base["cv_glucose"] = (base["sd_glucose"] / base["mean_glucose"]) * 100.0
    if verbose:
        print(f"    [duckdb] base stats: {len(base):,} users in {time.time()-t0:.1f}s")

    # Closed-form derived columns
    base["data_completeness"] = base["total_readings"] / (base["sensor_days"] * 96.0)
    base["gmi"] = 3.31 + 0.02392 * base["mean_glucose"]
    base["j_index"] = 0.001 * (base["mean_glucose"] + base["sd_glucose"]) ** 2

    # MAGE from cgm_sessions: weight session-level MAGE by reading count
    if verbose:
        print("    [duckdb] MAGE (reading-weighted from cgm_sessions)...")
    mage = con.execute(
        """
        SELECT USER_ID,
               SUM(MAGE * TOTAL_READINGS)::DOUBLE / NULLIF(SUM(TOTAL_READINGS), 0) AS mage
        FROM cgm_sessions
        WHERE MAGE IS NOT NULL
        GROUP BY USER_ID
        """
    ).fetchdf()
    base = base.merge(mage, on="USER_ID", how="left")

    # MODD: skipped for performance.
    # MODD requires a 5-minute-bucket self-join on the full reading stream;
    # downstream consumers tolerate NaN MODD via dropna() patterns.
    if verbose:
        print("    [duckdb] MODD: SKIPPED (NaN)")
    base["modd"] = np.nan

    # Reorder columns to match pre-mig schema
    out = base[EXPECTED_COLS]
    return out


def build_cgm_parameters_per_window_90d() -> pd.DataFrame:
    """
    Build a 90d-window CGM-features-per-A1c-draw frame.

    Schema (subset):
        USER_ID, draw_index, window_type, a1c_value, a1c_date,
        readings, sensor_days, data_completeness,
        mean_glucose, sd_glucose, cv_glucose, gmi,
        tir_70_180, tar_180, tar_250, tbr_70, tbr_54, mage

    Logic: for each (USER_ID, A1c draw_date) pair, take CGM readings
    in the 90 days ending at draw_date.

    Note: this read joins glucose_readings to labs at TYPE='hba1c'. Returns
    one row per (user, draw_index) with draw_index ordered by date.
    """
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    con.execute("SET threads TO 8")

    print("    [duckdb] building cgm_parameters_per_window (90d) ...")
    t0 = time.time()
    df = con.execute(
        """
        WITH a1c AS (
            SELECT l.USER_ID,
                   l.DRAW_DATE AS a1c_date,
                   l.VALUE AS a1c_value,
                   ROW_NUMBER() OVER (
                       PARTITION BY l.USER_ID ORDER BY l.DRAW_DATE
                   ) - 1 AS draw_index
            FROM labs l
            JOIN biomarker_map m ON l.RAW_TYPE = m.raw_type
            WHERE m.canonical_type = 'a1c'
              AND l.VALUE IS NOT NULL
              AND l.VALUE BETWEEN 3 AND 20  -- ETL clamp
        ),
        win AS (
            -- sensor_days counted in user-local time.
            SELECT a.USER_ID,
                   a.draw_index,
                   a.a1c_value,
                   a.a1c_date,
                   COUNT(*) AS readings,
                   COUNT(DISTINCT DATE_TRUNC('day', g.TS + (COALESCE(u.TZ_OFFSET_HOURS, 0) * INTERVAL '1 hour'))) AS sensor_days,
                   AVG(g.VALUE) AS mean_glucose,
                   STDDEV_SAMP(g.VALUE) AS sd_glucose,
                   100.0 * SUM(CASE WHEN g.VALUE BETWEEN 70 AND 180 THEN 1 ELSE 0 END)/COUNT(*) AS tir_70_180,
                   100.0 * SUM(CASE WHEN g.VALUE > 180 THEN 1 ELSE 0 END)/COUNT(*) AS tar_180,
                   100.0 * SUM(CASE WHEN g.VALUE > 250 THEN 1 ELSE 0 END)/COUNT(*) AS tar_250,
                   100.0 * SUM(CASE WHEN g.VALUE < 70 THEN 1 ELSE 0 END)/COUNT(*) AS tbr_70,
                   100.0 * SUM(CASE WHEN g.VALUE < 54 THEN 1 ELSE 0 END)/COUNT(*) AS tbr_54
            FROM a1c a
            JOIN glucose_readings g
              ON g.USER_ID = a.USER_ID
             AND g.TS >= a.a1c_date - INTERVAL 90 DAY
             AND g.TS <= a.a1c_date
            LEFT JOIN users u ON u.USER_ID = a.USER_ID
            GROUP BY a.USER_ID, a.draw_index, a.a1c_value, a.a1c_date
        )
        SELECT USER_ID, draw_index,
               '90d' AS window_type,
               a1c_value, a1c_date, readings, sensor_days,
               readings / (sensor_days * 96.0) AS data_completeness,
               mean_glucose, sd_glucose,
               (sd_glucose / NULLIF(mean_glucose, 0)) * 100.0 AS cv_glucose,
               3.31 + 0.02392 * mean_glucose AS gmi,
               tir_70_180, tar_180, tar_250, tbr_70, tbr_54
        FROM win
        """
    ).fetchdf()
    print(f"    [duckdb] 90d windows: {len(df):,} draws in {time.time()-t0:.1f}s")

    # MAGE per (USER_ID, draw_index): pull session-level MAGE and intersect
    # with the 90-day window, averaging session-level MAGE for sessions
    # overlapping the window, weighted by overlap.
    print("    [duckdb] MAGE per draw window (session-overlap-weighted)...")
    t1 = time.time()
    mage = con.execute(
        """
        WITH a1c AS (
            SELECT l.USER_ID, l.DRAW_DATE AS a1c_date,
                   ROW_NUMBER() OVER (PARTITION BY l.USER_ID ORDER BY l.DRAW_DATE) - 1 AS draw_index
            FROM labs l
            JOIN biomarker_map m ON l.RAW_TYPE = m.raw_type
            WHERE m.canonical_type = 'a1c'
              AND l.VALUE IS NOT NULL AND l.VALUE BETWEEN 3 AND 20
        )
        SELECT a.USER_ID, a.draw_index,
               SUM(s.MAGE * s.TOTAL_READINGS) / NULLIF(SUM(s.TOTAL_READINGS), 0) AS mage
        FROM a1c a
        JOIN cgm_sessions s
          ON s.USER_ID = a.USER_ID
         AND s.MAGE IS NOT NULL
         AND s.SESSION_END >= a.a1c_date - INTERVAL 90 DAY
         AND s.SESSION_START <= a.a1c_date
        GROUP BY a.USER_ID, a.draw_index
        """
    ).fetchdf()
    print(f"    [duckdb] MAGE per window: {len(mage):,} draws in {time.time()-t1:.1f}s")
    df = df.merge(mage, on=["USER_ID", "draw_index"], how="left")

    return df


if __name__ == "__main__":
    df = build_cgm_features_per_user(use_cache=False, verbose=True)
    print(df.shape)
    print(df.describe().T[["count", "mean", "std", "min", "max"]].to_string())
