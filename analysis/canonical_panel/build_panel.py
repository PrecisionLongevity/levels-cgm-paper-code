#!/usr/bin/env python3
"""Builds the canonical laboratory-CGM panel linking each participant's eligible HbA1c
anchor to CGM window metrics, other laboratory analytes, and anthropometrics.

Laboratory units are recognized explicitly and converted before individual-row
validity limits. Same-day values use medians; conflicting HbA1c days (normalized
raw spread >0.30001 percentage points) cannot be anchors. HbA1c uses half-up tenths
with float32 tolerance before thresholds. The first chronological accepted draw
with >=14 dense CGM days is selected anew after cleaning. Ever-diabetes flags use
all accepted raw observations, including those on excluded conflict days.

The primary window is +/-90 participant-local calendar days. The preceding
sensitivity selects its own first qualifying anchor over days -89..0. Dense days
require >=230 raw readings and >=77 observed 15-minute cells. Accounts must have
IS_INTERNAL_USER=false and a nonmissing timezone. Other laboratory values are
nearest within +/-7 days, ties earlier; HOMA/TyG/TG:HDL use same-day input pairs.
BMI is nearest within +/-90 days. CGM window-metric definitions live in
_window_features.py; raw rich features use its versioned shared gap-aware extractor.

Inputs: database tables `labs`, `users`, `weight`, `glucose_readings`, `day_panel`
(via LEVELS_DB); analysis/canonical_panel/_window_features.py and
cgmpaper/lab_normalization.py for shared definitions.

Outputs (OUTPUTS/canonical_panel/, participant-level and private):
  canonical_lab_cgm_panel.csv / canonical_lab_cgm_panel_preceding90.csv, each with a
  CSV.metadata.json sidecar; canonical_panel_summary.md and build_manifest.json
  (aggregate summaries only).

Usage:
  python analysis/canonical_panel/build_panel.py [--window symmetric|preceding]
    [--force] [--chunk-users N] [--threads N] [--memory-limit SIZE]
    [--database PATH] [--output-dir DIR] [--grid-cache PATH] [--require-grid-cache]
    [--skip-rich-features] [--skip-reading-mean]

Example, restricted-cache build without a raw-CGM scan:
  python analysis/canonical_panel/build_panel.py \
    --grid-cache /restricted/dense_day_grid.parquet --require-grid-cache \
    --skip-rich-features --skip-reading-mean

The skipped stages are explicit in the sidecar; this is not a complete rich-feature
panel. Omit the skip options to compute them after validating the shared extractor.
--force rebuilds caches.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402

if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "mplcache"
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _window_features as wf  # noqa: E402
from cgmpaper.lab_normalization import (  # noqa: E402
    BOUNDS as ANALYTE_RANGES, CANONICAL_UNITS, LAB_DEFINITION_VERSION,
    normalize_labs, round_a1c,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = database_path()
CANONICAL_PANEL_VERSION = "publication"
OUT = OUTPUTS / 'canonical_panel'
CACHE = OUT / "_cache"

SUMMARY_PATH = OUT / "canonical_panel_summary.md"
MANIFEST_PATH = OUT / "build_manifest.json"

CGM_WINDOW_DAYS = 90       # +/- window around the anchor draw
MIN_DENSE_DAYS = 14        # dense days required in the window
TIGHT_WINDOW_DAYS = 30     # w30_* tightness window
LAB_BAND_DAYS = 7          # other analytes: nearest draw within +/-7 d
BMI_BAND_DAYS = 90         # weight: nearest within +/-90 d
WEIGHT_RANGE = (80.0, 500.0)   # plausible body weight in pounds
HEIGHT_RANGE = (48.0, 90.0)    # plausible height clamp on users.HEIGHT_INCHES
BMI_RANGE = (15.0, 60.0)       # plausible BMI range

WINDOWS = {
    # name: (days before anchor, days after anchor, tight before, tight after, csv name)
    "symmetric": (CGM_WINDOW_DAYS, CGM_WINDOW_DAYS, TIGHT_WINDOW_DAYS, TIGHT_WINDOW_DAYS,
                  "canonical_lab_cgm_panel.csv"),
    "preceding": (CGM_WINDOW_DAYS - 1, 0, TIGHT_WINDOW_DAYS - 1, 0,
                  "canonical_lab_cgm_panel_preceding90.csv"),
}

ANALYTES = [
    "fasting_glucose", "fasting_insulin", "triglycerides", "hdl_cholesterol",
    "ldl_cholesterol", "total_cholesterol", "apo_b", "hscrp", "uric_acid", "alt", "ast",
]
DERIVED = ["homa_ir", "tyg", "tg_hdl_ratio"]
# Laboratory validity limits and unit transforms are centralized in lab_normalization.
DERIVED_RANGES = {
    "homa_ir": wf.DERIVED_SPECS["HOMA_IR"],
    "tyg": wf.DERIVED_SPECS["TYG"],
    "tg_hdl_ratio": wf.DERIVED_SPECS["TG_HDL_RATIO"],
}

from analysis.canonical_panel import _canonical_grid as accepted_grid

CELL_COLUMNS = [f"c{index:02d}" for index in range(wf.CELLS_PER_DAY)]
EPOCH = pd.Timestamp("1970-01-01")


def log(message: str) -> None:
    print(message, flush=True)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def configure_output(path: Path) -> None:
    global OUT, CACHE, SUMMARY_PATH, MANIFEST_PATH
    OUT = path.resolve()
    CACHE = OUT / "_cache"
    SUMMARY_PATH = OUT / "canonical_panel_summary.md"
    MANIFEST_PATH = OUT / "build_manifest.json"
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)


def feature_metadata() -> dict:
    trace_version = getattr(wf, "RICH_FEATURE_VERSION", "unversioned")
    return {
        "canonical_panel_version": CANONICAL_PANEL_VERSION,
        "dense_day_count_version": accepted_grid.DENSE_DAY_VERSION,
        "daily_mage_version": accepted_grid.MAGE_VERSION,
        "lab_definition_version": LAB_DEFINITION_VERSION,
        "trace_feature_version": trace_version,
        "core_grid_feature_version": getattr(wf, "CORE_GRID_FEATURE_VERSION", "unversioned"),
    }


def assert_unique(df: pd.DataFrame, columns: list[str], name: str) -> None:
    duplicates = int(df.duplicated(columns).sum())
    if duplicates:
        raise AssertionError(f"{name}: {duplicates} duplicate keys for {columns}")


def day_number(values: pd.Series) -> pd.Series:
    """Calendar date -> integer days since epoch (Timedelta arithmetic; pandas 3 safe)."""
    return ((pd.to_datetime(values) - EPOCH) / pd.Timedelta(days=1)).round().astype("int64")


def git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 — git is informational only
        return None


# ----------------------------------------------------------------------------
# Step 1: HbA1c draws and the selection funnel up to the dense-day gate
# ----------------------------------------------------------------------------
def prepare_laboratories(con: duckdb.DuckDBPyConnection) -> dict:
    """One unit-aware construction supplies anchors, endpoints, and ever-high flags."""
    names = list(ANALYTE_RANGES)
    placeholders = ",".join("?" for _ in names)
    raw = con.execute(f"""
        SELECT USER_ID, TYPE, DRAW_DATE, VALUE, UNIT
        FROM labs WHERE TYPE IN ({placeholders})
    """, names).fetchdf()
    daily, flags, audit = normalize_labs(raw)
    for name, frame in (("normalized_daily_labs", daily), ("normalized_ever_flags", flags)):
        con.register("_lab_frame", frame)
        con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS SELECT * FROM _lab_frame")
        con.unregister("_lab_frame")
    return audit


def ensure_laboratories(con: duckdb.DuckDBPyConnection) -> None:
    names = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if not {"normalized_daily_labs", "normalized_ever_flags"} <= names:
        prepare_laboratories(con)


def build_a1c_draws(con: duckdb.DuckDBPyConnection) -> dict:
    """Rebuild candidate dates after unit, row-validity, and conflict exclusions."""
    audit = prepare_laboratories(con)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE a1c_draws AS
        WITH per_date AS (
            SELECT USER_ID, DRAW_DATE, value AS a1c
            FROM normalized_daily_labs WHERE TYPE = 'a1c'
        )
        SELECT USER_ID, DRAW_DATE, a1c,
               ROW_NUMBER() OVER (PARTITION BY USER_ID ORDER BY DRAW_DATE) AS draw_rn,
               COUNT(*)    OVER (PARTITION BY USER_ID)                     AS n_draws
        FROM per_date
    """)
    return audit


def sensor_day_gate(con: duckdb.DuckDBPyConnection, before: int, after: int) -> pd.DataFrame:
    """Users with >=14 observed days around an accepted draw, before account/TZ gates.

    This is a descriptive funnel stage, not an eligibility gate.
    """
    return con.execute(f"""
        SELECT a.USER_ID, MIN(u.IS_INTERNAL_USER) AS is_internal,
               MIN(u.TZ_OFFSET_HOURS) AS tz_offset_hours
        FROM a1c_draws a
        JOIN day_panel d
          ON d.USER_ID = a.USER_ID
         AND d.MEAN_GLUCOSE IS NOT NULL
         AND d.DAY >= a.DRAW_DATE - INTERVAL '{before} day'
         AND d.DAY <= a.DRAW_DATE + INTERVAL '{after} day'
        LEFT JOIN users u ON u.USER_ID = a.USER_ID
        GROUP BY a.USER_ID, a.DRAW_DATE
        HAVING COUNT(*) >= {MIN_DENSE_DAYS}
    """).fetchdf().drop_duplicates("USER_ID")


# ----------------------------------------------------------------------------
# Step 2: dense-day grid cache (shared by both windows and w30_*)
# ----------------------------------------------------------------------------
def validate_cached_grid(grid: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Validate cached observed cells against every newly accepted candidate day.

    Superset caches are allowed, but missing days, changed account/timezone/count
    metadata, duplicate keys, and inconsistent cell counts fail closed. This is
    not a reconstruction of every cached cell from the raw glucose table.
    """
    required = {"USER_ID", "DAY", "tz_offset_hours", "n_raw_readings", "daily_mage", "n_cells", *CELL_COLUMNS}
    if not required <= set(grid.columns):
        raise ValueError(f"Grid cache missing columns: {sorted(required - set(grid.columns))}")
    grid = grid.copy()
    grid["DAY"] = pd.to_datetime(grid["DAY"]).dt.normalize()
    assert_unique(grid, ["USER_ID", "DAY"], "dense-day grid cache")
    expected = candidates.rename(columns={"tz": "expected_tz", "N_READINGS": "expected_readings"}).copy()
    expected["DAY"] = pd.to_datetime(expected["DAY"]).dt.normalize()
    selected = expected.merge(grid, on=["USER_ID", "DAY"], how="left", validate="one_to_one", indicator=True)
    missing = int(selected["_merge"].ne("both").sum())
    if missing:
        raise ValueError(f"Grid cache lacks {missing} accepted candidate days; rebuild instead of fixing old anchors")
    if not (selected["tz_offset_hours"].eq(selected["expected_tz"]).all()
            and selected["n_raw_readings"].eq(selected["expected_readings"]).all()):
        raise ValueError("Grid cache candidate timezone or reading counts differ from the database")
    cells = selected[CELL_COLUMNS].to_numpy(float)
    if np.isinf(cells).any() or not np.array_equal(np.isfinite(cells).sum(axis=1), selected["n_cells"].to_numpy()):
        raise ValueError("Grid cache observed-cell counts are inconsistent")
    finite = cells[np.isfinite(cells)]
    if len(finite) and (finite.min() < 20.0 or finite.max() > 500.0):
        raise ValueError("Grid cache contains cells outside the accepted glucose range")
    return selected[grid.columns].sort_values(["USER_ID", "DAY"]).reset_index(drop=True)


def build_dense_day_grid(con: duckdb.DuckDBPyConnection, force: bool, chunk_users: int,
                         grid_cache: Path | None = None, require_cache: bool = False) -> pd.DataFrame:
    """Observed 15-minute grids on accepted raw local days near any valid draw.

    At least 230 readings in 20–500 mg/dL; the database's day_panel table is not used.
    The separate 77-cell gate is applied before anchor selection. The daily
    turning-amplitude proxy is recomputed from accepted, gap-bounded raw runs.
    """
    path = CACHE / "dense_day_grid.parquet"
    candidate_path=CACHE/'accepted_candidate_days.parquet'
    if candidate_path.exists() and not force:
        candidates=pd.read_parquet(candidate_path)
    else:
        log('grid: counting accepted raw readings on actual local dates')
        candidates=accepted_grid.candidate_days(con)
    candidates['DAY']=pd.to_datetime(candidates['DAY'])
    candidates.to_parquet(candidate_path,index=False)
    con.register('_accepted_candidates',candidates)
    con.execute('CREATE OR REPLACE TEMP TABLE candidate_days AS SELECT * FROM _accepted_candidates')
    con.unregister('_accepted_candidates')
    source_cache = grid_cache or path
    if source_cache.exists() and not force:
        log(f"grid: validating cache {display_path(source_cache)} against the candidate days")
        try:
            grid = validate_cached_grid(pd.read_parquet(source_cache), candidates)
        except (AssertionError, ValueError, FileNotFoundError) as exc:
            if grid_cache is not None or require_cache:
                raise ValueError(f"Explicit grid cache validation failed: {exc}") from exc
            log(f"grid: cache rejected ({exc}); rebuilding from the database")
        else:
            grid.attrs["source_cache"] = {"path": display_path(source_cache),
                                         "candidate_days": int(len(candidates)), "candidate_coverage_verified": True,
                                         "raw_cell_values_fully_rebuilt": False}
            return grid
    elif grid_cache is not None and not source_cache.exists():
        raise FileNotFoundError(f"Grid cache not found: {source_cache}")
    if require_cache:
        raise ValueError("A verified existing grid cache is required; refusing a raw CGM scan")
    user_ids = np.sort(candidates["USER_ID"].unique())
    log(f"grid: {len(candidates):,} candidate days for {len(user_ids):,} users; "
        f"aggregating 15-min cells in chunks of {chunk_users}")

    frames: list[pd.DataFrame] = []
    started = time.time()
    for start in range(0, len(user_ids), chunk_users):
        chunk = user_ids[start:start + chunk_users]
        cells,amplitude=accepted_grid.grid_batch(con,chunk[0],chunk[-1])
        amplitude['DAY']=pd.to_datetime(amplitude['DAY'])
        cells["DAY"] = pd.to_datetime(cells["DAY"])
        keys = cells[["USER_ID", "DAY"]].drop_duplicates().reset_index(drop=True)
        key_index = pd.MultiIndex.from_frame(keys)
        row = key_index.get_indexer(pd.MultiIndex.from_frame(cells[["USER_ID", "DAY"]]))
        matrix = np.full((len(keys), wf.CELLS_PER_DAY), np.nan)
        matrix[row, cells["cell"].to_numpy(int)] = cells["glucose"].to_numpy(float)
        wide = pd.DataFrame(matrix, columns=CELL_COLUMNS)
        wide.insert(0, "n_cells", np.isfinite(matrix).sum(axis=1).astype("int64"))
        frames.append(pd.concat([keys, wide], axis=1).merge(amplitude,on=['USER_ID','DAY'],validate='one_to_one'))
        log(f"grid: users {min(start + chunk_users, len(user_ids)):,}/{len(user_ids):,}; "
            f"elapsed={time.time() - started:.0f}s")

    if not frames:
        raise ValueError("No accepted HbA1c candidate days remain after account/timezone/coverage gates")
    grid = pd.concat(frames, ignore_index=True)
    grid = candidates.rename(columns={
        "tz": "tz_offset_hours", "N_READINGS": "n_raw_readings",
    }).merge(grid, on=["USER_ID", "DAY"], how="left")
    grid["n_cells"] = grid["n_cells"].fillna(0).astype("int64")
    assert_unique(grid, ["USER_ID", "DAY"], "dense-day grid")
    grid = grid.sort_values(["USER_ID", "DAY"]).reset_index(drop=True)
    grid.to_parquet(path, index=False)
    log(f"grid: cached {len(grid):,} days ({int((grid['n_cells'] >= wf.MIN_15M_CELLS).sum()):,} "
        f"dense) -> {display_path(path)} in {time.time() - started:.0f}s")
    grid.attrs["source_cache"] = {"path": display_path(path),
                                 "candidate_days": int(len(candidates)), "candidate_coverage_verified": True,
                                 "raw_cell_values_fully_rebuilt": True}
    return grid


# ----------------------------------------------------------------------------
# Step 3: anchor selection with the dense-day gate
# ----------------------------------------------------------------------------
def select_anchors(con: duckdb.DuckDBPyConnection, dense: pd.DataFrame, before: int, after: int) -> pd.DataFrame:
    """First HbA1c draw per user with >= 14 dense days in [draw - before, draw + after]."""
    assert_unique(dense, ["USER_ID", "DAY"], "dense-day anchor input")
    con.register("dense_days", dense[["USER_ID", "DAY"]])
    anchors = con.execute(f"""
        WITH counted AS (
            SELECT a.USER_ID, a.DRAW_DATE, a.a1c, a.draw_rn, a.n_draws, COUNT(dd.DAY) AS n_dense
            FROM a1c_draws a
            JOIN dense_days dd
              ON dd.USER_ID = a.USER_ID
             AND dd.DAY >= a.DRAW_DATE - INTERVAL '{before} day'
             AND dd.DAY <= a.DRAW_DATE + INTERVAL '{after} day'
            GROUP BY a.USER_ID, a.DRAW_DATE, a.a1c, a.draw_rn, a.n_draws
            HAVING COUNT(dd.DAY) >= {MIN_DENSE_DAYS}
        ),
        picked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY USER_ID ORDER BY draw_rn) AS pick_rn
            FROM counted
        )
        SELECT p.USER_ID, p.DRAW_DATE AS anchor_date, p.a1c, p.draw_rn AS a1c_draw_rn,
               p.n_draws AS n_a1c_draws, u.TZ_OFFSET_HOURS AS tz_offset_hours,
               u.SEX AS sex, u.AGE_BUCKET AS age_bucket, u.TZ_SOURCE AS tz_source
        FROM picked p JOIN users u ON u.USER_ID = p.USER_ID
        WHERE p.pick_rn = 1 AND u.IS_INTERNAL_USER = FALSE AND u.TZ_OFFSET_HOURS IS NOT NULL
        ORDER BY p.USER_ID
    """).fetchdf()
    con.unregister("dense_days")
    anchors["anchor_date"] = pd.to_datetime(anchors["anchor_date"])
    anchors["anchor_year"] = anchors["anchor_date"].dt.year.astype("int64")
    anchors["a1c"] = round_a1c(anchors["a1c"])
    return anchors


# ----------------------------------------------------------------------------
# Step 4: window CGM metrics from the grid
# ----------------------------------------------------------------------------
def compute_window_metrics(anchors: pd.DataFrame, dense: pd.DataFrame, before: int, after: int,
                           tight_before: int, tight_after: int) -> pd.DataFrame:
    dense = dense.merge(anchors[["USER_ID", "anchor_date"]], on="USER_ID", how="inner")
    dense["offset"] = day_number(dense["DAY"]) - day_number(dense["anchor_date"])
    dense = dense[(dense["offset"] >= -before) & (dense["offset"] <= after)]
    dense = dense.sort_values(["USER_ID", "DAY"])
    dense["day_no"] = day_number(dense["DAY"])

    records: list[dict] = []
    started = time.time()
    groups = dense.groupby("USER_ID", sort=False)
    for index, (user_id, part) in enumerate(groups, start=1):
        grid = part[CELL_COLUMNS].to_numpy(float)
        days = part["day_no"].to_numpy()
        record = wf.window_metrics(days, grid, part["daily_mage"].to_numpy(float))
        record.update({
            "USER_ID": user_id,
            "n_readings": int(part["n_raw_readings"].sum()),
            "first_dense_day": part["DAY"].iloc[0].date().isoformat(),
            "last_dense_day": part["DAY"].iloc[-1].date().isoformat(),
        })
        tight = (part["offset"] >= -tight_before) & (part["offset"] <= tight_after)
        if int(tight.sum()) >= MIN_DENSE_DAYS:
            sub = part[tight]
            tight_record = wf.window_metrics(
                sub["day_no"].to_numpy(), sub[CELL_COLUMNS].to_numpy(float),
                sub["daily_mage"].to_numpy(float),
            )
            for column in wf.W30_COLUMNS:
                record[f"w30_{column}"] = tight_record[column]
        else:
            record["w30_n_dense_days"] = int(tight.sum())
            for column in wf.W30_COLUMNS[1:]:
                record[f"w30_{column}"] = np.nan
        records.append(record)
        if index % 1000 == 0 or index == groups.ngroups:
            log(f"window metrics: {index:,}/{groups.ngroups:,} users; elapsed={time.time() - started:.0f}s")
    metrics = pd.DataFrame(records)
    assert (metrics["n_dense_days"] >= MIN_DENSE_DAYS).all()
    return metrics


# ----------------------------------------------------------------------------
# Step 5: shared observed-run trace features on raw readings of dense days
# ----------------------------------------------------------------------------
def compute_rich_features(con: duckdb.DuckDBPyConnection, anchors: pd.DataFrame, dense: pd.DataFrame,
                          before: int, after: int, window: str, force: bool, chunk_users: int) -> pd.DataFrame:
    path = CACHE / f"rich_features_{window}.parquet"
    metadata_path = path.with_suffix(".metadata.json")
    days = dense.merge(anchors[["USER_ID", "anchor_date"]], on="USER_ID", how="inner")
    offset = day_number(days["DAY"]) - day_number(days["anchor_date"])
    days = days.loc[(offset >= -before) & (offset <= after)]
    if path.exists() and not force:
        cached = pd.read_parquet(path)
        assert_unique(cached, ["USER_ID"], "cached rich features")
        if set(cached["USER_ID"]) <= set(anchors["USER_ID"]):
            log(f"rich features: loading cache {display_path(path)}")
            return cached
        log("rich features: cache participants differ from current anchors; rebuilding")
    days = days[["USER_ID", "DAY", "tz_offset_hours"]]
    days = days.rename(columns={"tz_offset_hours": "tz"}).sort_values(["USER_ID", "DAY"])
    user_ids = np.sort(days["USER_ID"].unique())
    con.register("window_days", days)

    records: list[dict] = []
    started = time.time()
    for start in range(0, len(user_ids), chunk_users):
        chunk = user_ids[start:start + chunk_users]
        raw = con.execute("""
            SELECT g.USER_ID, g.TS, g.VALUE, c.tz
            FROM glucose_readings g
            JOIN window_days c
              ON g.USER_ID = c.USER_ID
             AND g.TS >= CAST(c.DAY AS TIMESTAMP) - c.tz * INTERVAL '1 hour'
             AND g.TS <  CAST(c.DAY AS TIMESTAMP) + INTERVAL '1 day' - c.tz * INTERVAL '1 hour'
            WHERE g.USER_ID >= ? AND g.USER_ID <= ?
              AND g.VALUE BETWEEN 20 AND 500
        """, [str(chunk[0]), str(chunk[-1])]).fetchdf()
        raw["t_min"] = (pd.to_datetime(raw["TS"]) - EPOCH) / pd.Timedelta(minutes=1)
        for user_id, part in raw.groupby("USER_ID", sort=False):
            feats = wf.rich_features(
                part["t_min"].to_numpy(), part["VALUE"].to_numpy(), float(part["tz"].iloc[0])
            )
            if feats is None:
                continue
            feats["USER_ID"] = user_id
            records.append(feats)
        log(f"rich features: users {min(start + chunk_users, len(user_ids)):,}/{len(user_ids):,}; "
            f"elapsed={time.time() - started:.0f}s")
    con.unregister("window_days")
    rich = pd.DataFrame(records) if records else pd.DataFrame({"USER_ID": pd.Series(dtype=anchors["USER_ID"].dtype)})
    assert_unique(rich, ["USER_ID"], "rich features")
    rich.to_parquet(path, index=False)
    metadata_path.write_text(json.dumps(feature_metadata(), indent=2) + "\n")
    return rich


# ----------------------------------------------------------------------------
# Step 6: reading-level window mean (all-days comparison column)
# ----------------------------------------------------------------------------
def compute_reading_level_mean(con: duckdb.DuckDBPyConnection, anchors: pd.DataFrame,
                               before: int, after: int) -> pd.DataFrame:
    """AVG of every raw reading whose participant-local calendar day lies in the window
    (all days, not only dense days), accepting only 20–500 mg/dL."""
    con.register("anchor_rows", anchors[["USER_ID", "anchor_date", "tz_offset_hours"]])
    df = con.execute(f"""
        SELECT a.USER_ID, AVG(g.VALUE) AS mean_glucose_readings, COUNT(*) AS n_readings_all_days
        FROM anchor_rows a
        JOIN glucose_readings g
          ON g.USER_ID = a.USER_ID
         AND g.VALUE BETWEEN 20 AND 500
         AND g.TS >= CAST(a.anchor_date AS TIMESTAMP) - INTERVAL '{before} day'
                     - a.tz_offset_hours * INTERVAL '1 hour'
         AND g.TS <  CAST(a.anchor_date AS TIMESTAMP) + INTERVAL '{after + 1} day'
                     - a.tz_offset_hours * INTERVAL '1 hour'
        GROUP BY a.USER_ID
    """).fetchdf()
    con.unregister("anchor_rows")
    return df


# ----------------------------------------------------------------------------
# Step 7: labs (nearest within +/-7 d), derived indices, cohort flags
# ----------------------------------------------------------------------------
def lab_wide(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Unit-aware daily analytes and same-day derived indices, before band selection."""
    ensure_laboratories(con)
    labs = con.execute("""
        SELECT USER_ID, TYPE, DRAW_DATE, value AS VALUE
        FROM normalized_daily_labs WHERE TYPE <> 'a1c'
    """).fetchdf()
    labs["DRAW_DATE"] = pd.to_datetime(labs["DRAW_DATE"])
    if labs.empty:
        wide = labs[["USER_ID", "DRAW_DATE"]].copy()
    else:
        wide = labs.pivot(index=["USER_ID", "DRAW_DATE"], columns="TYPE", values="VALUE").reset_index()
    wide.columns.name = None
    for analyte in ANALYTES:
        if analyte not in wide:
            wide[analyte] = np.nan
    wide["homa_ir"] = wide["fasting_glucose"] * wide["fasting_insulin"] / 405.0
    wide["tyg"] = np.log(wide["triglycerides"] * wide["fasting_glucose"] / 2.0)
    wide["tg_hdl_ratio"] = wide["triglycerides"] / wide["hdl_cholesterol"]
    for name, (low, high) in DERIVED_RANGES.items():
        wide.loc[~wide[name].between(low, high), name] = np.nan
    return wide


def nearest_within_band(anchors: pd.DataFrame, table: pd.DataFrame, date_column: str,
                        columns: list[str], band: int, offset_suffix: str = "_offset_days") -> pd.DataFrame:
    """Per column, the non-null row nearest the anchor within +/-band days (ties -> earlier)."""
    if band < 0:
        raise ValueError("Laboratory date band must be nonnegative")
    assert_unique(anchors, ["USER_ID"], "anchors for nearest laboratory")
    table = table.copy()
    table[date_column] = pd.to_datetime(table[date_column]).dt.normalize()
    assert_unique(table, ["USER_ID", date_column], "daily laboratory/anthropometric input")
    merged = anchors[["USER_ID", "anchor_date"]].merge(table, on="USER_ID", how="inner", validate="one_to_many")
    merged["offset_days"] = day_number(merged[date_column]) - day_number(merged["anchor_date"])
    merged = merged[merged["offset_days"].abs() <= band]
    merged = merged.assign(abs_offset=merged["offset_days"].abs()).sort_values(
        ["USER_ID", "abs_offset", date_column], kind="stable"
    )
    out = anchors[["USER_ID"]].copy()
    for column in columns:
        pick = merged.loc[merged[column].notna()].drop_duplicates("USER_ID")
        pick = pick[["USER_ID", column, date_column, "offset_days"]].rename(columns={
            date_column: f"{column}_date", "offset_days": f"{column}{offset_suffix}",
        })
        pick[f"{column}_date"] = pick[f"{column}_date"].dt.date.astype(str)
        out = out.merge(pick, on="USER_ID", how="left")
    return out


def cohort_flags(con: duckdb.DuckDBPyConnection, panel: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Classify from accepted raw records, including observations on conflict days."""
    ensure_laboratories(con)
    ever = con.execute("SELECT USER_ID, diabetes_range_ever FROM normalized_ever_flags").fetchdf()
    flags = panel[["USER_ID", "a1c", "fasting_glucose"]].merge(ever, on="USER_ID", how="left", validate="one_to_one")
    if flags["diabetes_range_ever"].isna().any():
        raise ValueError("A selected anchor has no normalized laboratory eligibility record")
    flags["a1c"] = round_a1c(flags["a1c"])
    flags["diabetes_range_ever"] = flags["diabetes_range_ever"].astype(bool)
    flags["no_diabetes_range"] = (
        ~flags["diabetes_range_ever"] & (flags["a1c"] < 6.5)
        & (flags["fasting_glucose"].isna() | (flags["fasting_glucose"] < 126.0))
    )
    flags["a1c_normal_strict"] = flags["no_diabetes_range"] & (flags["a1c"] >= 3.0) & (flags["a1c"] < 5.7)
    return flags[["USER_ID", "diabetes_range_ever", "no_diabetes_range", "a1c_normal_strict"]]


def anthropometrics(con: duckdb.DuckDBPyConnection, anchors: pd.DataFrame) -> pd.DataFrame:
    weights = con.execute(f"""
        SELECT w.USER_ID, w.DAY, MEDIAN(w.WEIGHT) AS weight_lb, ANY_VALUE(u.HEIGHT_INCHES) AS height_in
        FROM weight w JOIN users u ON u.USER_ID = w.USER_ID
        WHERE w.WEIGHT BETWEEN {WEIGHT_RANGE[0]} AND {WEIGHT_RANGE[1]}
        GROUP BY w.USER_ID, w.DAY
    """).fetchdf()
    weights["DAY"] = pd.to_datetime(weights["DAY"])
    picked = nearest_within_band(anchors, weights[["USER_ID", "DAY", "weight_lb"]], "DAY",
                                 ["weight_lb"], BMI_BAND_DAYS)
    picked = picked.merge(weights[["USER_ID", "height_in"]].drop_duplicates("USER_ID"),
                          on="USER_ID", how="left")
    height_ok = picked["height_in"].between(*HEIGHT_RANGE)
    picked["bmi"] = np.where(height_ok, picked["weight_lb"] * 703.0 / np.square(picked["height_in"]), np.nan)
    picked.loc[~picked["bmi"].between(*BMI_RANGE), "bmi"] = np.nan
    picked["bmi_offset_days"] = picked["weight_lb_offset_days"]
    return picked[["USER_ID", "bmi", "bmi_offset_days", "weight_lb"]]


# ----------------------------------------------------------------------------
# Validation, summary, manifest
# ----------------------------------------------------------------------------
def validate(panel: pd.DataFrame, wide: pd.DataFrame) -> None:
    assert_unique(panel, ["USER_ID"], "panel")
    assert panel["a1c"].between(3.0, 20.0).all(), "a1c outside 3–20"
    assert (panel["n_dense_days"] >= MIN_DENSE_DAYS).all(), "n_dense_days < 14"
    for column in ANALYTES + DERIVED:
        offsets = panel[f"{column}_offset_days"].dropna()
        assert offsets.abs().le(LAB_BAND_DAYS).all(), f"{column} offset beyond +/-{LAB_BAND_DAYS} d"
        assert panel[column].notna().eq(offsets.reindex(panel.index).notna()).all(), \
            f"{column}: value/offset presence mismatch"
    assert panel["bmi_offset_days"].dropna().abs().le(BMI_BAND_DAYS).all(), "BMI offset beyond +/-90 d"
    assert panel.loc[panel["bmi"].notna(), "bmi_offset_days"].notna().all()
    # HOMA-IR only where FPG and insulin share a date.
    homa = panel.loc[panel["homa_ir"].notna(), ["USER_ID", "homa_ir_date"]]
    pairs = wide.loc[wide["fasting_glucose"].notna() & wide["fasting_insulin"].notna(), ["USER_ID", "DRAW_DATE"]]
    pairs = pairs.assign(homa_ir_date=pairs["DRAW_DATE"].dt.date.astype(str))
    matched = homa.merge(pairs[["USER_ID", "homa_ir_date"]].drop_duplicates(), on=["USER_ID", "homa_ir_date"])
    assert len(matched) == len(homa), "homa_ir present without same-day FPG + insulin"
    assert set(REQUIRED_COLUMNS) <= set(panel.columns), \
        f"missing contract columns: {sorted(set(REQUIRED_COLUMNS) - set(panel.columns))}"


REQUIRED_COLUMNS = (
    ["USER_ID", "anchor_date", "anchor_year", "a1c", "a1c_draw_rn", "n_a1c_draws", "n_dense_days",
     "n_readings", "first_dense_day", "last_dense_day", "tz_offset_hours", "sex", "age_bucket",
     "bmi", "bmi_offset_days", "weight_lb"]
    + [f"{a}{s}" for a in ANALYTES for s in ("", "_date", "_offset_days")] + DERIVED
    + wf.WINDOW_METRIC_COLUMNS + ["mean_glucose_readings"]
    + [f"w30_{c}" for c in wf.W30_COLUMNS]
    + ["diabetes_range_ever", "no_diabetes_range", "a1c_normal_strict"]
)


def markdown_table(frame: pd.DataFrame, decimals: int = 2) -> str:
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:,.{decimals}f}"
        return f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)
    header = "| " + " | ".join(frame.columns) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_summary_section(window: str, body: str) -> None:
    marker_start = f"<!-- window:{window} start -->"
    marker_end = f"<!-- window:{window} end -->"
    section = f"{marker_start}\n{body}\n{marker_end}\n"
    if SUMMARY_PATH.exists():
        text = SUMMARY_PATH.read_text()
        if marker_start in text and marker_end in text:
            head, rest = text.split(marker_start, 1)
            _, tail = rest.split(marker_end, 1)
            text = head + section.rstrip("\n") + tail
        else:
            text = text.rstrip("\n") + "\n\n" + section
    else:
        text = ("# Canonical lab–CGM panel — build summary\n\n"
                "Builder: analysis/canonical_panel/build_panel.py.\n\n" + section)
    SUMMARY_PATH.write_text(text)


def write_manifest(window: str, entry: dict) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    manifest[window] = entry
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    global DB
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", choices=list(WINDOWS), default="symmetric")
    parser.add_argument("--force", action="store_true", help="rebuild parquet caches")
    parser.add_argument("--chunk-users", type=int, default=300)
    parser.add_argument("--threads", type=int, default=3, help="Maximum DuckDB worker threads")
    parser.add_argument("--memory-limit", default="3GB", help="DuckDB memory bound; pandas batches also need memory")
    parser.add_argument("--database", type=Path, default=DB, help="Read-only study database")
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Directory for the panel, caches, and aggregate provenance")
    parser.add_argument("--grid-cache", type=Path, help="Read-only existing dense-day grid; validated against the candidate days")
    parser.add_argument("--require-grid-cache", action="store_true", help="Fail rather than scan raw CGM if an existing grid cannot be verified")
    parser.add_argument("--skip-rich-features", action="store_true", help="Do not scan raw CGM for rich features; sidecar records an incomplete rich-feature stage")
    parser.add_argument("--skip-reading-mean", action="store_true", help="Omit the all-reading mean glucose (mean_glucose_readings) and its raw CGM scan")
    args = parser.parse_args(argv)
    if args.chunk_users < 1:
        parser.error("--chunk-users must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.force and args.require_grid_cache:
        parser.error("--force cannot be combined with --require-grid-cache")
    DB = args.database.resolve()
    configure_output(args.output_dir)
    window = args.window
    before, after, tight_before, tight_after, csv_name = WINDOWS[window]
    started = time.time()
    log(f"canonical panel: window={window} (days -{before}..+{after}; w30 = -{tight_before}..+{tight_after})")

    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET threads = ?", [args.threads])
    con.execute("SET memory_limit = ?", [args.memory_limit])
    con.execute("SET temp_directory = ?", [str(CACHE / "duckdb_spill")])
    con.execute("SET TimeZone='UTC'")

    # --- funnel ---------------------------------------------------------------------
    lab_audit = build_a1c_draws(con)
    n_draws, n_users_with_draws = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT USER_ID) FROM a1c_draws").fetchone()
    gate1 = sensor_day_gate(con, before, after)
    gate2 = gate1[gate1["is_internal"].eq(False).fillna(False)]
    gate3 = gate2[gate2["tz_offset_hours"].notna()]
    funnel = {
        "a1c_draws": int(n_draws),
        "users_with_a1c_draw": int(n_users_with_draws),
        "users_any_window_ge14_sensor_days": int(len(gate1)),
        "users_after_internal_exclusion": int(len(gate2)),
        "users_after_tz_gate": int(len(gate3)),
        "excluded_internal": int(gate1["is_internal"].eq(True).sum()),
        "excluded_unknown_internal_status": int(gate1["is_internal"].isna().sum()),
        "excluded_missing_tz": int(len(gate2) - len(gate3)),
    }
    log(f"funnel: draws={n_draws:,} users={n_users_with_draws:,} -> >=14 sensor-days {len(gate1):,} "
        f"-> non-internal {len(gate2):,} -> TZ {len(gate3):,}")

    # --- dense-day grid + anchors --------------------------------------------------------
    grid = build_dense_day_grid(con, args.force, args.chunk_users, args.grid_cache, args.require_grid_cache)
    grid_provenance = grid.attrs.get("source_cache", {})
    dense = grid[(grid["n_cells"] >= wf.MIN_15M_CELLS) & (grid["n_raw_readings"] >= wf.RAW_DAILY_GATE)].reset_index(drop=True)
    anchors = select_anchors(con, dense, before, after)
    if anchors.empty:
        raise ValueError("No participants have an accepted HbA1c anchor with sufficient dense CGM days")
    funnel["users_after_dense_day_gate"] = int(len(anchors))
    funnel["excluded_dense_day_gate"] = int(len(set(gate3.USER_ID)-set(anchors.USER_ID)))
    funnel["anchors_not_in_day_panel_funnel"] = int(len(set(anchors.USER_ID)-set(gate3.USER_ID)))
    funnel["anchor_is_first_a1c_draw"] = int((anchors["a1c_draw_rn"] == 1).sum())
    funnel["anchor_tz_population_fallback"] = int((anchors["tz_source"] == "population_fallback").sum())
    log(f"anchors: {len(anchors):,} users pass the dense-day gate "
        f"({funnel['excluded_dense_day_gate']:,} excluded)")

    # --- CGM ---------------------------------------------------------------------------
    if feature_metadata()["core_grid_feature_version"] == "unversioned":
        raise ValueError("_window_features.py defines no CORE_GRID_FEATURE_VERSION; the panel needs versioned core metrics")
    metrics = compute_window_metrics(anchors, dense, before, after, tight_before, tight_after)
    if args.skip_rich_features:
        rich = anchors[["USER_ID"]].copy()
    else:
        if feature_metadata()["trace_feature_version"] == "unversioned":
            raise ValueError("_window_features.py defines no RICH_FEATURE_VERSION; rich features need the versioned trace extractor")
        rich = compute_rich_features(con, anchors, dense, before, after, window, args.force, args.chunk_users)
    reading_mean = (anchors[["USER_ID"]].assign(mean_glucose_readings=np.nan, n_readings_all_days=np.nan)
                    if args.skip_reading_mean else compute_reading_level_mean(con, anchors, before, after))

    # --- labs / anthropometrics / flags --------------------------------------------------
    wide = lab_wide(con)
    labs = nearest_within_band(anchors, wide, "DRAW_DATE", ANALYTES + DERIVED, LAB_BAND_DAYS)
    anthro = anthropometrics(con, anchors)

    panel = (anchors.drop(columns=["tz_source"])
             .merge(anthro, on="USER_ID", how="left")
             .merge(labs, on="USER_ID", how="left")
             .merge(metrics, on="USER_ID", how="left")
             .merge(reading_mean, on="USER_ID", how="left")
             .merge(rich, on="USER_ID", how="left"))
    panel = panel.merge(cohort_flags(con, panel, wide), on="USER_ID", how="left")
    panel["anchor_date"] = panel["anchor_date"].dt.date.astype(str)
    assert panel["mean_glucose"].notna().all(), "window metrics missing for some anchors"
    validate(panel, wide)

    con.close()

    # --- write ---------------------------------------------------------------------------
    ordered = list(dict.fromkeys(REQUIRED_COLUMNS)) + [c for c in panel.columns if c not in REQUIRED_COLUMNS]
    panel = panel[ordered].sort_values("USER_ID").reset_index(drop=True)
    csv_path = OUT / csv_name
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    panel.to_csv(temporary_csv, index=False)
    temporary_csv.replace(csv_path)
    database_stat = DB.stat()
    sidecar = {
        **feature_metadata(),
        "window": window, "window_days_before": before, "window_days_after": after,
        "complete_features": not args.skip_rich_features and not args.skip_reading_mean,
        "rich_features_complete": not args.skip_rich_features,
        "reading_mean_complete": not args.skip_reading_mean,
        "dense_grid_complete": True,
        "grid_provenance": grid_provenance,
        "laboratory_processing": lab_audit,
        "laboratory_definition": {
            "units": CANONICAL_UNITS, "validity_bounds_after_conversion": ANALYTE_RANGES,
            "a1c_rounding": "half_up_tenth_with_1e-6_percentage_point_float32_tolerance",
            "conflict_rule": "A1c normalized raw same-day spread >0.30001 percentage points",
            "ever_diabetes_rule": "Any accepted raw rounded HbA1c >=6.5% or glucose >=126 mg/dL, including conflict days",
            "account_rule": "IS_INTERNAL_USER=false; missing account status excluded",
            "anchor_rule": "First accepted chronological HbA1c with >=14 qualifying dense days; reselect after exclusions",
            "nearest_lab_ties": "earlier normalized calendar date",
        },
        "database_identity": {"filename": DB.name, "size": database_stat.st_size, "mtime_ns": database_stat.st_mtime_ns},
        "database_timezone": "UTC",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metadata_path = csv_path.with_suffix(".metadata.json")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(sidecar, indent=2, default=str) + "\n")
    temporary_metadata.replace(metadata_path)
    runtime = time.time() - started
    log(f"wrote {display_path(csv_path)}: {len(panel):,} rows x {panel.shape[1]} columns "
        f"in {runtime:.0f}s")

    analyte_rows = []
    for column in ANALYTES + DERIVED:
        offsets = panel[f"{column}_offset_days"].dropna()
        analyte_rows.append({
            "analyte": column, "n": int(panel[column].notna().sum()),
            "median_offset_days": float(offsets.median()) if len(offsets) else np.nan,
            "p90_abs_offset_days": float(offsets.abs().quantile(0.9)) if len(offsets) else np.nan,
            "n_offset_0": int((offsets == 0).sum()),
        })
    bmi_offsets = panel["bmi_offset_days"].dropna()
    analyte_rows.append({
        "analyte": "bmi (+/-90 d)", "n": int(panel["bmi"].notna().sum()),
        "median_offset_days": float(bmi_offsets.median()) if len(bmi_offsets) else np.nan,
        "p90_abs_offset_days": float(bmi_offsets.abs().quantile(0.9)) if len(bmi_offsets) else np.nan,
        "n_offset_0": int((bmi_offsets == 0).sum()),
    })
    analyte_table = pd.DataFrame(analyte_rows)
    funnel_table = pd.DataFrame({"gate": list(funnel), "n": list(funnel.values())})
    dense_desc = panel["n_dense_days"].describe(percentiles=[0.1, 0.5, 0.9])
    tiers = {t: int(panel[t].sum()) for t in ("diabetes_range_ever", "no_diabetes_range", "a1c_normal_strict")}
    tiers["all_lab_linked"] = int(len(panel))
    w30_n = int(panel["w30_mean_glucose"].notna().sum())
    rich_n = int(panel["acf_30m"].notna().sum()) if "acf_30m" in panel else 0

    body = f"""## Window: {window}  (days -{before}..+{after} around the anchor; w30 = -{tight_before}..+{tight_after})

Built {datetime.now(timezone.utc).isoformat(timespec='seconds')} · runtime {runtime:.0f}s · rows {len(panel):,} · columns {panel.shape[1]} · file `{csv_name}`

### Selection funnel

{markdown_table(funnel_table)}

`users_any_window_ge14_sensor_days` counts observed-day eligibility around accepted HbA1c
dates in the database's day_panel table, before account/timezone exclusion. This is a
descriptive funnel, not an eligibility stage. Accepted raw
local-day counts determine current eligibility, including days missing from that table.

Laboratory rule `{LAB_DEFINITION_VERSION}` accepted {lab_audit['accepted_raw_rows_before_conflicts']:,}
raw laboratory records across the supported analytes, excluded
{lab_audit['unknown_or_missing_unit_rows']:,} records for unknown/missing units and
{lab_audit['a1c_conflict_days_excluded']:,} conflicting HbA1c days.
Detailed aggregate attrition and unit conversions are in the CSV metadata sidecar.

### Dense days per participant

count {int(dense_desc['count']):,} · median {dense_desc['50%']:.0f} · p10 {dense_desc['10%']:.0f} · p90 {dense_desc['90%']:.0f} · min {dense_desc['min']:.0f} · max {dense_desc['max']:.0f} ·
contiguous runs median {panel['n_contiguous_runs'].median():.0f} · MODD day-pairs median {panel['n_modd_day_pairs'].median():.0f} ·
w30 available (>=14 dense days in the tight window) {w30_n:,} · rich features available {rich_n:,}

### Cohort tiers

{markdown_table(pd.DataFrame({'tier': list(tiers), 'n': list(tiers.values())}))}

### Analytes: n and offset distribution (days from anchor; +/-7 d band, BMI +/-90 d)

{markdown_table(analyte_table)}
"""
    if window == "symmetric":
        body += """
### Implementation choices

- Laboratory units are explicitly recognized and converted before individual-record
  validity limits. Missing, malformed, or unsupported units are excluded. Valid same-date
  records use their median; HbA1c dates with normalized raw spread >0.30001 percentage
  points are excluded before anchor selection. HbA1c medians use half-up rounding to
  one decimal before classification, with a float32 tolerance. Accepted raw high values
  still contribute to ever-diabetes flags even on conflicting days or when a median is lower.
- TZ gate uses `users.TZ_OFFSET_HOURS IS NOT NULL`. The ETL fills every user
  (`TZ_SOURCE` = food_logs_mode or population_fallback); the
  number of panel users on the population-fallback offset is reported as
  `anchor_tz_population_fallback` for transparency.
- Dense days require >=230 accepted raw readings in 20–500 mg/dL and >=77 occupied
  15-minute cells on the participant-local date. The database's day_panel counts are not used.
- Interpolation (<=2 cells) is applied within runs of consecutive calendar days only; a
  10-cell NaN spacer separates non-adjacent runs so lag-1..10 pairs never cross a boundary.
- MAGE proxy = mean daily successive turning-point amplitude above the daily sample SD,
  from accepted unique raw timestamps; runs stop at gaps >10 minutes. Strict extrema
  and each run's endpoints are retained. Undefined daily values are omitted, not zero.
- Skewness / kurtosis use bias-corrected (sample) estimators, matching DuckDB's
  SKEWNESS / KURTOSIS in `analysis/metabolic/cgm_features.py`. TIR uses inclusive bounds (BETWEEN).
- Rich features use the versioned gap-aware shared extractor on raw readings (20–500)
  on the window's DENSE days. When requested, every primary CGM
  column is computed from the same days. The extractor's own `n_readings` is stored as
  `rich_n_readings`; the panel's `n_readings` counts accepted raw rows over dense days.
  The extractor's column names are kept verbatim (`peak_prom_mean`, `recovery_mean`,
  `mean_morning`, ...).
- `mean_glucose_readings` averages accepted raw readings (20–500 mg/dL) whose
  participant-local calendar day lies in the window (all days, not only dense days).
- Derived indices (HOMA-IR, TyG, TG:HDL) are computed on same-day rows and then selected
  as their own nearest-within-+/-7-d draw, so their date may differ from the individually
  nearest FPG/TG draw; `<derived>_date` / `<derived>_offset_days` columns are added.
- Weight: 80–500 lb, median per day, nearest within +/-90 d (ties earlier);
  height = users.HEIGHT_INCHES in 48–90 in; BMI outside 15–60 is set to NaN while
  `weight_lb` and `bmi_offset_days` are kept.
- `no_diabetes_range` treats a missing anchor-band fasting glucose as not disqualifying.
- Preceding window: anchor = first accepted draw with >=14 dense days in days -89..0;
  w30_* uses days -29..0.
- Extra columns beyond the contract: `n_contiguous_runs`, `n_modd_day_pairs`,
  `n_15m_observed`, `n_15m_interpolated`, `n_readings_all_days`, derived `_date`/`_offset_days`.
"""
    write_summary_section(window, body)
    write_manifest(window, {
        **sidecar,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "window": window,
        "window_days_before": before, "window_days_after": after,
        "tight_days_before": tight_before, "tight_days_after": tight_after,
        "min_dense_days": MIN_DENSE_DAYS, "raw_daily_gate": wf.RAW_DAILY_GATE,
        "min_15m_cells": wf.MIN_15M_CELLS, "lab_band_days": LAB_BAND_DAYS,
        "bmi_band_days": BMI_BAND_DAYS, "chunk_users": args.chunk_users,
        "output_csv": csv_name, "n_rows": int(len(panel)), "n_columns": int(panel.shape[1]),
        "funnel": funnel, "cohort_tiers": tiers,
        "analytes": analyte_table.to_dict(orient="records"),
        "runtime_seconds": round(runtime, 1),
    })
    log(f"summary -> {display_path(SUMMARY_PATH)}; manifest -> {display_path(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
