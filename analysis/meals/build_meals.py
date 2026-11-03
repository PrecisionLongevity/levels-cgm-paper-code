#!/usr/bin/env python3
"""Build the all-food-isolated, observed-cell meal response substrate.

Identifies discrete meal events from all food logs (not restricted to logs with
macro data), pairs each with its CGM trace on a 5-minute grid from -30 to +240
minutes relative to meal start, and derives meal-response measures restricted to
cells with complete, observed (non-imputed) CGM coverage.

Meal clustering: consecutive food logs for a participant less than 15 minutes
apart are merged into one meal event (event timestamp = first log in the
cluster). Isolation is defined by gaps to the nearest neighboring cluster
(e.g. IS_ISOLATED_2_2 = no other cluster within 2h before or after).

Response measures (restricted to meals with complete 2-hour CGM coverage):
  BASELINE_PRE   - mean of the three 5-minute cells in [-15, 0) minutes
  IAUC_2H/4H     - 5 * sum of positive deviations from baseline over the window
  PEAK_GLUCOSE_2H / PEAK_RISE_2H / TIME_TO_PEAK_MIN - from the observed 0-120 min cells
  RETURN_10_MIN / RETURN_10_REACHED - time to return within 10 mg/dL of baseline
  AUC_SHAPE_RATIO - IAUC_2H normalized by a rectangular peak-height proxy

Inputs:
  database tables (via LEVELS_DB): food_logs, macro_nutrients, users, glucose_readings

Outputs (OUTPUTS/meals/):
  meal_events_all_food.parquet   - all-food meal events with isolation flags (intermediate)
  meal_cells_observed.parquet    - per-event observed 5-minute CGM cells (intermediate)
  meal_traces_observed.parquet   - per-event CGM grid for shape-eligible meals (participant-level)
  meal_food_substrate.parquet    - per-event meal/food features and response measures (participant-level)
  meal_shape_substrate.parquet   - per-event response measures for shape-eligible meals (participant-level)

Usage: python analysis/meals/build_meals.py [--database PATH] [--output-dir DIR]
  [--stage {all,events,traces,responses}] [--batches N] [--threads N] [--memory-limit SIZE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
VERSION = "all-food-observed-cell-meals"
DEFAULT_OUT = OUTPUTS / 'meals'
GRID = np.arange(-27.5, 240., 5.)
MACROS = {"CARBS": ("CARBS_G", 1500), "PROTEIN": ("PROTEIN_G", 1500),
          "FAT": ("FAT_G", 1500), "FIBER": ("FIBER_G", 200),
          "CALORIES": ("CALORIES", 50000)}


def sql_path(path):
    return "'" + str(Path(path).resolve()).replace("'", "''") + "'"


def event_query(batch=0, batches=1):
    """All logs determine clusters; macro sums never skip a missing item/log."""
    clean = ", ".join(
        f"CASE WHEN {raw} BETWEEN 0 AND {hi} THEN {raw} END AS {name}"
        for raw, (name, hi) in MACROS.items())
    sums = ", ".join(f"CASE WHEN count({name})=count(*) THEN sum({name}) END AS {name}"
                     for name, _ in MACROS.values())
    fields = ", ".join("m." + name for name, _ in MACROS.values())
    return f"""
    WITH macro_clean AS (
      SELECT USER_ID, LOG_ID, FOOD_ITEM_NAME, BARCODE, {clean}
      FROM macro_nutrients WHERE hash(USER_ID) % {batches} = {batch}
    ), macros AS (
      SELECT USER_ID, LOG_ID, count(*) AS n_macro_rows, {sums},
             CASE WHEN count(*)=1 THEN nullif(lower(trim(first(FOOD_ITEM_NAME))), '') END AS food_key,
             CASE WHEN count(*)=1 THEN nullif(trim(first(BARCODE)), '') END AS barcode_key
      FROM macro_clean GROUP BY USER_ID, LOG_ID
    ), logs AS (
      SELECT f.USER_ID, f.ID AS LOG_ID, f.TIMESTAMP AS MEAL_TS, f.TYPE,
             f.FOOD_CATEGORY, m.n_macro_rows, m.food_key, m.barcode_key, {fields},
             CASE WHEN u.TZ_SOURCE='food_logs_mode' THEN u.TZ_OFFSET_HOURS END AS tz_offset,
             lag(f.TIMESTAMP) OVER (PARTITION BY f.USER_ID ORDER BY f.TIMESTAMP, f.ID) AS prior_ts
      FROM food_logs f JOIN users u ON u.USER_ID=f.USER_ID
      LEFT JOIN macros m ON m.USER_ID=f.USER_ID AND m.LOG_ID=f.ID
      WHERE f.TYPE IN ('food', 'food_with_macros') AND f.TIMESTAMP IS NOT NULL
        AND u.IS_INTERNAL_USER=false AND hash(f.USER_ID) % {batches} = {batch}
    ), numbered AS (
      SELECT *, sum(CASE WHEN prior_ts IS NULL OR MEAL_TS-prior_ts > INTERVAL '15 minutes'
                         THEN 1 ELSE 0 END)
        OVER (PARTITION BY USER_ID ORDER BY MEAL_TS, LOG_ID ROWS UNBOUNDED PRECEDING) AS cluster
      FROM logs
    ), clusters AS (
      SELECT USER_ID, first(LOG_ID ORDER BY MEAL_TS, LOG_ID) AS LOG_ID,
             min(MEAL_TS) AS MEAL_TS, max(MEAL_TS) AS LAST_TS,
             count(*) AS N_LOGS_IN_CLUSTER, sum(coalesce(n_macro_rows,0)) AS N_MACRO_ROWS,
             count(*) FILTER (WHERE TYPE='food') AS N_ORDINARY_FOOD_LOGS,
             {sums}, first(tz_offset) AS tz_offset,
             CASE WHEN count(*)=1 THEN first(food_key) END AS food_key,
             CASE WHEN count(*)=1 THEN first(barcode_key) END AS barcode_key,
             CASE WHEN count(*)=1 THEN first(FOOD_CATEGORY) END AS FOOD_CATEGORY
      FROM numbered GROUP BY USER_ID, cluster
    ), neighbors AS (
      SELECT *, lag(LAST_TS) OVER (PARTITION BY USER_ID ORDER BY MEAL_TS, LOG_ID) AS PRIOR_LOG_TS,
                lead(MEAL_TS) OVER (PARTITION BY USER_ID ORDER BY MEAL_TS, LOG_ID) AS NEXT_LOG_TS
      FROM clusters
    ), intervals AS (
      SELECT *, 'v2_' || md5(USER_ID || ':' || LOG_ID::VARCHAR) AS MEAL_EVENT_ID,
             epoch(MEAL_TS-PRIOR_LOG_TS)/3600. AS HOURS_SINCE_PRIOR_LOG,
             epoch(NEXT_LOG_TS-MEAL_TS)/3600. AS HOURS_TO_NEXT_LOG,
             epoch(LAST_TS-MEAL_TS)/60. AS CLUSTER_DURATION_MIN,
             hour(MEAL_TS + tz_offset * INTERVAL '1 hour') AS MEAL_HOUR
      FROM neighbors
    ) SELECT *,
      (PRIOR_LOG_TS IS NULL OR HOURS_SINCE_PRIOR_LOG>=2) AS IS_FASTED_2H,
      (PRIOR_LOG_TS IS NULL OR HOURS_SINCE_PRIOR_LOG>=4) AS IS_FASTED_4H,
      (PRIOR_LOG_TS IS NULL OR HOURS_SINCE_PRIOR_LOG>=6) AS IS_FASTED_6H,
      (PRIOR_LOG_TS IS NULL OR HOURS_SINCE_PRIOR_LOG>=2) AND
        (NEXT_LOG_TS IS NULL OR HOURS_TO_NEXT_LOG>=2) AS IS_ISOLATED_2_2,
      (PRIOR_LOG_TS IS NULL OR HOURS_SINCE_PRIOR_LOG>=3) AND
        (NEXT_LOG_TS IS NULL OR HOURS_TO_NEXT_LOG>=2) AS IS_ISOLATED_3_2,
      (PRIOR_LOG_TS IS NULL OR HOURS_SINCE_PRIOR_LOG>=4) AND
        (NEXT_LOG_TS IS NULL OR HOURS_TO_NEXT_LOG>=3) AS IS_ISOLATED_4_3,
      (PRIOR_LOG_TS IS NULL OR HOURS_SINCE_PRIOR_LOG>=3) AND
        (NEXT_LOG_TS IS NULL OR HOURS_TO_NEXT_LOG>=4) AS IS_ISOLATED_3_4
    FROM intervals
    """


def trace_query(events, batch=0, batches=1):
    """Hour equijoin followed by exact half-open relative-cell assignment.

    Duplicates are averaged at their native timestamp before cell averaging.
    Meal-relative cells use floor, including negative offsets and fractional seconds.
    """
    cells = ", ".join(f"max(glucose) FILTER (WHERE cell={i}) AS g{i}" for i in range(54))
    return f"""
    WITH meals AS (
      SELECT USER_ID, MEAL_EVENT_ID, epoch(MEAL_TS) AS meal_s
      FROM read_parquet({sql_path(events)})
      WHERE IS_ISOLATED_2_2 AND CLUSTER_DURATION_MIN<=15 AND N_MACRO_ROWS>0
        AND hash(USER_ID) % {batches} = {batch}
    ), buckets AS (
      SELECT m.*, unnest(generate_series(floor((meal_s-1800)/3600)::BIGINT,
                                        floor((meal_s+14400)/3600)::BIGINT)) AS bucket
      FROM meals m
    ), timestamps AS (
      SELECT b.USER_ID, b.MEAL_EVENT_ID, b.meal_s, g.TS, avg(g.VALUE) AS glucose
      FROM buckets b JOIN glucose_readings g
        ON g.USER_ID=b.USER_ID AND floor(epoch(g.TS)/3600)::BIGINT=b.bucket
      WHERE epoch(g.TS)>=b.meal_s-1800 AND epoch(g.TS)<b.meal_s+14400
        AND g.VALUE BETWEEN 20 AND 500
      GROUP BY b.USER_ID, b.MEAL_EVENT_ID, b.meal_s, g.TS
    ), cell_means AS (
      SELECT USER_ID, MEAL_EVENT_ID, floor((epoch(TS)-meal_s+1800)/300)::INTEGER AS cell,
             avg(glucose) AS glucose
      FROM timestamps GROUP BY USER_ID, MEAL_EVENT_ID, cell
    ) SELECT USER_ID, MEAL_EVENT_ID, count(*) AS n_cells, {cells}
      FROM cell_means GROUP BY USER_ID, MEAL_EVENT_ID HAVING count(*)>=27
    """


def derive_responses(frame):
    """Return complete two-hour responses; preserve observed four-hour eligibility."""
    if frame.duplicated(["USER_ID", "MEAL_EVENT_ID"]).any():
        raise ValueError("Duplicate participant/event keys")
    grid = frame[[f"g{i}" for i in range(54)]].to_numpy(float)
    eligible = np.isfinite(grid[:, 3:30]).all(axis=1)
    out = frame.loc[eligible].copy().reset_index(drop=True)
    grid = grid[eligible]
    baseline = grid[:, 3:6].mean(axis=1)
    delta = grid[:, 6:30] - baseline[:, None]
    peaks = delta.argmax(axis=1)
    peak = delta[np.arange(len(out)), peaks]
    area = 5 * np.maximum(delta, 0).sum(axis=1)
    out["BASELINE_PRE"] = baseline
    out["IAUC_2H"] = area
    out["PEAK_GLUCOSE_2H"] = peak + baseline
    out["PEAK_RISE_2H"] = peak
    out["TIME_TO_PEAK_MIN"] = 2.5 + 5 * peaks
    post_peak = np.arange(24)[None, :] > peaks[:, None]
    returned = post_peak & (delta <= 10)
    observed_return = returned.any(axis=1)
    out["RETURN_10_MIN"] = np.where(peak <= 10, 0.,
        np.where(observed_return, 2.5 + 5 * returned.argmax(axis=1), np.nan))
    out["RETURN_10_REACHED"] = (peak <= 10) | observed_return
    out["AUC_SHAPE_RATIO"] = np.divide(area, 120*peak, out=np.full(len(out), np.nan), where=peak>=10)
    out["CGM_COVERAGE_OK"] = True
    out["SHAPE_COVERAGE_OK"] = np.isfinite(grid).all(axis=1)
    out["IAUC_4H"] = np.where(out.SHAPE_COVERAGE_OK,
                             5*np.maximum(grid[:, 6:54]-baseline[:, None], 0).sum(axis=1), np.nan)
    return out


def source_signature(database):
    stat = Path(database).resolve().stat()
    return {"source_size_bytes": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}


def write_metadata(path, extra):
    metadata = {"definition_version": VERSION, **extra}
    Path(path).with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=database_path())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stage", choices=["all", "events", "traces", "responses"], default="all")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GB")
    args = parser.parse_args(argv)
    if args.batches<1 or args.threads<1:
        parser.error("batches and threads must be positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    con = duckdb.connect(str(args.database), read_only=True)
    con.execute(f"SET threads={args.threads}")
    con.execute("SET memory_limit=?", [args.memory_limit])
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET TimeZone='UTC'")
    con.execute("SET temp_directory=?", [str(out / "_meal_spill")])
    events = out / "meal_events_all_food.parquet"
    traces = out / "meal_cells_observed.parquet"
    signature = source_signature(args.database)
    for stage, target in [("events", events), ("traces", traces)]:
        if args.stage not in ("all", stage):
            continue
        parts = out / ("_"+stage+"_parts")
        parts.mkdir(exist_ok=True)
        paths = []
        total = 0
        for batch in range(args.batches):
            part = parts / f"part_{batch:03}.parquet"
            query = event_query(batch, args.batches) if stage=="events" else trace_query(events, batch, args.batches)
            con.execute(f"COPY ({query}) TO {sql_path(part)} (FORMAT PARQUET, COMPRESSION ZSTD)")
            n = con.execute(f"SELECT count(*) FROM read_parquet({sql_path(part)})").fetchone()[0]
            total += n
            paths.append(part)
            print(f"{stage} {batch+1}/{args.batches}: {n:,} rows; {time.time()-started:.0f}s elapsed", flush=True)
        # An explicit file list prevents stale extra partitions from entering a rerun.
        inputs = "[" + ",".join(sql_path(p) for p in paths) + "]"
        temporary = target.with_suffix(".partial.parquet")
        con.execute(f"COPY (SELECT * FROM read_parquet({inputs})) TO {sql_path(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)")
        temporary.replace(target)
        write_metadata(target, {**signature, "rows": total})
        for part in paths:
            part.unlink()
        print(f"Wrote {target.name}: {total:,} rows", flush=True)
    if args.stage in ("all", "responses"):
        # Process response derivation in participant partitions to bound Python RAM.
        food_parts, shape_parts, grid_parts = [], [], []
        for batch in range(args.batches):
            frame = con.execute(f"""SELECT e.*, t.* EXCLUDE (USER_ID, MEAL_EVENT_ID)
              FROM read_parquet({sql_path(events)}) e JOIN read_parquet({sql_path(traces)}) t
                USING(USER_ID, MEAL_EVENT_ID) WHERE hash(e.USER_ID)%{args.batches}={batch}""").fetchdf()
            responses = derive_responses(frame)
            shape = responses.loc[responses.SHAPE_COVERAGE_OK & responses.IS_ISOLATED_3_4].copy()
            grid_columns = [f"g{i}" for i in range(54)]
            for label, data, path_list in (
                ("food", responses.drop(columns=grid_columns), food_parts),
                ("shape", shape.drop(columns=grid_columns), shape_parts),
                ("grid", shape[["USER_ID", "MEAL_EVENT_ID", "BASELINE_PRE", "n_cells"]+grid_columns], grid_parts)):
                path = out / f"_response_{label}_{batch:03}.parquet"
                data.to_parquet(path, index=False)
                path_list.append(path)
        definition = {**signature, "all_food_logs_used": True, "meal_log_source": "all_food_logs",
            "matched_shape_iauc_baseline": True, "noninternal_only": True,
            "baseline_definition": "Mean of all three observed 5-minute cells in [-15,0)",
            "iauc_definition": "5 * sum(positive cell mean deviation), all 24 cells [0,120)",
            "cgm_gap_policy": "No imputation; all required 5-minute cells observed",
            "grid_minutes": GRID.tolist(), "grid_interval_minutes": [-30,240]}
        for name, paths, pre, post in [
            ("meal_traces_observed.parquet", grid_parts, 180,240),
            ("meal_food_substrate.parquet", food_parts, 120,120),
            ("meal_shape_substrate.parquet", shape_parts, 180,240)]:
            target = out / name
            inputs = "[" + ",".join(sql_path(p) for p in paths) + "]"
            con.execute(f"COPY (SELECT * FROM read_parquet({inputs})) TO {sql_path(target)} (FORMAT PARQUET, COMPRESSION ZSTD)")
            n, users = con.execute(f"SELECT count(*),count(DISTINCT USER_ID) FROM read_parquet({sql_path(target)})").fetchone()
            write_metadata(target, {**definition, "rows": n, "participants": users,
                "shape_end_minutes":240, "pre_meal_isolation_minutes":pre, "post_meal_isolation_minutes":post,
                "isolation_definition":f"No known other cluster in prior {pre} min or subsequent {post} min; all food logs"})
            for part in paths:
                part.unlink()
            print(f"Wrote {name}: {n:,} rows, {users:,} participants", flush=True)
    con.close()


if __name__ == "__main__":
    main()
