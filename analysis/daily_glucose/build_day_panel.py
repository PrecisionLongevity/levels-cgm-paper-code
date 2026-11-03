"""Build the dense, participant-local-day glucose panel used for the daily glycemic analyses.

Reads the glucose and user tables from the study database and aggregates readings into
participant-local calendar days (timezone fixed per participant from food-log timing).

Population: non-internal accounts, known food-inferred timezones, UTC offset in [-12, 14].
Default cohort (`--cohort no_observed_diabetes_range`): excludes participants with any
accepted diabetes-range laboratory value ever observed; missing screening laboratories are
treated as unknown, not as a diagnosis of no diabetes. `--cohort all` keeps everyone.

Accepted glucose readings: 20-500 mg/dL. A day is kept only if dense: >=MIN_READINGS raw
readings and >=MIN_CELLS observed 15-minute cells (see cgmpaper day-panel constants).

Inputs: database tables `users`, `glucose_readings` (via LEVELS_DB); normalized lab flags
from cgmpaper.lab_reference.load_normalized_labs.

Outputs:
  OUTPUTS/daily_glucose/local_day_provenance.json - run metadata (participant counts, dense
    day counts, lab/exposure audit information).
  Cached intermediate parquet files (gluc_local_day.parquet, user_tz_offset.parquet) used by
    downstream daily-glucose analyses.

Usage: `python analysis/daily_glucose/build_day_panel.py [--cohort no_observed_diabetes_range|all]
  [--stage all|exposures]` (--stage exposures reuses the cached CGM/timezone substrate to
  refresh only exposure data).
"""
import argparse
import json
import sys
import time
import duckdb
from _day_panel import ROOT, OUT, CACHE, DAY_VERSION, MIN_READINGS, MIN_CELLS, EXPOSURES, build_exposure_cache

sys.path.insert(0, str(ROOT))
from cgmpaper.paths import database_path  # noqa: E402
from cgmpaper.lab_reference import load_normalized_labs

def load_saved_cgm(cohort):
    """Load the saved CGM substrate metadata for an exposure-only refresh."""
    record = json.loads((CACHE/"local_day_metadata.json").read_text())
    return record


def build(cohort="no_observed_diabetes_range", stage="all"):
    CACHE.mkdir(parents=True, exist_ok=True)
    record = load_saved_cgm(cohort) if stage == "exposures" else None
    con = duckdb.connect(str(database_path()), read_only=True)
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='3GB'")
    con.execute("SET TimeZone='UTC'")
    con.execute("SET temp_directory=?", [str(CACHE/"spill")])
    if stage == "exposures":
        record["exposure_audit"] = build_exposure_cache(con)
        record["cache_files"][EXPOSURES.name] = {"size":EXPOSURES.stat().st_size,
                                               "mtime_ns":EXPOSURES.stat().st_mtime_ns}
        for path in [CACHE/"local_day_metadata.json", OUT/"local_day_provenance.json"]:
            path.write_text(json.dumps(record,indent=2,default=int)+"\n")
        con.close()
        print(json.dumps(record["exposure_audit"],indent=2,default=int),flush=True)
        return
    _, flags, lab_audit = load_normalized_labs(con, types=["a1c", "fasting_glucose"])
    con.register("screening_flags", flags)
    users = con.sql("""
      SELECT u.USER_ID,u.TZ_OFFSET_HOURS AS TZ_OFFSET_HR,
             f.USER_ID IS NOT NULL AS HAS_SCREENING_LAB,
             coalesce(f.diabetes_range_ever,false) AS DIABETES_RANGE_EVER
      FROM users u LEFT JOIN screening_flags f USING(USER_ID)
      WHERE u.IS_INTERNAL_USER=false AND u.TZ_SOURCE='food_logs_mode'
        AND u.TZ_OFFSET_HOURS BETWEEN -12 AND 14
    """).df()
    reference = {"noninternal_known_timezone_users":len(users),
                 "accepted_screening_lab_users":int(users.HAS_SCREENING_LAB.sum()),
                 "unknown_screening_status_users":int((~users.HAS_SCREENING_LAB).sum()),
                 "observed_diabetes_range_users":int(users.DIABETES_RANGE_EVER.sum())}
    if cohort == "no_observed_diabetes_range":
        users = users.loc[~users.DIABETES_RANGE_EVER].copy()
    users.to_parquet(CACHE/"user_tz_offset.parquet",index=False)
    con.register("selected_users", users)
    print(time.strftime("%H:%M:%S"),f"Starting bounded CGM scan for {len(users):,} participants",flush=True)
    destination = CACHE/"gluc_local_day.parquet"
    temporary = CACHE/"gluc_local_day.building.parquet"
    con.execute(f"""
      COPY (
        WITH local AS (
          SELECT g.USER_ID,g.TS+u.TZ_OFFSET_HR*INTERVAL '1 hour' AS LTS,g.VALUE AS v
          FROM glucose_readings g JOIN selected_users u USING(USER_ID)
          WHERE g.VALUE BETWEEN 20 AND 500
        ), cells AS (
          SELECT *, (date_part('hour',LTS)*4+floor(date_part('minute',LTS)/15))::INTEGER AS cell
          FROM local
        ), daily AS (
          SELECT USER_ID,CAST(LTS AS DATE) LOCAL_DAY,count(*) N_READINGS,
            (bit_count(bit_or(CASE WHEN cell<64 THEN 1::UBIGINT << cell ELSE 0::UBIGINT END))
             +bit_count(bit_or(CASE WHEN cell>=64 THEN 1::UBIGINT << (cell-64) ELSE 0::UBIGINT END)))::INTEGER N_CELLS_15M,
            avg(v) MEAN,stddev_samp(v) SD,stddev_samp(v)/nullif(avg(v),0) CV,
            min(v) MINV,max(v) MAXV,
            100.0*avg((v BETWEEN 70 AND 140)::INTEGER) TIR_70_140,
            100.0*avg((v BETWEEN 70 AND 180)::INTEGER) TIR_70_180,
            100.0*avg((v>160)::INTEGER) PCT_ABOVE_160,
            avg(v) FILTER(WHERE date_part('hour',LTS)<5) OVERNIGHT_MEAN,
            avg(v) FILTER(WHERE date_part('hour',LTS) BETWEEN 5 AND 8) MORNING_MEAN,
            avg(v) FILTER(WHERE date_part('hour',LTS) BETWEEN 9 AND 17) DAYTIME_MEAN,
            avg(v) FILTER(WHERE date_part('hour',LTS)>=18) EVENING_MEAN
          FROM cells GROUP BY USER_ID,LOCAL_DAY
          HAVING N_READINGS>={MIN_READINGS} AND N_CELLS_15M>={MIN_CELLS}
        )
        SELECT *,year(LOCAL_DAY)::INTEGER AS YEAR,month(LOCAL_DAY)::INTEGER AS MONTH,
          (isodow(LOCAL_DAY)-1)::INTEGER AS DOW,isodow(LOCAL_DAY)>=6 AS IS_WEEKEND
        FROM daily
      ) TO '{temporary}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    temporary.replace(destination)
    counts=con.sql(f"""SELECT count(*) AS days,count(DISTINCT USER_ID) AS users,
        min(LOCAL_DAY)::VARCHAR first_day,max(LOCAL_DAY)::VARCHAR last_day,
        min(N_READINGS) min_readings,min(N_CELLS_15M) min_cells,
        sum(N_READINGS) accepted_readings
      FROM read_parquet('{destination}')""").df().iloc[0].to_dict()
    db=database_path().resolve()
    exposure_audit = build_exposure_cache(con)
    record={"day_definition_version":DAY_VERSION,"dense_only":True,"cohort":cohort,
      "density":{"raw_readings":MIN_READINGS,"observed_15m_cells":MIN_CELLS},
      "timezone":"users food_logs_mode, integer fixed UTC offset; no population fallback",
      "reference_participant_counts_before_CGM_gate":reference,
      "selected_participants_before_CGM_gate":len(users),"dense_day_counts":counts,
      "exposure_audit":exposure_audit,
      "lab_audit":lab_audit,"database":{"path":str(db),"size":db.stat().st_size,"mtime_ns":db.stat().st_mtime_ns},
      "cache_files":{p.name:{"size":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns}
                     for p in [destination,CACHE/"user_tz_offset.parquet",EXPOSURES]}}
    (CACHE/"local_day_metadata.json").write_text(json.dumps(record,indent=2,default=int)+"\n")
    (OUT/"local_day_provenance.json").write_text(json.dumps(record,indent=2,default=int)+"\n")
    con.close()
    print(time.strftime("%H:%M:%S"),json.dumps(counts,default=int),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort",choices=["no_observed_diabetes_range","all"],default="no_observed_diabetes_range")
    parser.add_argument("--stage",choices=["all","exposures"],default="all",
                        help="exposures reuses verified unchanged CGM/timezone caches")
    args = parser.parse_args()
    build(args.cohort, args.stage)
