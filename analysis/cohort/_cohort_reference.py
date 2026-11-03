"""Whole-record CGM and recorded-behavior reference definitions.

Computes, per participant, consented monitoring duration from consent periods, CGM-derived
summary statistics (mean/SD/CV glucose, median glucose, time-in-range bands, GMI, MAGE) over
accepted raw readings (20-500 mg/dL) aggregated to 15-minute cells, and recorded-behavior
summaries (sleep hours, exercise session frequency, step counts, first recorded weight) from
lifestyle-event, weight, and step logs for non-internal users.
"""

VERSION='cohort-reference-observed-cells'


def consent_durations(con,path):
    periods=con.execute("""SELECT USER_ID,
      try_cast(replace(VALID_FROM,' Z','') AS TIMESTAMP) AS start_time,
      try_cast(replace(VALID_TO,' Z','') AS TIMESTAMP) AS end_time
      FROM read_csv_auto(?,all_varchar=true)""",[str(path)]).df()
    if periods.isna().any().any() or (periods.end_time<periods.start_time).any():
        raise ValueError('Consent periods have missing or reversed endpoints')
    con.register('consent_periods',periods)
    return con.execute("""WITH ordered AS (
      SELECT *,max(end_time)OVER(PARTITION BY USER_ID ORDER BY start_time,end_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_end FROM consent_periods
    ) SELECT USER_ID,count(*) AS num_periods,
      sum(CASE WHEN prior_end IS NULL OR start_time>=prior_end THEN epoch(end_time)-epoch(start_time)
        WHEN end_time>prior_end THEN epoch(end_time)-epoch(prior_end) ELSE 0 END)/86400. AS total_consented_days,
      count(*)FILTER(WHERE start_time<prior_end) AS overlapping_periods
      FROM ordered GROUP BY USER_ID""").df()


def cgm_batch(con,roster):
    """Bounded scan; roster contains USER_ID and valid fixed TZ_OFFSET_HOURS."""
    con.register('cohort_batch',roster)
    con.execute("""CREATE OR REPLACE TEMP TABLE reference_raw AS
      SELECT g.USER_ID,g.TS,g.VALUE,u.TZ_OFFSET_HOURS,
        cast(g.TS+u.TZ_OFFSET_HOURS*INTERVAL '1 hour' AS DATE) AS DAY,
        time_bucket(INTERVAL '15 minutes',g.TS+u.TZ_OFFSET_HOURS*INTERVAL '1 hour') AS cell
      FROM glucose_readings g JOIN cohort_batch u USING(USER_ID)
      WHERE g.USER_ID>=? AND g.USER_ID<=? AND g.VALUE BETWEEN 20 AND 500 AND g.TS IS NOT NULL
    """,[str(roster.USER_ID.min()),str(roster.USER_ID.max())])
    features=con.execute("""WITH cells AS (
      SELECT USER_ID,DAY,cell,avg(VALUE) AS value,count(*) AS n_raw
      FROM reference_raw GROUP BY USER_ID,DAY,cell
    ) SELECT USER_ID,count(*) AS observed_cells,sum(n_raw) AS accepted_raw_readings,
        count(DISTINCT DAY) AS sensor_days,avg(value) AS mean_glucose,
        stddev_samp(value) AS sd_glucose,median(value) AS median_glucose,
        100.*avg(CASE WHEN value BETWEEN 70 AND 180 THEN 1 ELSE 0 END) AS tir_70_180,
        100.*avg(CASE WHEN value BETWEEN 70 AND 140 THEN 1 ELSE 0 END) AS tir_70_140,
        100.*avg(CASE WHEN value>140 THEN 1 ELSE 0 END) AS tar_140,
        100.*avg(CASE WHEN value>180 THEN 1 ELSE 0 END) AS tar_180,
        100.*avg(CASE WHEN value<70 THEN 1 ELSE 0 END) AS tbr_70,
        avg(value)FILTER(WHERE extract(hour FROM cell) BETWEEN 4 AND 7) AS mean_fasting_glucose
      FROM cells GROUP BY USER_ID""").df()
    amplitude=con.execute("""WITH unique_values AS (
      SELECT USER_ID,DAY,TS,avg(VALUE) AS v FROM reference_raw GROUP BY USER_ID,DAY,TS
    ), previous AS (
      SELECT *,lag(TS)OVER(PARTITION BY USER_ID,DAY ORDER BY TS) AS prior_ts,
        stddev_samp(v)OVER(PARTITION BY USER_ID,DAY) AS daily_sd FROM unique_values
    ), runs AS (
      SELECT *,sum(CASE WHEN prior_ts IS NULL OR TS-prior_ts>INTERVAL '10 minutes' THEN 1 ELSE 0 END)
        OVER(PARTITION BY USER_ID,DAY ORDER BY TS) AS run FROM previous
    ), neighbors AS (
      SELECT *,lag(v)OVER(PARTITION BY USER_ID,DAY,run ORDER BY TS) AS prev_v,
        lead(v)OVER(PARTITION BY USER_ID,DAY,run ORDER BY TS) AS next_v FROM runs
    ), turns AS (
      SELECT * FROM neighbors WHERE prev_v IS NULL OR next_v IS NULL OR
        (v>prev_v AND v>next_v) OR (v<prev_v AND v<next_v)
    ), amplitudes AS (
      SELECT *,abs(v-lag(v)OVER(PARTITION BY USER_ID,DAY,run ORDER BY TS)) AS amplitude FROM turns
    ), daily AS (
      SELECT USER_ID,DAY,avg(amplitude)FILTER(WHERE amplitude>daily_sd) AS daily_mage
      FROM amplitudes GROUP BY USER_ID,DAY
    ) SELECT USER_ID,avg(daily_mage) AS mage,count(daily_mage) AS mage_days
      FROM daily GROUP BY USER_ID""").df()
    con.execute('DROP TABLE reference_raw')
    result=features.merge(amplitude,on='USER_ID',validate='one_to_one')
    result['cv_glucose']=100*result.sd_glucose/result.mean_glucose
    result['gmi']=3.31+.02392*result.mean_glucose
    result['observed_equivalent_days']=result.observed_cells/96.
    return result


def behavior_reference(con):
    """Recorded event coverage plus validated participant-level summaries."""
    con.execute("""CREATE OR REPLACE TEMP TABLE reference_events AS
      SELECT DISTINCT e.USER_ID,e.TYPE,cast(e.EVENT_DATE AS DATE) AS DAY,e.START_TS,e.END_TS,
        e.DURATION_MIN,e.DOSE_NUMERIC,e.DOSE_UNIT,e.SOURCE_APP
      FROM lifestyle_events e JOIN users u USING(USER_ID)
      WHERE u.IS_INTERNAL_USER=FALSE AND e.TYPE IN ('exercise','sleep') AND e.EVENT_DATE IS NOT NULL""")
    frame=con.execute("""WITH sleep_daily AS (
      SELECT USER_ID,DAY,max(DURATION_MIN)/60. AS hours FROM reference_events
      WHERE TYPE='sleep' AND DURATION_MIN BETWEEN 120 AND 840 GROUP BY USER_ID,DAY
    ), sleep AS (
      SELECT USER_ID,count(*) AS valid_sleep_days,
        CASE WHEN count(*)>=3 THEN avg(hours) END AS avg_sleep_hours FROM sleep_daily GROUP BY USER_ID
    ), activity AS (
      SELECT USER_ID,count(*)FILTER(WHERE TYPE='sleep') AS sleep_log_count,
        count(*)FILTER(WHERE TYPE='exercise') AS exercise_log_count,
        date_diff('day',min(DAY)FILTER(WHERE TYPE='exercise'),max(DAY)FILTER(WHERE TYPE='exercise'))+1 AS exercise_span
      FROM reference_events GROUP BY USER_ID
    ), weights AS (
      SELECT w.USER_ID,cast(w.DAY AS DATE) AS DAY,median(w.WEIGHT) AS pounds
      FROM weight w JOIN users u USING(USER_ID)
      WHERE u.IS_INTERNAL_USER=FALSE AND w.WEIGHT BETWEEN 80 AND 500 AND w.DAY IS NOT NULL
      GROUP BY w.USER_ID,cast(w.DAY AS DATE)
    ), first_weight AS (
      SELECT USER_ID,arg_min(pounds,DAY) AS first_weight_lb FROM weights GROUP BY USER_ID
    ), steps_daily AS (
      SELECT s.USER_ID,s.DAY,
        CASE WHEN count(TOTAL_STEPS)=count(*) AND min(TOTAL_STEPS)>=0 AND isfinite(max(TOTAL_STEPS))
          AND min(TOTAL_STEPS)=max(TOTAL_STEPS) THEN max(TOTAL_STEPS) END AS value
      FROM steps s JOIN users u USING(USER_ID) WHERE u.IS_INTERNAL_USER=FALSE AND s.DAY IS NOT NULL
      GROUP BY s.USER_ID,s.DAY
    ), steps_average AS (
      SELECT USER_ID,count(value) AS valid_step_days,
        CASE WHEN count(value)>=3 THEN avg(value) END AS mean_daily_steps FROM steps_daily GROUP BY USER_ID
    ) SELECT u.USER_ID,coalesce(a.sleep_log_count,0) AS sleep_log_count,
        coalesce(a.exercise_log_count,0) AS exercise_log_count,a.exercise_span,
        CASE WHEN a.exercise_log_count>=2 AND a.exercise_span>=28
          THEN a.exercise_log_count*7./a.exercise_span END AS exercise_sessions_per_week,
        sl.avg_sleep_hours,sl.valid_sleep_days,w.first_weight_lb,s.mean_daily_steps,s.valid_step_days
      FROM users u LEFT JOIN activity a USING(USER_ID) LEFT JOIN sleep sl USING(USER_ID)
      LEFT JOIN first_weight w USING(USER_ID) LEFT JOIN steps_average s USING(USER_ID)
      WHERE u.IS_INTERNAL_USER=FALSE ORDER BY u.USER_ID""").df()
    con.execute('DROP TABLE reference_events')
    return frame
