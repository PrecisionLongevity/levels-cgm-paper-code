"""Behavior references (sleep, exercise, weight, steps) on fixed participant-local dates.

Source-event presence for a modality is tracked separately from whether a dated,
participant-local reference value could be computed for it: events remain eligible
for modality presence even when missing timestamps or offsets prevent a dated
reference. The whole-record CGM reference implementation lives separately in
analysis/cohort/_cohort_reference.py and is not duplicated here.
"""
from cgmpaper.local_exposure_dates import event_day_sql, sleep_midpoint_sql

VERSION = 'cohort-behavior-fixed-local-dates'


def behavior_reference(con):
    """Recorded event coverage plus validated participant-level summaries."""
    day = event_day_sql('e', 'u.TZ_OFFSET_HOURS')
    midpoint = sleep_midpoint_sql('e', 'u.TZ_OFFSET_HOURS')
    con.execute(f"""CREATE OR REPLACE TEMP TABLE reference_events AS
      SELECT DISTINCT e.USER_ID,e.TYPE,cast(e.EVENT_DATE AS DATE) AS SOURCE_DAY,
        CASE WHEN u.TZ_OFFSET_HOURS BETWEEN -12 AND 14 THEN {day} END AS DAY,
        CASE WHEN u.TZ_OFFSET_HOURS BETWEEN -12 AND 14 THEN {midpoint} END AS local_midpoint,
        e.START_TS,e.END_TS,
        e.DURATION_MIN,e.DOSE_NUMERIC,e.DOSE_UNIT,e.SOURCE_APP
      FROM lifestyle_events e JOIN users u USING(USER_ID)
      WHERE u.IS_INTERNAL_USER=FALSE AND e.TYPE IN ('exercise','sleep') """)
    frame=con.execute("""WITH sleep_daily AS (
      SELECT USER_ID,DAY,max(DURATION_MIN)/60. AS hours FROM reference_events
      WHERE TYPE='sleep' AND DAY IS NOT NULL AND DURATION_MIN BETWEEN 120 AND 840 GROUP BY USER_ID,DAY
    ), sleep AS (
      SELECT USER_ID,count(*) AS valid_sleep_days,
        CASE WHEN count(*)>=3 THEN avg(hours ORDER BY DAY) END AS avg_sleep_hours FROM sleep_daily GROUP BY USER_ID
    ), activity AS (
      SELECT USER_ID,count(*)FILTER(WHERE TYPE='sleep') AS sleep_log_count,
        count(*)FILTER(WHERE TYPE='exercise') AS exercise_log_count,
        count(*)FILTER(WHERE TYPE='exercise' AND DAY IS NOT NULL) AS dated_exercise_log_count,
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
        CASE WHEN count(value)>=3 THEN avg(value ORDER BY DAY) END AS mean_daily_steps FROM steps_daily GROUP BY USER_ID
    ) SELECT u.USER_ID,coalesce(a.sleep_log_count,0) AS sleep_log_count,
        coalesce(a.exercise_log_count,0) AS exercise_log_count,a.exercise_span,
        a.dated_exercise_log_count,
        CASE WHEN a.dated_exercise_log_count>=2 AND a.exercise_span>=28
          THEN a.dated_exercise_log_count*7./a.exercise_span END AS exercise_sessions_per_week,
        sl.avg_sleep_hours,sl.valid_sleep_days,w.first_weight_lb,s.mean_daily_steps,s.valid_step_days
      FROM users u LEFT JOIN activity a USING(USER_ID) LEFT JOIN sleep sl USING(USER_ID)
      LEFT JOIN first_weight w USING(USER_ID) LEFT JOIN steps_average s USING(USER_ID)
      WHERE u.IS_INTERNAL_USER=FALSE ORDER BY u.USER_ID""").df()
    con.execute('DROP TABLE reference_events')
    return frame
