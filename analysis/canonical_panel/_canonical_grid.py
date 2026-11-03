"""Helpers for building the canonical daily glucose grid.

Selects local days near an accepted HbA1c draw that have sufficient CGM
coverage, and computes a gap-bounded daily excursion amplitude (an
operational MAGE proxy: the mean successive turning-point
change above the daily sample SD, computed on unique accepted timestamps
within runs with no >10-minute gap, with strict extrema and run endpoints
retained). Used by the canonical lab/CGM panel builder.
"""

DENSE_DAY_VERSION='accepted-raw-local-day'
MAGE_VERSION='accepted-raw-gap-bounded-turning-amplitude'


def candidate_days(con):
    """All accepted local days near any accepted HbA1c; never trust day_panel counts."""
    return con.sql("""
      WITH ranges AS (
        SELECT USER_ID,min(DRAW_DATE)-INTERVAL '90 days' AS lo,
               max(DRAW_DATE)+INTERVAL '91 days' AS hi FROM a1c_draws GROUP BY USER_ID
      ), daily AS (
        SELECT g.USER_ID,CAST(g.TS+u.TZ_OFFSET_HOURS*INTERVAL '1 hour' AS DATE) AS DAY,
               u.TZ_OFFSET_HOURS AS tz,count(*) AS N_READINGS
        FROM glucose_readings g JOIN ranges r USING(USER_ID) JOIN users u USING(USER_ID)
        WHERE u.IS_INTERNAL_USER=false AND u.TZ_OFFSET_HOURS BETWEEN -12 AND 14
          AND g.VALUE BETWEEN 20 AND 500
          AND g.TS>=r.lo-u.TZ_OFFSET_HOURS*INTERVAL '1 hour'
          AND g.TS<r.hi-u.TZ_OFFSET_HOURS*INTERVAL '1 hour'
        GROUP BY g.USER_ID,DAY,u.TZ_OFFSET_HOURS HAVING count(*)>=230
      ) SELECT d.* FROM daily d WHERE EXISTS (
        SELECT 1 FROM a1c_draws a WHERE a.USER_ID=d.USER_ID
          AND abs(date_diff('day',d.DAY,a.DRAW_DATE))<=90)
      ORDER BY USER_ID,DAY
    """).df()


def grid_batch(con,first,last):
    """One bounded raw scan; use accepted readings for cells and amplitude proxy.

    Amplitude is the mean successive turning-point change above the daily
    sample SD, on unique accepted timestamps and within runs with no >10-minute
    gap. Strict extrema and run endpoints are retained. This is an operational MAGE proxy, not a clinical
    standard or an interpolation across non-wear.
    """
    con.execute("""CREATE OR REPLACE TEMP TABLE canonical_grid_raw AS
      SELECT c.USER_ID,c.DAY,g.TS,g.VALUE,
        (extract(hour FROM g.TS+c.tz*INTERVAL '1 hour')*4+
         extract(minute FROM g.TS+c.tz*INTERVAL '1 hour')//15)::INTEGER AS cell
      FROM glucose_readings g JOIN candidate_days c ON g.USER_ID=c.USER_ID
        AND g.TS>=CAST(c.DAY AS TIMESTAMP)-c.tz*INTERVAL '1 hour'
        AND g.TS<CAST(c.DAY AS TIMESTAMP)+INTERVAL '1 day'-c.tz*INTERVAL '1 hour'
      WHERE g.USER_ID>=? AND g.USER_ID<=? AND g.VALUE BETWEEN 20 AND 500
    """,[str(first),str(last)])
    cells=con.sql('SELECT USER_ID,DAY,cell,avg(VALUE) AS glucose FROM canonical_grid_raw GROUP BY USER_ID,DAY,cell').df()
    amplitude=con.sql("""
      WITH unique_values AS (
        SELECT USER_ID,DAY,TS,avg(VALUE) v FROM canonical_grid_raw GROUP BY USER_ID,DAY,TS
      ), previous AS (
        SELECT *,lag(TS) OVER(PARTITION BY USER_ID,DAY ORDER BY TS) prior_ts,
          stddev_samp(v) OVER(PARTITION BY USER_ID,DAY) daily_sd FROM unique_values
      ), runs AS (
        SELECT *,sum(CASE WHEN prior_ts IS NULL OR TS-prior_ts>INTERVAL '10 minutes' THEN 1 ELSE 0 END)
          OVER(PARTITION BY USER_ID,DAY ORDER BY TS) run FROM previous
      ), neighbors AS (
        SELECT *,lag(v) OVER(PARTITION BY USER_ID,DAY,run ORDER BY TS) prev_v,
          lead(v) OVER(PARTITION BY USER_ID,DAY,run ORDER BY TS) next_v FROM runs
      ), turns AS (
        SELECT * FROM neighbors WHERE prev_v IS NULL OR next_v IS NULL
          OR (v>prev_v AND v>next_v) OR (v<prev_v AND v<next_v)
      ), amplitudes AS (
        SELECT *,abs(v-lag(v) OVER(PARTITION BY USER_ID,DAY,run ORDER BY TS)) amplitude FROM turns
      ) SELECT USER_ID,DAY,avg(amplitude) FILTER(WHERE amplitude>daily_sd) AS daily_mage
        FROM amplitudes GROUP BY USER_ID,DAY
    """).df()
    con.execute('DROP TABLE canonical_grid_raw')
    return cells,amplitude
