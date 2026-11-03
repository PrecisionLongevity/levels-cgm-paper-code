"""Helper for deriving local-date aggregates of lifestyle events (sleep, exercise, alcohol).

CGM and events use the same participant UTC offset. Sleep is assigned to its local
wake date (END_TS), not the start date. Sleep midpoint is computed from the
start/end timestamps.
"""
EXPOSURE_VERSION = "fixed-local-event-exposures"


def event_day_sql(event_alias, offset_sql):
    """Trusted SQL expressions: local wake date for sleep, start date otherwise."""
    return (f"CAST((CASE WHEN {event_alias}.TYPE='sleep' THEN {event_alias}.END_TS "
            f"ELSE {event_alias}.START_TS END) + ({offset_sql})*INTERVAL '1 hour' AS DATE)")


def sleep_midpoint_sql(event_alias, offset_sql):
    """Local [0,24) midpoint hour; missing/reversed timestamps yield NULL."""
    a = event_alias
    midpoint = f"({a}.START_TS+({a}.END_TS-{a}.START_TS)/2+({offset_sql})*INTERVAL '1 hour')"
    return (f"CASE WHEN {a}.START_TS IS NOT NULL AND {a}.END_TS IS NOT NULL "
            f"AND {a}.END_TS>={a}.START_TS THEN "
            f"date_part('hour',{midpoint})+date_part('minute',{midpoint})/60.0"
            f"+date_part('second',{midpoint})/3600.0 END")


def build_event_daily(con, eligible_relation):
    """Create temporary publication_event_daily and return aggregate diagnostics.

    eligible_relation must have unique USER_ID and TZ_OFFSET_HR. All dates are
    reconstructed before selecting CGM dates, so midnight crossings can enter or
    leave dense days. Exercise/alcohol retain every stored classified event.
    Sleep is the maximum accepted 60–900-minute duration per local wake date.
    Tied longest events need an identical nonmissing midpoint to define timing.
    """
    day = event_day_sql('e', 'u.TZ_OFFSET_HR')
    midpoint = sleep_midpoint_sql('e', 'u.TZ_OFFSET_HR')
    con.execute(f"""CREATE OR REPLACE TEMP VIEW publication_local_events AS
      SELECT e.USER_ID,e.TYPE,e.DURATION_MIN,e.METS,e.EX_CATEGORY,
        CAST(e.EVENT_DATE AS DATE) stored_day,{day} AS DAY,{midpoint} midpoint,
        e.START_TS,e.END_TS
      FROM lifestyle_events e JOIN {eligible_relation} u USING(USER_ID)
      WHERE e.TYPE IN ('sleep','exercise','alcohol')""")
    audit = con.sql("""SELECT TYPE,count(*) source_events,
        count(*) FILTER(WHERE DAY IS NULL) excluded_missing_required_timestamp,
        count(*) FILTER(WHERE DAY IS NOT NULL AND stored_day IS NULL) missing_stored_date,
        count(*) FILTER(WHERE DAY<>stored_day) moved_from_stored_date,
        count(*) FILTER(WHERE TYPE='sleep' AND DAY IS NOT NULL
          AND (DURATION_MIN IS NULL OR DURATION_MIN NOT BETWEEN 60 AND 900)) excluded_sleep_duration,
        count(*) FILTER(WHERE TYPE='sleep' AND DAY IS NOT NULL
          AND DURATION_MIN BETWEEN 60 AND 900 AND midpoint IS NULL) accepted_sleep_missing_midpoint
      FROM publication_local_events GROUP BY TYPE ORDER BY TYPE""").df().to_dict('records')
    con.execute("""CREATE OR REPLACE TEMP TABLE publication_sleep_daily AS
      WITH accepted AS (
        SELECT *,max(DURATION_MIN) OVER(PARTITION BY USER_ID,DAY) longest
        FROM publication_local_events WHERE TYPE='sleep' AND DAY IS NOT NULL
          AND DURATION_MIN BETWEEN 60 AND 900
      ), collapsed AS (
        SELECT USER_ID,DAY,max(longest) SLEEP_DURATION_MIN,count(*) SLEEP_LOG_COUNT,
          CASE WHEN count(*) FILTER(WHERE DURATION_MIN=longest)=
                    count(midpoint) FILTER(WHERE DURATION_MIN=longest)
                AND min(midpoint) FILTER(WHERE DURATION_MIN=longest)=
                    max(midpoint) FILTER(WHERE DURATION_MIN=longest)
          THEN max(midpoint) FILTER(WHERE DURATION_MIN=longest) END SLEEP_MIDPOINT_HR
        FROM accepted GROUP BY USER_ID,DAY
      ), centered AS (
        SELECT *,CASE WHEN SLEEP_MIDPOINT_HR>12 THEN SLEEP_MIDPOINT_HR-24
          ELSE SLEEP_MIDPOINT_HR END midpoint_unwrapped FROM collapsed
      )
      SELECT USER_ID,DAY,SLEEP_DURATION_MIN,SLEEP_LOG_COUNT,SLEEP_MIDPOINT_HR,
        CASE WHEN count(midpoint_unwrapped) OVER w>=4
          THEN stddev_samp(midpoint_unwrapped) OVER w END SLEEP_REGULARITY_7D
      FROM centered WINDOW w AS (PARTITION BY USER_ID ORDER BY DAY
        RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW)""")
    categories = ['STRENGTH', 'VIGOROUS_CARDIO', 'WALKING', 'MODERATE_CARDIO']
    category_fields = ','.join(f"count(*) FILTER(WHERE TYPE='exercise' AND EX_CATEGORY='{name.lower()}') EX_SESSIONS_{name}" for name in categories)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE publication_event_daily AS
      WITH daily AS (
        SELECT USER_ID,DAY,
          count(*) FILTER(WHERE TYPE='alcohol') ALC_LOG_COUNT,
          count(*) FILTER(WHERE TYPE='exercise') EX_LOG_COUNT,
          sum(DURATION_MIN) FILTER(WHERE TYPE='exercise') EX_TOTAL_DURATION_MIN,
          sum(METS*DURATION_MIN) FILTER(WHERE TYPE='exercise' AND METS IS NOT NULL AND DURATION_MIN>0)
          /nullif(sum(DURATION_MIN) FILTER(WHERE TYPE='exercise' AND METS IS NOT NULL AND DURATION_MIN>0),0) EX_AVG_METS,
          {category_fields}
        FROM publication_local_events WHERE DAY IS NOT NULL GROUP BY USER_ID,DAY
      )
      SELECT d.*,ALC_LOG_COUNT>0 ALC_ANY,EX_LOG_COUNT>0 EX_ANY,
        s.SLEEP_DURATION_MIN,s.SLEEP_LOG_COUNT,s.SLEEP_MIDPOINT_HR,s.SLEEP_REGULARITY_7D
      FROM daily d LEFT JOIN publication_sleep_daily s USING(USER_ID,DAY)""")
    ties = con.sql("""WITH longest AS (
        SELECT USER_ID,DAY,max(DURATION_MIN) longest FROM publication_local_events
        WHERE TYPE='sleep' AND DAY IS NOT NULL AND DURATION_MIN BETWEEN 60 AND 900 GROUP BY USER_ID,DAY
      ), selected AS (
        SELECT e.USER_ID,e.DAY,count(*) n,count(e.midpoint) n_mid,min(e.midpoint) lo,max(e.midpoint) hi
        FROM publication_local_events e JOIN longest l USING(USER_ID,DAY)
        WHERE e.TYPE='sleep' AND e.DURATION_MIN=l.longest GROUP BY e.USER_ID,e.DAY
      ) SELECT count(*) FILTER(WHERE n>1) longest_tie_dates,
        count(*) FILTER(WHERE n>1 AND (n_mid<n OR lo<>hi)) longest_tie_dates_missing_midpoint
      FROM selected""").df().iloc[0].to_dict()
    return {'event_counts_by_type':audit,'sleep_ties':{k:int(v) for k,v in ties.items()},
            'date_rule':'sleep END_TS; exercise/alcohol START_TS; fixed participant UTC offset; missing required timestamps excluded',
            'sleep_rule':'longest 60–900-minute event per local wake date; tied longest midpoints must agree and all be nonmissing',
            'regularity_rule':'sample SD of local midpoint unwrapped about midnight (>12 subtract24); current and preceding6 calendar dates; >=4 valid nights; all sleep dates before CGM-day selection'}
