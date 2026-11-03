#!/usr/bin/env python3
"""Aggregate-only distributions of laboratory, body-size and behavior streams.

Computes population distributions of biomarkers, BMI, step count, sleep and exercise per week, and derives
revised step and exercise definitions on top of the shared cohort reference: average daily steps bounded to a
plausible range (unbounded daily totals can include corrupt readings up to ~1e21), and exercise *days* per week
rather than exercise *sessions* per week. Writes ONLY binned histogram counts and summary statistics (n, mean,
SD, median, IQR); no participant identifier or participant-level row is written.

Populations
  Laboratory / BMI   primary canonical lab-CGM panel (first HbA1c with >= 14 dense CGM days within +/-90 d, dense
                     day = >= 230 accepted readings and >= 77 observed 15-minute cells, participant-local; no
                     observed diabetes-range laboratory value) (cgmpaper/publication_panels.py load_panel).
                     Analytes are those of the canonical draw (HbA1c anchor, others within +/-7 d, BMI
                     as recorded by the panel builder). HbA1c rounded half-up to 0.1%.
  Behavior           all consenting noninternal participants with enough dated records:
    steps            participant-local date-only daily totals (agreement rule: all rows of a day agree,
                     finite, >= 0) AND 100 <= steps <= 60,000 (plausibility bound); average over >= 3 such days.
    sleep            avg_sleep_hours: longest 120-840-min sleep per participant-local wake date
                     (fixed participant offset in [-12, 14] h), average over >= 3 nights.
    exercise days    distinct participant-local exercise dates x 7 / exercise_span (first to last dated
                     exercise date + 1), with eligibility (>= 2 dated exercise logs, span >= 28 days).
                     Days, not sessions; bounded 0-7.

Inputs
  Database (LEVELS_DB): lifestyle_events, steps, users tables.
  OUTPUTS/cohort/reference_analysis.metadata.json, reference_values.parquet, cohort_participants.parquet,
    cohort_summary.json (analysis/cohort/cohort_reference.py).
  OUTPUTS/cohort/stream_overlap_regions/summary.json (analysis/cohort/stream_overlap.py).
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py).

Outputs (aggregate only, OUTPUTS/cohort/multimodal_distributions/)
  distribution_summary.csv, distribution_histograms.csv, summary.json

Usage
  python analysis/cohort/multimodal_distributions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from analysis.cohort import _behavior_reference as behavior_reference# noqa: E402
from cgmpaper.local_exposure_dates import event_day_sql  # noqa: E402
from cgmpaper.publication_panels import load_panel  # noqa: E402

VERSION = 'multimodal-distributions'
PUB = OUTPUTS
CC = PUB / 'cohort'
REF_META = CC / 'reference_analysis.metadata.json'
S_STREAM_OVERLAP = CC / 'stream_overlap_regions'
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
DB = database_path()
OUT = CC / 'multimodal_distributions'
REQUIRED_REVISION = 'cohort-reference-observed-cells'
STEP_BOUNDS = (100, 60000)
MIN_STEP_DAYS = 3  # cohort reference rule
BEHAVIOR_COLS = ['sleep_log_count', 'exercise_log_count', 'exercise_span', 'dated_exercise_log_count',
                 'exercise_sessions_per_week', 'avg_sleep_hours', 'valid_sleep_days', 'first_weight_lb',
                 'mean_daily_steps', 'valid_step_days']

# measure key -> (label, unit, population, display lo, display hi, bin width); fixed a priori, not data-derived
MEASURES = {
    'a1c': ('HbA1c', '%', 'primary', 3.95, 6.45, 0.1),
    'fasting_insulin': ('Fasting insulin', 'µIU/mL', 'primary', 0.0, 40.0, 1.0),
    'triglycerides': ('Triglycerides', 'mg/dL', 'primary', 0.0, 400.0, 10.0),
    'bmi': ('BMI', 'kg/m²', 'primary', 15.0, 50.0, 1.0),
    'daily_steps': ('Average daily steps', 'steps/day', 'consenting', 0.0, 25000.0, 500.0),
    'sleep_hours': ('Sleep per night', 'h', 'consenting', 4.0, 11.0, 0.25),
    'exercise_days_per_week': ('Exercise days per week', 'days/week', 'consenting', 0.0, 7.0, 0.25),
}
POPULATIONS = {
    'primary': 'primary canonical lab-CGM panel (no observed diabetes-range laboratory value), canonical draw',
    'consenting': 'all consenting noninternal participants with enough dated records of the stream',
}


def fail(msg):
    raise SystemExit('GUARD FAILED: ' + msg)


def same(a, b):
    """Exact per-participant equality with NaN == NaN (floats compared exactly)."""
    a = pd.to_numeric(a, errors='coerce').to_numpy(float); b = pd.to_numeric(b, errors='coerce').to_numpy(float)
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(((a == b) | both_nan).all())


def extended_query(con):
    """Script 12 reference_events rebuilt verbatim, plus the two revised definitions."""
    day = event_day_sql('e', 'u.TZ_OFFSET_HOURS')
    con.execute(f"""CREATE OR REPLACE TEMP TABLE s_multimodal_events AS
      SELECT DISTINCT e.USER_ID,e.TYPE,cast(e.EVENT_DATE AS DATE) AS SOURCE_DAY,
        CASE WHEN u.TZ_OFFSET_HOURS BETWEEN -12 AND 14 THEN {day} END AS DAY,
        e.START_TS,e.END_TS,e.DURATION_MIN,e.DOSE_NUMERIC,e.DOSE_UNIT,e.SOURCE_APP
      FROM lifestyle_events e JOIN users u USING(USER_ID)
      WHERE u.IS_INTERNAL_USER=FALSE AND e.TYPE IN ('exercise','sleep') """)
    lo, hi = STEP_BOUNDS
    frame = con.execute(f"""WITH ex AS (
      SELECT USER_ID,count(*) FILTER(WHERE TYPE='exercise') AS exercise_events,
        count(*) FILTER(WHERE TYPE='sleep') AS sleep_events,
        count(*) FILTER(WHERE TYPE='exercise' AND DAY IS NOT NULL) AS dated_exercise_log_count,
        count(DISTINCT DAY) FILTER(WHERE TYPE='exercise' AND DAY IS NOT NULL) AS exercise_days,
        date_diff('day',min(DAY)FILTER(WHERE TYPE='exercise'),max(DAY)FILTER(WHERE TYPE='exercise'))+1 AS exercise_span
      FROM s_multimodal_events GROUP BY USER_ID
    ), steps_daily AS (
      SELECT s.USER_ID,s.DAY,
        CASE WHEN count(TOTAL_STEPS)=count(*) AND min(TOTAL_STEPS)>=0 AND isfinite(max(TOTAL_STEPS))
          AND min(TOTAL_STEPS)=max(TOTAL_STEPS) THEN max(TOTAL_STEPS) END AS value
      FROM steps s JOIN users u USING(USER_ID) WHERE u.IS_INTERNAL_USER=FALSE AND s.DAY IS NOT NULL
      GROUP BY s.USER_ID,s.DAY
    ), steps_avg AS (
      SELECT USER_ID,count(value) AS valid_step_days,
        count(value) FILTER(WHERE value BETWEEN {lo} AND {hi}) AS plausible_step_days,
        CASE WHEN count(value) FILTER(WHERE value BETWEEN {lo} AND {hi})>={MIN_STEP_DAYS}
          THEN avg(value ORDER BY DAY) FILTER(WHERE value BETWEEN {lo} AND {hi}) END AS daily_steps
      FROM steps_daily GROUP BY USER_ID
    ), step_rows AS (
      SELECT s.USER_ID,count(*) AS step_rows FROM steps s JOIN users u USING(USER_ID)
      WHERE u.IS_INTERNAL_USER=FALSE GROUP BY s.USER_ID
    ) SELECT u.USER_ID,coalesce(ex.exercise_events,0) AS exercise_events,coalesce(ex.sleep_events,0) AS sleep_events,
        ex.dated_exercise_log_count,ex.exercise_days,ex.exercise_span,
        CASE WHEN ex.dated_exercise_log_count>=2 AND ex.exercise_span>=28
          THEN ex.exercise_days*7./ex.exercise_span END AS exercise_days_per_week,
        sa.valid_step_days,sa.plausible_step_days,sa.daily_steps,coalesce(sr.step_rows,0) AS step_rows
      FROM users u LEFT JOIN ex USING(USER_ID) LEFT JOIN steps_avg sa USING(USER_ID) LEFT JOIN step_rows sr USING(USER_ID)
      WHERE u.IS_INTERNAL_USER=FALSE ORDER BY u.USER_ID""").df()
    con.execute('DROP TABLE s_multimodal_events')
    return frame


def summarize(key, values):
    label, unit, pop, lo, hi, width = MEASURES[key]
    v = pd.to_numeric(values, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        fail('no values for ' + key)
    edges = np.round(np.arange(lo, hi + width / 2, width), 6)
    vis = v.loc[v.between(lo, hi)]
    counts, _ = np.histogram(vis, bins=edges)
    below, above = int(v.lt(lo).sum()), int(v.gt(hi).sum())
    if int(counts.sum()) + below + above != len(v):
        fail('histogram does not partition ' + key)
    stats = dict(measure=key, label=label, unit=unit, population=pop, n=int(len(v)), mean=float(v.mean()),
                 sd=float(v.std(ddof=1)), median=float(v.median()), q25=float(v.quantile(.25)), q75=float(v.quantile(.75)),
                 display_lo=lo, display_hi=hi, bin_width=width, n_below_display=below, n_above_display=above)
    hist = pd.DataFrame(dict(measure=key, bin_lo=edges[:-1], bin_hi=edges[1:], count=counts.astype(int)))
    return stats, hist


def main():
    guards = {}
    meta = json.loads(REF_META.read_text())
    if meta.get('behavior_definition_version') != behavior_reference.VERSION:
        fail('behavior definition version')
    summary_cohort_reference = json.loads((CC / 'cohort_summary.json').read_text())
    ref = pd.read_parquet(CC / 'reference_values.parquet', columns=['USER_ID', *BEHAVIOR_COLS]).sort_values('USER_ID').reset_index(drop=True)
    roster = pd.read_parquet(CC / 'cohort_participants.parquet', columns=['USER_ID', 'log_rows', 'step_rows', 'sleep_log_count'])
    if len(ref) != summary_cohort_reference['total'] or ref.USER_ID.duplicated().any():
        fail('reference_values roster')
    with duckdb.connect(str(DB), read_only=True) as con:
        con.execute("SET threads=4; SET memory_limit='4GB'")
        b_cohort_reference = behavior_reference.behavior_reference(con).sort_values('USER_ID').reset_index(drop=True)
        X = extended_query(con).sort_values('USER_ID').reset_index(drop=True)
    # consistency: roster order and behavior rows match the shared cohort reference
    if not b_cohort_reference.USER_ID.equals(ref.USER_ID) or not X.USER_ID.equals(ref.USER_ID):
        fail('roster order differs from reference_values')
    diff = [c for c in BEHAVIOR_COLS if not same(b_cohort_reference[c], ref[c])]
    if diff:
        fail(f'behavior rows not reproduced: {diff}')
    guards['behavior_rows_reproduced'] = dict(participants=len(ref), columns=BEHAVIOR_COLS)
    # consistency: extended query intermediates match the shared cohort reference
    for mine, theirs in (('valid_step_days', 'valid_step_days'), ('dated_exercise_log_count', 'dated_exercise_log_count'),
                         ('exercise_span', 'exercise_span'), ('exercise_events', 'exercise_log_count'), ('sleep_events', 'sleep_log_count')):
        if not same(X[mine], b_cohort_reference[theirs]):
            fail(f'extended query does not reproduce {theirs}')
    ok = X.exercise_days_per_week.dropna()
    if not ok.between(0, 7).all() or not X.exercise_days_per_week.notna().eq(b_cohort_reference.exercise_sessions_per_week.notna()).all():
        fail('exercise days per week outside 0-7 or eligibility differs from sessions per week')
    guards['extended_query_reproduces_intermediates'] = True
    # per-stream participant counts equal cohort reference coverage
    cov = summary_cohort_reference['coverage']
    r = roster.set_index('USER_ID').loc[X.USER_ID]
    present = dict(steps_any=int(X.step_rows.gt(0).sum()), exercise_any=int(X.exercise_events.gt(0).sum()),
                   sleep_any=int(X.sleep_events.gt(0).sum()))
    if present != {k: cov[k] for k in present} or not X.step_rows.gt(0).to_numpy().__eq__(r.step_rows.gt(0).to_numpy()).all() \
            or not X.exercise_events.gt(0).to_numpy().__eq__(r.log_rows.gt(0).to_numpy()).all():
        fail(f'stream presence does not equal cohort reference coverage: {present}')
    s_stream_overlap = json.loads((S_STREAM_OVERLAP / 'summary.json').read_text())
    if {k: s_stream_overlap['stream_coverage'][k] for k in present} != present:
        fail('stream presence does not equal stream overlap coverage')
    subsets = dict(daily_steps=X.daily_steps.notna() & ~X.step_rows.gt(0), sleep_hours=b_cohort_reference.avg_sleep_hours.notna() & ~X.sleep_events.gt(0),
                   exercise_days_per_week=X.exercise_days_per_week.notna() & ~X.exercise_events.gt(0))
    if any(v.any() for v in subsets.values()):
        fail('a displayed denominator is not a subset of its stream')
    guards['stream_presence_equals_reference_coverage'] = present
    # primary canonical panel
    P = load_panel(PANEL, cohort='no_diabetes_range', required=['bmi', 'fasting_insulin', 'triglycerides'], expected_window='symmetric')
    if len(P) != s_stream_overlap['funnel']['primary_no_diabetes_range_labs'] or len(P) != summary_cohort_reference['primary_aligned']:
        fail('primary cohort n differs from stream overlap funnel / cohort reference')
    guards['primary_panel'] = dict(n=len(P))
    # distributions
    src = dict(a1c=P.a1c, fasting_insulin=P.fasting_insulin, triglycerides=P.triglycerides, bmi=P.bmi,
               daily_steps=X.daily_steps, sleep_hours=b_cohort_reference.avg_sleep_hours, exercise_days_per_week=X.exercise_days_per_week)
    stats, hists = [], []
    for k in MEASURES:
        s, h = summarize(k, src[k]); stats.append(s); hists.append(h)
    guards['histograms_partition_n'] = True
    S = pd.DataFrame(stats); H = pd.concat(hists, ignore_index=True)
    # descriptive context (aggregates only)
    context = dict(
        steps=dict(participants_with_step_rows=present['steps_any'], with_script_cohort_reference_average=int(b_cohort_reference.mean_daily_steps.notna().sum()),
                   with_bounded_average=int(X.daily_steps.notna().sum()), bounds=list(STEP_BOUNDS), min_days=MIN_STEP_DAYS,
                   script_cohort_reference_unbounded_mean_of_participant_averages=float(b_cohort_reference.mean_daily_steps.mean()),
                   script_cohort_reference_unbounded_median_of_participant_averages=float(b_cohort_reference.mean_daily_steps.median())),
        exercise=dict(participants_with_exercise_events=present['exercise_any'], eligible=int(X.exercise_days_per_week.notna().sum()),
                      script_cohort_reference_sessions_per_week_median=float(b_cohort_reference.exercise_sessions_per_week.median())),
        sleep=dict(participants_with_sleep_events=present['sleep_any'], eligible=int(b_cohort_reference.avg_sleep_hours.notna().sum())),
        primary_panel_n=len(P))
    OUT.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT / 'distribution_summary.csv', index=False)
    H.to_csv(OUT / 'distribution_histograms.csv', index=False)
    summ = dict(version=VERSION, populations=POPULATIONS, measures={s['measure']: s for s in stats}, context=context,
                definitions=dict(daily_steps=f'average of participant-local daily totals, cohort-reference agreement rule and {STEP_BOUNDS[0]}-{STEP_BOUNDS[1]} steps, >= {MIN_STEP_DAYS} days',
                                 sleep_hours='avg_sleep_hours: longest 120-840-min sleep per local wake date, >= 3 nights',
                                 exercise_days_per_week='distinct local exercise dates x 7 / (first to last dated exercise date + 1); >= 2 dated logs, span >= 28 d',
                                 labs='canonical draw of the primary panel; HbA1c rounded half-up to 0.1%'))
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1) + '\n')
    md = dict(analysis='multimodal_distributions', version=VERSION,
              design='aggregate binned counts and summary statistics only; descriptive', guards=guards,
              participant_identifiers_written=False)
    (OUT / 'multimodal_distributions.metadata.json').write_text(json.dumps(md, indent=1) + '\n')
    print(S[['measure', 'population', 'n', 'mean', 'median', 'q25', 'q75', 'n_below_display', 'n_above_display']].to_string(index=False))
    print(json.dumps(context, indent=1))


if __name__ == '__main__':
    main()
