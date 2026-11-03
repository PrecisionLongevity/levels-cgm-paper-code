#!/usr/bin/env python3
"""Operational spike and glycemic-exposure sensitivity analysis.

This is not an exact replication of the PROGRESS (Carletti et al.) spike analysis; it is a
separate, operational sensitivity check using this cohort's observed-cell glucose grid and
logged-meal/step data. Spike and exposure metrics (mean glucose, nocturnal minimum, percent
time above 150 mg/dL, max relative spike, daily spike count, spike resolution, and
mean+SD-based peak frequency) are correlated (Spearman, simple and age/sex-partial) against
HbA1c, BMI, age bucket, daily logged carbohydrates, and daily recorded steps, separately for
the primary cohort, high loggers, and participants with a known food-log-derived timezone.
Carbohydrate exposure is the mean of daily sums over complete logged days only; unlogged or
incomplete days are left missing rather than zero-filled. Multiple testing is controlled by
Benjamini-Hochberg FDR within each sample x covariate x statistic group.

Inputs:
  - Study database (LEVELS_DB): glucose_readings, food_logs, macro_nutrients, steps, users.
  - OUTPUTS/canonical_panel/... panel produced by analysis/canonical_panel/build_panel.py
    (loaded via cgmpaper.publication_panels.load_panel), including canonical dense-day
    selection and grid provenance.

Outputs (OUTPUTS/meals/):
  - _cache/spike_daily.parquet, _cache/spike_exposure_daily.parquet,
    _cache/spike_participants.parquet: participant-level feature caches (private).
  - carletti_spike_replication_results.csv: per sample/covariate/metric correlation results.
  - spike_exposure_attrition.csv: eligible/cohort counts per metric.
  - carletti_spike_replication.md: narrative summary report.
  - spike_exposure.metadata.json: analysis metadata.

Usage: python analysis/meals/spike_exposure.py [--stage features|analyze|all]
       [--batch-users N]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from cgmpaper.publication_panels import load_panel, write_analysis_metadata, DEFAULT_PANEL
from analysis.meals import _spike_analysis as core
from analysis.meals.build_meals import MACROS

OUT = OUTPUTS / 'meals'
DB = database_path()
METRICS = {'mean glucose':'mean_glucose', 'nocturnal minimum':'nocturnal_min',
    '% time >150':'pct_gt150', 'max spike rel':'max_spike_rel',
    'daily n spikes':'daily_n_spikes', 'spike resolution':'spike_resolution',
    'mean+SD peaks/day':'peaks_per_day'}
COVARIATES = {'HbA1c':'a1c', 'BMI':'bmi', 'Age':'age_bucket',
    'Daily logged carbs':'daily_carbs', 'Daily recorded steps':'steps'}


def dense_days(panel):
    meta=panel.attrs['panel_metadata']; path=ROOT/meta['grid_provenance']['path']
    days=pd.read_parquet(path,columns=['USER_ID','DAY','n_cells','n_raw_readings'])
    days=days.loc[days.n_cells.ge(77)&days.n_raw_readings.ge(230)].merge(
        panel[['USER_ID','anchor_date','tz_offset_hours','n_dense_days']],on='USER_ID',validate='many_to_one')
    days.DAY=pd.to_datetime(days.DAY)
    days=days.loc[(days.DAY-pd.to_datetime(days.anchor_date)).dt.days.between(-90,90)]
    if not np.array_equal(panel.USER_ID.map(days.groupby('USER_ID').size()),panel.n_dense_days):
        raise ValueError('Selected dates disagree with canonical day counts')
    return days[['USER_ID','DAY','tz_offset_hours']].sort_values(['USER_ID','DAY']).reset_index(drop=True)


def exposure_days(con):
    """Sum items to logs, then actual local dates; incomplete days stay missing.

    daily_selection is verified canonical USER_ID,DAY,tz_offset_hours. Conflicting
    steps are missing. Source step DATE is used without an additional time shift.
    """
    carb_max=MACROS['CARBS'][1]
    return con.execute(f'''
      WITH people AS (SELECT DISTINCT USER_ID,tz_offset_hours FROM daily_selection),
      macro_items AS (
        SELECT m.USER_ID,m.LOG_ID,
          CASE WHEN m.CARBS BETWEEN 0 AND {carb_max} THEN m.CARBS END AS carbs
        FROM macro_nutrients m JOIN people p USING (USER_ID)
      ), macros AS (
        SELECT USER_ID,LOG_ID,
          CASE WHEN count(carbs)=count(*) THEN sum(carbs) END AS carbs
        FROM macro_items GROUP BY USER_ID,LOG_ID
      ), food AS (
        SELECT f.USER_ID,cast(f.TIMESTAMP+p.tz_offset_hours*INTERVAL '1 hour' AS DATE) AS DAY,
          count(*) AS n_logs,count(m.carbs) AS n_complete_logs,
          CASE WHEN count(m.carbs)=count(*) THEN sum(m.carbs) END AS daily_carbs
        FROM food_logs f JOIN people p USING (USER_ID)
        LEFT JOIN macros m ON m.USER_ID=f.USER_ID AND m.LOG_ID=f.ID
        WHERE f.TYPE IN ('food','food_with_macros') AND f.TIMESTAMP IS NOT NULL
        GROUP BY f.USER_ID,DAY
      ), daily_steps AS (
        SELECT s.USER_ID,s.DAY,
          CASE WHEN count(*)=count(TOTAL_STEPS) AND min(TOTAL_STEPS)>=0
            AND isfinite(max(TOTAL_STEPS)) AND min(TOTAL_STEPS)=max(TOTAL_STEPS)
            THEN max(TOTAL_STEPS) END AS steps
        FROM steps s JOIN people p USING (USER_ID) GROUP BY s.USER_ID,s.DAY
      ) SELECT d.USER_ID,d.DAY,coalesce(f.n_logs,0) AS n_logs,
          coalesce(f.n_complete_logs,0) AS n_complete_logs,f.daily_carbs,s.steps
        FROM daily_selection d LEFT JOIN food f USING (USER_ID,DAY)
        LEFT JOIN daily_steps s USING (USER_ID,DAY) ORDER BY d.USER_ID,d.DAY
    ''').df()


def raw_day_query():
    # Arrays have observed cells only; daily_metrics rejects invalid/missing cells.
    return '''
      WITH people AS (SELECT DISTINCT USER_ID,tz_offset_hours FROM batch_days),
      local_raw AS (
        SELECT g.USER_ID,g.TS,g.VALUE,g.TS+p.tz_offset_hours*INTERVAL '1 hour' AS LOCAL_TS
        FROM glucose_readings g JOIN people p USING (USER_ID) WHERE g.VALUE BETWEEN 20 AND 500
      ), selected AS (
        SELECT g.USER_ID,d.DAY,g.TS,g.VALUE,
          floor((epoch(g.LOCAL_TS)-epoch(d.DAY))/300)::INTEGER AS cell
        FROM local_raw g JOIN batch_days d ON d.USER_ID=g.USER_ID AND d.DAY=cast(g.LOCAL_TS AS DATE)
      ), native AS (
        SELECT USER_ID,DAY,TS,cell,avg(VALUE) AS g FROM selected GROUP BY USER_ID,DAY,TS,cell
      ), cells AS (
        SELECT USER_ID,DAY,cell,avg(g) AS glucose FROM native GROUP BY USER_ID,DAY,cell
      ) SELECT USER_ID,DAY,list(cell ORDER BY cell) AS bins,list(glucose ORDER BY cell) AS glucose
        FROM cells GROUP BY USER_ID,DAY ORDER BY USER_ID,DAY
    '''


def build_features(panel,batch_size):
    days=dense_days(panel); people=panel.USER_ID.sort_values().to_numpy()
    started=time.monotonic(); parts=[]
    with duckdb.connect(str(DB),read_only=True) as con:
        con.execute('SET threads=3'); con.execute("SET memory_limit='3GB'")
        for first in range(0,len(people),batch_size):
            batch=days.loc[days.USER_ID.isin(people[first:first+batch_size])]
            con.register('batch_days',batch)
            raw=con.execute(raw_day_query()).df()
            if len(raw)!=len(batch) or raw.duplicated(['USER_ID','DAY']).any():
                raise ValueError('Raw spike grid did not retain exactly the requested dense dates')
            if set(map(tuple,raw[['USER_ID','DAY']].to_numpy()))!=set(map(tuple,batch[['USER_ID','DAY']].to_numpy())):
                raise ValueError('Raw spike grid dates differ from canonical selection')
            records=[{'USER_ID':r.USER_ID,'DAY':r.DAY,**core.daily_metrics(r.bins,r.glucose)}
                     for r in raw.itertuples(index=False)]
            parts.append(pd.DataFrame(records)); con.unregister('batch_days')
            print(f'Raw spikes: {min(first+batch_size,len(people))}/{len(people)} people; '
                  f'{time.monotonic()-started:.0f}s',flush=True)
        daily=pd.concat(parts,ignore_index=True)
        con.register('daily_selection',days); exposures=exposure_days(con)
        timezones=con.execute('SELECT USER_ID,TZ_SOURCE AS tz_source FROM users').df()
    metrics=core.summarize_days(daily)
    exposure=core.summarize_exposures(exposures).drop(columns='n_dense_days')
    features=panel[['USER_ID','a1c','bmi','age_bucket','sex','n_dense_days',
                    'mean_glucose','peaks_per_day']].merge(metrics,on='USER_ID',validate='one_to_one')
    features=features.merge(timezones,on='USER_ID',validate='one_to_one')
    features=features.merge(exposure,on='USER_ID',validate='one_to_one')
    features['sex_male']=features.sex.map({'male':1.,'female':0.})
    cache=OUT/'_cache'; cache.mkdir(parents=True,exist_ok=True)
    paths=[cache/'spike_daily.parquet',cache/'spike_exposure_daily.parquet',cache/'spike_participants.parquet']
    for table,path in zip([daily,exposures,features],paths):table.to_parquet(path,index=False)
    return features


def load_features(panel):
    cache=OUT/'_cache'
    return pd.read_parquet(cache/'spike_participants.parquet')


def analyze(panel,features):
    rows=[]
    for sample,frame in [('primary',features),('high_loggers',features.loc[features.high_logger]),
                         ('known_timezone',features.loc[features.tz_source.eq('food_logs_mode')])]:
        covariates=COVARIATES if sample=='primary' else {'Daily logged carbs':'daily_carbs'}
        for covariate,x in covariates.items():
            for metric,y in METRICS.items():
                rows.append({'sample':sample,'covariate':covariate,'metric':metric,
                    **core.pair_result(frame,x,y,age_exposure=x=='age_bucket')})
            print(f'Correlations: {sample} / {covariate}',flush=True)
    results=pd.DataFrame(rows); results['q_partial']=np.nan; results['q_simple']=np.nan
    for _,group in results.groupby(['sample','covariate']):
        for kind in ['partial','simple']:
            results.loc[group.index,'q_'+kind]=core.bh_fdr(group['p_'+kind])
    paths=[]
    def save(name,frame):
        path=OUT/name;frame.to_csv(path,index=False);paths.append(path)
    save('carletti_spike_replication_results.csv',results)
    save('spike_exposure_attrition.csv',pd.DataFrame([
        {'metric':metric,'n_eligible':int(features[column].notna().sum()),'n_cohort':len(features)}
        for metric,column in METRICS.items()]))
    coverage={c:{'n':int(features[c].notna().sum()),'median':float(features[c].median()),
                  'p10':float(features[c].quantile(.1)),'p90':float(features[c].quantile(.9))}
              for c in ['n_dense_days','n_peak_days','n_nights','n_logged_days','n_complete_carb_days',
                        'logging_day_fraction','complete_carb_day_fraction','resolved_fraction']}
    details={'cohort':'no_diabetes_range','definition_version':core.VERSION,
        'n_participants':len(features),'n_high_loggers':int(features.high_logger.sum()),'coverage':coverage,
        'bootstrap_replicates':500,'seed':20260916,
        'partial_adjustment':'ranked age bucket and recorded binary sex; Age exposure controls sex only',
        'multiple_testing':'BH separately within sample x covariate x statistic, seven metrics',
        'sources':{'article':'https://www.nature.com/articles/s41591-025-03849-7',
            'code':'https://github.com/scripps-research/progress_nat_med',
            'inspected_commit':'9f50e5b36e05a4a0827deedbf00c293d49c1fb8c'}}
    report=['# Operational spike and exposure sensitivity','',
        'This is not an exact PROGRESS replication. No causal inference follows.',
        f'Canonical primary participants: {len(features):,}; high loggers: {int(features.high_logger.sum()):,}.','',
        'Carbohydrate is the mean of daily sums over complete logged days, not per-meal grams.',
        'Unlogged and incomplete days are missing. Coverage and selection remain limitations.',
        'Mean+SD peak frequency is copied from the screening-model predictor. Relative-rise',
        'count, amplitude and half-recovery are a separate observed-cell sensitivity.','',
        '| Sample | Metric | Paired n | Spearman rho | 95% bootstrap CI | Age/sex partial rho |',
        '|---|---|---:|---:|---|---:|']
    for r in results.loc[results.covariate.eq('Daily logged carbs')].itertuples():
        report.append(f'| {r.sample} | {r.metric} | {r.n:,} | {r.rho_simple:.3f} | '
                      f'{r.rho_simple_ci_low:.3f} to {r.rho_simple_ci_high:.3f} | {r.rho_partial:.3f} |')
    report+=['','Intervals are pointwise participant-percentile bootstrap intervals (500 draws).',
        'Simple and partial associations use identical complete cases, including age and sex.',
        'Differences between correlations are not tested. Nonsignificance is not equivalence.',
        'Age is bucketed. PRS, microbiome and resting heart rate are unavailable; no proxies are substituted.',
        'Recovery is conditional on observed half-recovery and is subject to censoring.']
    report_path=OUT/'carletti_spike_replication.md'; report_path.write_text('\n'.join(report)+'\n'); paths.append(report_path)
    write_analysis_metadata(OUT/'spike_exposure.metadata.json',script=__file__,
        panels={'symmetric':DEFAULT_PANEL},outputs=paths,details=details)
    print(f'Wrote {len(results)} aggregate associations',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['features','analyze','all'],default='all')
    parser.add_argument('--batch-users',type=int,default=200)
    args=parser.parse_args(argv)
    if args.batch_users<1:parser.error('--batch-users must be positive')
    OUT.mkdir(parents=True,exist_ok=True)
    panel=load_panel(DEFAULT_PANEL,required=['peaks_per_day'],expected_window='symmetric')
    features=build_features(panel,args.batch_users) if args.stage in ['features','all'] else load_features(panel)
    if args.stage in ['analyze','all']:analyze(panel,features)


if __name__=='__main__':main()
