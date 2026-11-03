#!/usr/bin/env python3
"""Build the whole-record cohort reference used by Figure 1 and the cohort summary tables.

Scope: all non-internal participant accounts in the study database. For each
participant, combines a whole-record (not sensor-window-bounded) count of accepted
CGM readings with behavior-log presence (food, steps, weight, sleep, exercise) and
consent-duration statistics, and reports participant counts by modality combination
(the "Euler diagram" cells used in Figure 1) and by age/sex. Also builds a
first-available-value lab reference table (lipids, HOMA-IR, eGFR) merged with the
behavior reference.

Consent durations can come from (a) an included upstream aggregate consent
reference (default), (b) a verified reuse of previously derived consent
counts/durations, or (c) an explicit raw consent export file (--consent-file).

Inputs (read via LEVELS_DB unless noted):
  users, food_logs, steps, weight, labs tables
  analysis/canonical_panel/build_panel.py output (OUTPUTS/canonical_panel/...), via
    cgmpaper.publication_panels.load_panel
  OUTPUTS/cohort/consent_reference.json  aggregate consent-duration summary derived upstream from the consent
                                         records (restricted input; read with cgmpaper.consent_reference)

Outputs (OUTPUTS/cohort/...):
  cgm_reference.parquet - participant-level whole-record CGM counts (private)
  cohort_participants.parquet - participant-level cohort table (private)
  reference_values.parquet - participant-level first lab values and behavior reference (private)
  cohort_summary.json - aggregate cohort counts and statistics

Usage: python analysis/cohort/cohort_reference.py [--stage cgm|cohort|all]
  [--consent-file PATH | --reuse-consent-from-verified-reference] [--batch-size N]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from analysis.cohort import _cohort_reference as reference
from analysis.cohort import _behavior_reference as behavior_reference
from analysis.cohort import _first_lab_values as first_lab_values
from cgmpaper import local_exposure_dates
from cgmpaper.lab_reference import load_normalized_labs,same_day_indices
from cgmpaper.publication_panels import load_panel
from cgmpaper.consent_reference import load_reference

OUT=OUTPUTS / 'cohort'
CONSENT_REFERENCE=OUTPUTS/'cohort'/'consent_reference.json'   # aggregate consent summary (restricted upstream input)
DB=database_path()


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def cgm(con,roster,batch_size):
    cache=OUT/'cache/cgm_batches';cache.mkdir(parents=True,exist_ok=True)
    roster=roster.loc[roster.TZ_OFFSET_HOURS.between(-12,14)].sort_values('USER_ID')
    metadata_path=cache/'manifest.json'
    metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    frames=[]
    for start in range(0,len(roster),batch_size):
        name=f'part_{start:06d}.parquet';path=cache/name
        if path.exists():
            frame=pd.read_parquet(path)
        else:
            began=time.time()
            frame=reference.cgm_batch(con,roster.iloc[start:start+batch_size])
            frame.to_parquet(path,index=False)
            write_json(metadata_path,metadata)
            print(f'CGM batch {start//batch_size+1}/{(len(roster)+batch_size-1)//batch_size}: {len(frame):,} participants; {time.time()-began:.1f}s',flush=True)
        frames.append(frame)
    result=pd.concat(frames,ignore_index=True)
    if result.USER_ID.duplicated().any():raise ValueError('Duplicate CGM participant')
    result.to_parquet(OUT/'cgm_reference.parquet',index=False)
    metadata['complete']=True
    write_json(metadata_path,metadata)
    print(f'Whole-record observed-cell CGM complete: {len(result):,} participants',flush=True)
    return result


def verified_consent_reuse(roster, *, directory=None, database=None):
    """Reuse unchanged derived consent counts/durations; not a raw reproduction."""
    directory=Path(directory) if directory is not None else OUT
    database=Path(database) if database is not None else DB
    if roster.USER_ID.isna().any() or roster.USER_ID.duplicated().any():
        raise ValueError('Noninternal roster must contain unique nonmissing participants')
    roster_ids=sorted(roster.USER_ID.astype(str))
    cache=directory/'cache/verified_consent_reference.parquet'
    provenance=cache.with_suffix('.metadata.json')
    if cache.exists()!=provenance.exists():
        raise ValueError('Incomplete verified consent cache; explicit recovery is required')
    columns=['USER_ID','num_periods','total_consented_days']
    if cache.exists():
        meta=json.loads(provenance.read_text())
        consent=pd.read_parquet(cache)
        if consent.columns.tolist()!=columns:
            raise ValueError('Verified consent cache has unexpected columns')
    else:
        source=directory/'cohort_participants.parquet'
        summary_path=directory/'cohort_summary.json'
        summary=json.loads(summary_path.read_text())
        consent=pd.read_parquet(source,columns=columns)
        meta=dict(
            consent_overlapping_periods=summary.get('consent_overlapping_periods'),
            scope='unchanged derived consent counts/durations from verified prior reference; not raw consent reproduction')
    if (consent.USER_ID.isna().any() or consent.USER_ID.duplicated().any()
        or sorted(consent.USER_ID.astype(str))!=roster_ids):
        raise ValueError('Consent reuse and exact noninternal roster differ')
    values=consent[['num_periods','total_consented_days']].to_numpy(float)
    if (not np.isfinite(values).all() or (values[:,0]<1).any()
        or (values[:,0]!=np.floor(values[:,0])).any() or (values[:,1]<0).any()):
        raise ValueError('Invalid reused consent counts/durations')
    overlaps=meta.get('consent_overlapping_periods')
    if type(overlaps) is not int or overlaps<0:
        raise ValueError('Missing overlap aggregate')
    if not cache.exists():
        cache.parent.mkdir(parents=True,exist_ok=True)
        consent.to_parquet(cache,index=False)
        write_json(provenance,meta)
    return consent,meta


def cohort(con,roster,consent_file=None,reuse_consent=False):
    all_ids=set(con.execute('SELECT USER_ID FROM users').df().USER_ID)
    aggregate_consent = consent_file is None and not reuse_consent
    if reuse_consent:
        consent,consent_input=verified_consent_reuse(roster)
        overlapping_periods=consent_input['consent_overlapping_periods']
    elif consent_file is not None:
        consent=reference.consent_durations(con,consent_file)
        if set(consent.USER_ID)!=all_ids:raise ValueError('Consent and database rosters differ')
        overlapping_periods=int(consent.loc[consent.USER_ID.isin(roster.USER_ID),'overlapping_periods'].sum())
        consent_input=dict(scope='raw consent export; exact full database roster verified')
    else:
        reference_path=CONSENT_REFERENCE
        consent_input=load_reference(reference_path,expected_participants=len(roster))
        consent_statistics=consent_input['consent_summary']
        overlapping_periods=consent_statistics['consent_overlapping_periods']
    if aggregate_consent:
        data=roster.copy()
    else:
        data=roster.merge(consent,on='USER_ID',validate='one_to_one')
        consent_statistics=dict(consent_median=float(data.total_consented_days.median()),
                                consent_q25=float(data.total_consented_days.quantile(.25)),
                                consent_q75=float(data.total_consented_days.quantile(.75)),
                                consent_overlapping_periods=overlapping_periods)
    behavior=behavior_reference.behavior_reference(con)
    data=data.merge(behavior,on='USER_ID',validate='one_to_one')
    # Presence of unvalidated source records is distinct from accepted reference values.
    for name,sql in {
        'food_log_rows':"SELECT USER_ID,count(*) AS n FROM food_logs WHERE TYPE IN ('food','food_with_macros') AND TIMESTAMP IS NOT NULL GROUP BY USER_ID",
        'step_rows':'SELECT USER_ID,count(*) AS n FROM steps GROUP BY USER_ID',
        'weight_measurements':'SELECT USER_ID,count(*) AS n FROM weight GROUP BY USER_ID',
        'total_lab_records':'SELECT USER_ID,count(*) AS n FROM labs GROUP BY USER_ID',
    }.items():
        counts=con.execute(sql).df().rename(columns={'n':name})
        data=data.merge(counts,on='USER_ID',how='left',validate='one_to_one')
        data[name]=data[name].fillna(0).astype(int)
    cgm_path=OUT/'cgm_reference.parquet'
    manifest=json.loads((OUT/'cache/cgm_batches/manifest.json').read_text())
    if manifest.get('complete') is not True:
        raise ValueError('Whole-record CGM is incomplete')
    cgm_data=pd.read_parquet(cgm_path)
    data=data.merge(cgm_data[['USER_ID','accepted_raw_readings']],on='USER_ID',how='left',validate='one_to_one')
    data['glucose_rows']=data.accepted_raw_readings.fillna(0).astype(int)
    data['log_rows']=data.exercise_log_count
    data=data.rename(columns={'SEX':'sex','AGE_BUCKET':'age_bucket'})
    data.to_parquet(OUT/'cohort_participants.parquet',index=False)
    daily,_,audit=load_normalized_labs(con)
    first=daily.sort_values(['USER_ID','TYPE','DRAW_DATE']).drop_duplicates(['USER_ID','TYPE'])
    lab=first.pivot(index='USER_ID',columns='TYPE',values='value').reset_index()
    indices=same_day_indices(daily)
    homa=indices.loc[indices.homa_ir.notna()].sort_values(['USER_ID','DRAW_DATE']).drop_duplicates('USER_ID')[['USER_ID','homa_ir']]
    lab=lab.merge(homa,on='USER_ID',how='outer',validate='one_to_one').rename(columns={'ldl_cholesterol':'ldl','hdl_cholesterol':'hdl'})
    extra,extra_audit=first_lab_values.build_first_values(con)
    lab=lab.merge(extra[['USER_ID','egfr_creatinine']],on='USER_ID',how='outer',validate='one_to_one')
    values=lab.merge(behavior,on='USER_ID',how='outer',validate='one_to_one')
    values.to_parquet(OUT/'reference_values.parquet',index=False)
    canonical=load_panel(cohort='all_lab_linked',expected_window='symmetric')
    primary=canonical.no_diabetes_range
    flags={k:data[col].gt(0) for k,col in dict(cgm='glucose_rows',food='food_log_rows',exercise='log_rows',
        steps='step_rows',weight='weight_measurements',labs='total_lab_records',sleep='sleep_log_count').items()}
    core=flags['cgm']&flags['food'];beh=core&flags['exercise']&flags['steps']&flags['weight'];labs=core&flags['labs']
    pair=data.USER_ID.isin(canonical.USER_ID)
    e=dict(core=int(core.sum()),behavior=int(beh.sum()),sleep_in_behavior=int((beh&flags['sleep']).sum()),
           labs=int(labs.sum()),both=int((beh&labs).sum()),behavior_only=int((beh&~labs).sum()),
           labs_only=int((labs&~beh).sum()),core_neither=int((core&~beh&~labs).sum()),
           pair=int(pair.sum()),pair_both=int((pair&beh&labs).sum()),pair_labs_only=int((pair&labs&~beh).sum()),
           pair_outside_core=int((pair&~core).sum()))
    if e['pair']!=len(canonical) or (pair&core&~labs).any():raise ValueError('Canonical membership disagrees with source modalities')
    sex=data.sex.where(data.sex.isin(['female','male']),'not_recorded').fillna('not_recorded')
    ages={}
    for label,lo,hi in [('20s',20,20),('30s',30,30),('40s',40,40),('50s',50,50),('60s',60,60),('70s',70,70),('80s+',80,100)]:
        band=data.age_bucket.between(lo,hi)
        ages[label]={s:int((band&sex.eq(s)).sum()) for s in ['female','male','not_recorded']}
    summary=dict(version=reference.VERSION,total=len(data),excluded_accounts=len(all_ids)-len(data),euler=e,
                 coverage={k+'_any':int(v.sum()) for k,v in flags.items()},aligned=len(canonical),primary_aligned=int(primary.sum()),
                 **consent_statistics,
                 ages=ages,female=int(sex.eq('female').sum()),male=int(sex.eq('male').sum()),
                 age_off_range=int(data.age_bucket.eq(10).sum()),age_missing=int((~data.age_bucket.between(10,100).fillna(False)).sum()),
                 timezone_source_counts=data.TZ_SOURCE.fillna('missing').value_counts().to_dict(),
                 cgm_n=len(cgm_data),laboratory_audit=audit,egfr_audit=extra_audit['egfr'])
    write_json(OUT/'cohort_summary.json',summary)
    write_json(OUT/'reference_analysis.metadata.json',dict(
        consent_input_mode=('upstream_aggregate_reference' if aggregate_consent else
                            'verified_derived_reuse' if reuse_consent else 'optional_raw_reconstruction'),
        consent_file_read=not (aggregate_consent or reuse_consent),
        consent_input=consent_input,behavior_definition_version=behavior_reference.VERSION,
        event_date_definition_version=local_exposure_dates.EXPOSURE_VERSION,
        behavior_date_rule='sleep END_TS wake date; exercise START_TS date; fixed participant offset in [-12,14]; missing dates excluded from references, not source-presence counts'))
    print(json.dumps({k:summary[k] for k in ['total','excluded_accounts','cgm_n','aligned','primary_aligned','consent_median','euler']},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['cgm','cohort','all'],default='all')
    consent_options=parser.add_mutually_exclusive_group()
    consent_options.add_argument('--consent-file',type=Path,help='Optional raw reconstruction; default uses included upstream aggregate consent reference')
    consent_options.add_argument('--reuse-consent-from-verified-reference',action='store_true',
                        help='cohort stage only: explicitly reuse fingerprint-verified prior derived consent; not full raw reproduction')
    parser.add_argument('--batch-size',type=int,default=1000)
    args=parser.parse_args()
    if not 1<=args.batch_size<=2000:parser.error('Batch size must be 1..2000')
    if args.reuse_consent_from_verified_reference and args.stage!='cohort':
        parser.error('Verified derived consent reuse is allowed only with --stage cohort')
    if args.consent_file is not None and not args.consent_file.exists():
        parser.error(f'Consent file not found: {args.consent_file}')
    OUT.mkdir(parents=True,exist_ok=True)
    with duckdb.connect(str(DB),read_only=True) as con:
        con.execute("SET threads=3; SET memory_limit='3GB'")
        con.execute("SET temp_directory=?",[str(OUT/'cache/duckdb_spill')])
        roster=con.execute('SELECT USER_ID,SEX,AGE_BUCKET,TZ_OFFSET_HOURS,TZ_SOURCE FROM users WHERE IS_INTERNAL_USER=FALSE ORDER BY USER_ID').df()
        if args.stage in ('cgm','all'):cgm(con,roster,args.batch_size)
        if args.stage in ('cohort','all'):cohort(con,roster,args.consent_file,args.reuse_consent_from_verified_reference)


if __name__=='__main__':main()
