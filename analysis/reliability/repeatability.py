#!/usr/bin/env python3
"""Computes whole-cohort CGM test-retest reliability and metric structure.

Two analyses on the same dense-day substrate as analysis/daily_glucose/build_day_panel.py:
(1) test-retest reliability between a participant's first and second non-overlapping
7-day windows (ICC(A,1), Pearson/Spearman r, median absolute/signed difference), both
pooled and stratified by the gap between windows, plus a window-length curve (1-14 days)
under several pairing designs (fixed_gap, adjacent, fixed_start_offset, interleaved) on
participants whose first qualifying dense run spans at least 42 days, with windows kept
14 full days apart for the primary (fixed_gap) design; (2) descriptive structure (PCA and
three-factor principal-axis extraction with varimax rotation) across CGM metric axes from
first-week features. A day is dense if it has at least 77 of 96 15-minute cells observed;
short gaps (<=2 cells) are linearly interpolated before computing metrics.

Inputs: glucose_readings and users tables via LEVELS_DB; daily dense-day/day-grid structure
from analysis/daily_glucose/build_day_panel.py (analysis/daily_glucose/_day_panel.py).

Outputs (OUTPUTS/reliability/): repeatability_selection.json, axis_reliability.csv,
gap_stratified_reliability.csv, reliability_by_window_length.csv, pca_variance.csv,
pca_loadings.csv, efa_three_factor_loadings.csv, axis_correlations.csv,
repeatability_analysis.metadata.json; cgm_axis_features_primary.parquet (participant-level,
stays private) under OUT/cache.

Usage: python analysis/reliability/repeatability.py [--stage select|grids|features|analyze|all]
[--output-dir DIR] [--chunk-users N] [--bootstrap N] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from cgmpaper.cgm_grid_metrics import GRID_FEATURE_VERSION,interpolate_short_gaps,autocorrelation,paired_difference
from analysis.daily_glucose import _day_panel as temporal

VERSION="publication-repeatability"
GRID_EXPORT_VERSION="masked-array-null-preserving"
OUT=OUTPUTS / 'reliability'
DB=database_path()
METRICS=["mean_glucose","sd_glucose","modd","cv_glucose","ac_var"]
ALL_METRICS=["mean_glucose","median_glucose","p10_glucose","p90_glucose","sd_glucose","cv_glucose",
             "iqr_glucose","tar140","tbr70","modd","day_mean_sd","conga_2h","gvp","ac_mean","ac_var"]
DESIGNS=["fixed_gap","adjacent","fixed_start_offset","interleaved"]
SEED=20260916


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def log(message):print(time.strftime("%H:%M:%S"),message,flush=True)


def load_module(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def design_days(k,design):
    if not 1<=k<=14:raise ValueError("Window length must be 1..14 days")
    if design=="fixed_gap":return np.arange(k),np.arange(k+14,2*k+14)
    if design=="adjacent":return np.arange(k),np.arange(k,2*k)
    if design=="fixed_start_offset":return np.arange(k),np.arange(14,14+k)
    if design=="interleaved":return np.arange(0,2*k,2),np.arange(1,2*k,2)
    raise ValueError("Unknown window design")


def metrics_for_days(day_numbers,grid,full=False):
    """Same core definitions as canonical windows; actual dates retain gaps."""
    days=np.asarray(day_numbers,dtype=np.int64);grid=np.asarray(grid,dtype=float)
    if grid.shape!=(len(days),96) or len(days)==0 or np.any(np.diff(days)<=0):
        raise ValueError("Expected unique increasing calendar dates and one 96-cell row/day")
    if np.any(np.isfinite(grid).sum(axis=1)<77):raise ValueError("Non-dense day entered metrics")
    day_grid=np.full_like(grid,np.nan);pieces=[];raw_pieces=[]
    for block in np.split(np.arange(len(days)),np.flatnonzero(np.diff(days)!=1)+1):
        raw=grid[block].reshape(-1);filled=interpolate_short_gaps(raw,2)
        day_grid[block]=filled.reshape(-1,96)
        if pieces:pieces.append(np.full(10,np.nan));raw_pieces.append(np.full(10,np.nan))
        pieces.append(filled);raw_pieces.append(raw)
    series=np.concatenate(pieces);raw_series=np.concatenate(raw_pieces)
    finite=series[np.isfinite(series)];mean=float(finite.mean());sd=float(finite.std(ddof=1))
    adjacent_dates=np.diff(days)==1
    pairmask=np.isfinite(day_grid[:-1])&np.isfinite(day_grid[1:])&adjacent_dates[:,None]
    differences=np.abs(day_grid[1:]-day_grid[:-1])[pairmask]
    ac=np.array([autocorrelation(series,lag) for lag in range(1,11)])
    result=dict(mean_glucose=mean,sd_glucose=sd,cv_glucose=100*sd/mean,
                modd=float(differences.mean()) if len(differences) else np.nan,
                ac_var=float(ac.var(ddof=1)) if np.isfinite(ac).all() else np.nan)
    if full:
        q10,q25,q50,q75,q90=np.quantile(finite,[.1,.25,.5,.75,.9])
        diff=paired_difference(series,1);lag8=paired_difference(series,8)
        rawmask=np.isfinite(grid[:-1])&np.isfinite(grid[1:])&adjacent_dates[:,None]
        rawdiff=np.abs(grid[1:]-grid[:-1])[rawmask]
        rawac=np.array([autocorrelation(raw_series,lag) for lag in range(1,11)])
        result.update(median_glucose=q50,p10_glucose=q10,p90_glucose=q90,iqr_glucose=q75-q25,
            tar140=100*np.mean(finite>140),tbr70=100*np.mean(finite<70),
            day_mean_sd=float(np.nanmean(day_grid,axis=1).std(ddof=1)) if len(days)>1 else np.nan,
            conga_2h=float(lag8.std(ddof=1)) if len(lag8)>1 else np.nan,
            gvp=100*(np.sqrt(diff**2+15**2).sum()/(15*len(diff))-1) if len(diff) else np.nan,
            ac_mean=float(ac.mean()) if np.isfinite(ac).all() else np.nan,
            modd_raw=float(rawdiff.mean()) if len(rawdiff) else np.nan,
            ac_mean_raw=float(rawac.mean()) if np.isfinite(rawac).all() else np.nan,
            ac_var_raw=float(rawac.var(ddof=1)) if np.isfinite(rawac).all() else np.nan,
            n_15m_observed=int(np.isfinite(grid).sum()),
            n_15m_interpolated=int(np.isfinite(day_grid).sum()-np.isfinite(grid).sum()),
            min_15m_cells_day=int(np.isfinite(grid).sum(axis=1).min()))
    return result


def icc_a1(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);n=len(x)
    if n<2:return np.nan
    grand=(x.mean()+y.mean())/2;row=(x+y)/2
    msr=2*np.square(row-grand).sum()/(n-1)
    msc=n*((x.mean()-grand)**2+(y.mean()-grand)**2)
    mse=(np.square(x-row-x.mean()+grand).sum()+np.square(y-row-y.mean()+grand).sum())/(n-1)
    denom=msr+mse+2*(msc-mse)/n
    return float((msr-mse)/denom) if denom>0 else np.nan


def summarize_pair(x,y,n_eligible,bootstrap=500,seed=SEED):
    mask=np.isfinite(x)&np.isfinite(y);x=np.asarray(x)[mask];y=np.asarray(y)[mask];n=len(x)
    out=dict(n_eligible=int(n_eligible),n_users=n,n_missing=int(n_eligible-n),icc_a1=np.nan,
             ci_low=np.nan,ci_high=np.nan,pearson_r=np.nan,median_abs_diff=np.nan)
    if n<2:return out
    out.update(icc_a1=icc_a1(x,y),pearson_r=float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else np.nan,
               median_abs_diff=float(np.median(np.abs(x-y))))
    rng=np.random.default_rng(seed);boots=np.empty(bootstrap)
    for b in range(bootstrap):
        pick=rng.integers(0,n,n);boots[b]=icc_a1(x[pick],y[pick])
    if np.isfinite(boots).all():out["ci_low"],out["ci_high"]=np.quantile(boots,[.025,.975])
    return out


def select_windows(con,cache):
    temporal_meta=temporal.require_cache()

    con.execute(f"""CREATE TEMP TABLE dense_days AS
      SELECT g.USER_ID,CAST(g.LOCAL_DAY AS TIMESTAMP) AS DAY,g.N_READINGS,g.N_CELLS_15M,t.TZ_OFFSET_HR
      FROM read_parquet('{temporal.CACHE/'gluc_local_day.parquet'}') g
      JOIN read_parquet('{temporal.CACHE/'user_tz_offset.parquet'}') t USING(USER_ID)""")
    con.execute("""CREATE TEMP TABLE positioned AS SELECT *,
      DAY-row_number()OVER(PARTITION BY USER_ID ORDER BY DAY)*INTERVAL '1 day' AS island
      FROM dense_days""")
    con.execute("""CREATE TEMP TABLE runs AS SELECT USER_ID,island,min(DAY) AS run_start,count(*) AS n_days
      FROM positioned GROUP BY USER_ID,island""")
    windows=con.sql("""WITH indexed AS (SELECT *,row_number()OVER(PARTITION BY USER_ID,island ORDER BY DAY) AS pos FROM positioned),
      blocks AS (SELECT USER_ID,island,floor((pos-1)/7) AS block,min(DAY) AS window_start,
        count(*) AS dense_days,min(N_READINGS) AS min_raw_readings_day,sum(N_READINGS) AS raw_readings,
        min(N_CELLS_15M) AS min_cells_day,any_value(TZ_OFFSET_HR) AS TZ_OFFSET_HOURS
        FROM indexed GROUP BY USER_ID,island,block HAVING count(*)=7),
      ranked AS (SELECT *,row_number()OVER(PARTITION BY USER_ID ORDER BY window_start) AS window_number FROM blocks)
      SELECT r.USER_ID,r.USER_ID||'_w'||window_number::VARCHAR AS window_id,window_number,window_start,
        window_start+INTERVAL '7 days' AS window_end,window_start+INTERVAL '6 days' AS last_dense_day,
        dense_days,min_raw_readings_day,raw_readings,min_cells_day,r.TZ_OFFSET_HOURS,u.SEX,u.AGE_BUCKET
      FROM ranked r JOIN users u USING(USER_ID) WHERE window_number<=2 ORDER BY USER_ID,window_number""").df()
    spans=con.sql("""WITH first_run AS (SELECT *,row_number()OVER(PARTITION BY USER_ID ORDER BY run_start) AS rn FROM runs WHERE n_days>=42)
      SELECT f.USER_ID,run_start,n_days FROM first_run f WHERE rn=1 ORDER BY USER_ID""").df()
    all_requests=pd.concat([windows[["USER_ID","window_start","window_end"]],
        spans.rename(columns={"run_start":"window_start"}).assign(window_end=lambda x:x.window_start+pd.Timedelta(days=42))[["USER_ID","window_start","window_end"]]],ignore_index=True)
    con.register("all_requests",all_requests)
    requested=con.sql("""WITH requested AS (SELECT DISTINCT r.USER_ID,d.DAY
      FROM all_requests r CROSS JOIN LATERAL generate_series(r.window_start,r.window_end-INTERVAL '1 day',INTERVAL '1 day') d(DAY))
      SELECT g.* FROM dense_days g JOIN requested r USING(USER_ID,DAY) ORDER BY USER_ID,DAY""").df()
    con.unregister("all_requests")
    windows.to_parquet(cache/"seven_day_windows.parquet",index=False)
    spans.to_parquet(cache/"length_42day_spans.parquet",index=False)
    requested.to_parquet(cache/"requested_dense_days.parquet",index=False)
    log(f"Selected {len(windows):,} seven-day windows ({windows.USER_ID.nunique():,} people); {len(spans):,} common 42-day participants; {len(requested):,} requested days")
    return windows,spans,requested,temporal_meta


def grid_query():
    return """WITH cells AS (
      SELECT d.USER_ID,d.DAY,floor(epoch(g.TS-(d.DAY-d.TZ_OFFSET_HR*INTERVAL '1 hour'))/900)::INTEGER AS cell,
        avg(g.VALUE) AS glucose,count(*) AS readings
      FROM glucose_readings g JOIN requested_chunk d ON g.USER_ID=d.USER_ID
        AND g.TS>=d.DAY-d.TZ_OFFSET_HR*INTERVAL '1 hour'
        AND g.TS<d.DAY+INTERVAL '1 day'-d.TZ_OFFSET_HR*INTERVAL '1 hour'
      WHERE g.USER_ID>=? AND g.USER_ID<=? AND g.VALUE BETWEEN 20 AND 500
      GROUP BY d.USER_ID,d.DAY,cell)
      SELECT d.USER_ID,d.DAY,any_value(d.TZ_OFFSET_HR) AS TZ_OFFSET_HR,
        any_value(d.N_READINGS) AS source_readings,any_value(d.N_CELLS_15M) AS source_cells,
        sum(c.readings) AS accepted_readings,count(c.cell) AS n_cells,
        list(c.glucose ORDER BY r.cell) AS glucose
      FROM requested_chunk d CROSS JOIN range(96) r(cell)
      LEFT JOIN cells c ON c.USER_ID=d.USER_ID AND c.DAY=d.DAY AND c.cell=r.cell
      GROUP BY d.USER_ID,d.DAY ORDER BY d.USER_ID,d.DAY"""


def normalize_grid_arrays(frame):
    """Materialize DuckDB masked-list NULLs before pandas/Arrow serialization.

    Arrow conversion of ndarray objects can expose the masked array's underlying
    storage, including zeros. SQL count agreement alone cannot detect that loss.
    """
    result=frame.copy()
    result['glucose']=[np.ma.asarray(values,dtype=float).filled(np.nan) for values in frame.glucose]
    validate_grid_arrays(result)
    return result


def validate_grid_arrays(frame):
    grid=np.stack(frame.glucose.to_numpy())
    if grid.shape!=(len(frame),96):raise ValueError('Invalid exported daily grid shape')
    finite=np.isfinite(grid)
    if not np.array_equal(finite.sum(axis=1),frame.n_cells.to_numpy()):
        raise ValueError('Exported grid NULL mask disagrees with observed-cell count')
    if np.any((grid[finite]<20)|(grid[finite]>500)):
        raise ValueError('Exported grid contains invalid observed glucose')


def extract_grids(con,requested,cache,chunk_users,force=False):
    folder=cache/"repeatability_daily_grids";folder.mkdir(exist_ok=True)
    manifest=cache/"daily_grid_manifest.json"
    digest=hashlib.sha256(grid_query().encode())
    digest.update(GRID_EXPORT_VERSION.encode())
    digest.update(inspect.getsource(normalize_grid_arrays).encode())
    digest.update(pd.util.hash_pandas_object(requested,index=False).to_numpy().tobytes())
    stat=DB.stat();digest.update(str((stat.st_size,stat.st_mtime_ns)).encode());fingerprint=digest.hexdigest()
    if manifest.exists() and not force:
        saved=json.loads(manifest.read_text())
        if saved.get("fingerprint")==fingerprint and all((folder/name).exists() and sha(folder/name)==value for name,value in saved["parts"].items()):
            log("Using fingerprint-verified daily grids");return saved
    users=np.sort(requested.USER_ID.unique());parts={};total=0
    for i,start in enumerate(range(0,len(users),chunk_users)):
        selected=users[start:start+chunk_users];chunk=requested[requested.USER_ID.isin(selected)]
        con.register("requested_chunk",chunk)
        result=con.execute(grid_query(),[str(selected[0]),str(selected[-1])]).df()
        con.unregister("requested_chunk")
        if len(result)!=len(chunk) or not result.source_readings.eq(result.accepted_readings).all() or not result.source_cells.eq(result.n_cells).all():
            raise ValueError("Raw grid counts disagree with the validated dense-day substrate")
        result=normalize_grid_arrays(result)
        path=folder/f"part_{i:04d}.parquet";result.to_parquet(path,index=False)
        validate_grid_arrays(pd.read_parquet(path))
        parts[path.name]=sha(path);total+=len(result)
        log(f"Raw grids {min(start+chunk_users,len(users)):,}/{len(users):,} participants; {total:,} days")
    record=dict(fingerprint=fingerprint,parts=parts,n_days=total,n_users=len(users),
                database=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns),grid_export_version=GRID_EXPORT_VERSION)
    manifest.write_text(json.dumps(record,indent=2)+"\n")
    return record


def compute_features(windows,spans,cache,force=False):
    manifest=json.loads((cache/"daily_grid_manifest.json").read_text())
    if manifest.get('grid_export_version')!=GRID_EXPORT_VERSION:
        raise ValueError('Daily grids predate NULL-preserving serialization; rerun --stage grids')
    feature_meta=cache/"repeatability_features.metadata.json"
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode())
    for value in [inspect.getsource(metrics_for_days),inspect.getsource(design_days),sha(ROOT/"cgmpaper/cgm_grid_metrics.py")]:digest.update(value.encode())
    digest.update(sha(cache/"seven_day_windows.parquet").encode());digest.update(sha(cache/"length_42day_spans.parquet").encode())
    fingerprint=digest.hexdigest();feature_path=cache/"seven_day_cgm_features.parquet"
    if feature_meta.exists() and not force:
        old=json.loads(feature_meta.read_text())
        if old.get("fingerprint")==fingerprint and all((cache/name).exists() and sha(cache/name)==value for name,value in old["files"].items()):
            log("Using fingerprint-verified repeatability features");return
    by_user={uid:part for uid,part in windows.groupby("USER_ID",sort=False)}
    span_start=spans.set_index("USER_ID").run_start.to_dict();features=[];pair_files={}
    pair_folder=cache/"length_pairs";pair_folder.mkdir(exist_ok=True)
    for name in manifest["parts"]:
        path=cache/"repeatability_daily_grids"/name
        if sha(path)!=manifest["parts"][name]:raise ValueError("Daily grids changed after extraction")
        daily=pd.read_parquet(path);validate_grid_arrays(daily);pair_rows=[]
        for uid,part in daily.groupby("USER_ID",sort=False):
            days=pd.to_datetime(part.DAY).to_numpy(dtype="datetime64[D]").astype(np.int64)
            grid=np.stack(part.glucose.to_numpy())
            for w in by_user[uid].itertuples():
                first=np.datetime64(w.window_start,"D").astype(np.int64);wanted=first+np.arange(7)
                indices=np.searchsorted(days,wanted)
                if not np.array_equal(days[indices],wanted):raise ValueError("Selected week lost a dense calendar day")
                features.append(dict(window_id=w.window_id,**metrics_for_days(wanted,grid[indices],full=True)))
            if uid not in span_start:continue
            first=np.datetime64(span_start[uid],"D").astype(np.int64);wanted=first+np.arange(42)
            indices=np.searchsorted(days,wanted)
            if not np.array_equal(days[indices],wanted):raise ValueError("Common 42-day span lost a dense day")
            span_grid=grid[indices];metric_cache={}
            for design in DESIGNS:
                for k in range(1,15):
                    ab=[]
                    for chosen in design_days(k,design):
                        key=tuple(chosen)
                        if key not in metric_cache:metric_cache[key]=metrics_for_days(chosen,span_grid[chosen])
                        ab.append(metric_cache[key])
                    row=dict(USER_ID=uid,design=design,days=k)
                    for metric in METRICS:row[metric+"_1"],row[metric+"_2"]=ab[0][metric],ab[1][metric]
                    pair_rows.append(row)
        if pair_rows:
            pair_path=pair_folder/name;pd.DataFrame(pair_rows).to_parquet(pair_path,index=False)
            pair_files[str(pair_path.relative_to(cache))]=sha(pair_path)
        log(f"Metrics {len(features):,}/{len(windows):,} seven-day windows; completed grid part {name}")
    frame=pd.DataFrame(features)
    if frame.window_id.duplicated().any() or len(frame)!=len(windows):raise ValueError("Incorrect window feature grain")
    frame.to_parquet(feature_path,index=False)
    record=dict(fingerprint=fingerprint,analysis_version=VERSION,core_grid_feature_version=GRID_FEATURE_VERSION,
                n_windows=len(frame),files={feature_path.name:sha(feature_path),**pair_files})
    feature_meta.write_text(json.dumps(record,indent=2)+"\n")


def reliability_outputs(windows,features,spans,cache,bootstrap):
    joined=windows.merge(features,on="window_id",validate="one_to_one")
    pairs=joined[joined.window_number.eq(1)].merge(joined[joined.window_number.eq(2)],on="USER_ID",suffixes=("_1","_2"),validate="one_to_one")
    pairs["gap_days"]=(pairs.window_start_2-pairs.window_end_1).dt.days
    if (pairs.gap_days<0).any():raise ValueError("Overlapping seven-day windows")
    reliability=[];strata=[]
    for i,metric in enumerate(METRICS+["modd_raw","ac_var_raw"]):
        x,y=pairs[metric+"_1"].to_numpy(),pairs[metric+"_2"].to_numpy()
        result=summarize_pair(x,y,len(pairs),bootstrap,SEED+i)
        valid=np.isfinite(x)&np.isfinite(y)
        result.update(metric=metric,median_absolute_difference=result["median_abs_diff"],
                      median_signed_difference=float(np.median(y[valid]-x[valid])),
                      spearman_r=float(stats.spearmanr(x[valid],y[valid]).statistic))
        reliability.append(result)
        if metric not in METRICS:continue
        for label,lo,hi in [("pooled",0,10**9),("adjacent_0d",0,0),("gap_1_30d",1,30),("gap_over_30d",31,10**9)]:
            sub=pairs[pairs.gap_days.between(lo,hi)]
            r=summarize_pair(sub[metric+"_1"].to_numpy(),sub[metric+"_2"].to_numpy(),len(sub),bootstrap,SEED+i)
            strata.append(dict(metric=metric,stratum=label,n=r["n_users"],**r))
    pd.DataFrame(reliability).to_csv(OUT/"axis_reliability.csv",index=False)
    pd.DataFrame(strata).to_csv(OUT/"gap_stratified_reliability.csv",index=False)
    fm=json.loads((cache/"repeatability_features.metadata.json").read_text())
    length=pd.concat([pd.read_parquet(cache/name) for name in fm["files"] if name.startswith("length_pairs/")],ignore_index=True)
    if length.duplicated(["USER_ID","design","days"]).any():raise ValueError("Duplicate length-design participant rows")
    rows=[]
    for di,design in enumerate(DESIGNS):
        for k in range(1,15):
            sub=length[length.design.eq(design)&length.days.eq(k)].sort_values("USER_ID")
            if len(sub)!=len(spans):raise ValueError("Length analysis changed the common eligible cohort")
            for metric in METRICS:
                result=summarize_pair(sub[metric+"_1"].to_numpy(),sub[metric+"_2"].to_numpy(),len(spans),bootstrap,SEED+100*di+k)
                note="no consecutive calendar-day pairs" if metric=="modd" and (k==1 or design=="interleaved") else "insufficient valid dynamic pairs" if result["n_users"]<len(spans) else ""
                rows.append(dict(design=design,days=k,metric=metric,label=metric,missing_reason=note,**result))
            log(f"ICC intervals {design}, {k} days")
    pd.DataFrame(rows).to_csv(OUT/"reliability_by_window_length.csv",index=False)
    return joined,pairs


def analyze(con,windows,spans,cache,bootstrap):
    """Regenerate aggregate reliability and descriptive structure from verified features.

    Laboratory models and behavioral-context associations have separate producers;
    this stage must neither overwrite them nor claim that they were refreshed.
    """
    compute_features(windows,spans,cache)
    features=pd.read_parquet(cache/"seven_day_cgm_features.parquet")
    joined,pairs=reliability_outputs(windows,features,spans,cache,bootstrap)
    primary=joined.loc[joined.window_number.eq(1)].copy()
    if primary.USER_ID.duplicated().any():raise ValueError("Duplicate primary participant")
    helper=load_module("publication_repeatability_structure","analysis/reliability/_window_structure.py")
    pca,loadings,efa=helper.factor_analysis(primary)
    for frame,name in [(pca,"pca_variance.csv"),(loadings,"pca_loadings.csv"),(efa,"efa_three_factor_loadings.csv")]:
        frame.to_csv(OUT/name,index=False)
    rows=[]
    for left in helper.AXES:
        for right in helper.AXES:
            x=primary[left].to_numpy();y=primary[right].to_numpy()
            valid=np.isfinite(x)&np.isfinite(y)
            rows.append(dict(axis_left=left,axis_right=right,n=int(valid.sum()),
                pearson_r=float(stats.pearsonr(x[valid],y[valid]).statistic),
                spearman_r=float(stats.spearmanr(x[valid],y[valid]).statistic)))
    pd.DataFrame(rows).to_csv(OUT/"axis_correlations.csv",index=False)
    # This participant-level artifact remains under the ignored publication tree.
    primary.to_parquet(cache/"cgm_axis_features_primary.parquet",index=False)
    metadata=dict(analysis_version=VERSION,core_grid_feature_version=GRID_FEATURE_VERSION,
        n_first_week=int(len(primary)),n_second_week=int(len(pairs)),n_common_42day=int(len(spans)),
        n_structure_complete=int(primary[helper.METRIC_COLUMNS].replace([np.inf,-np.inf],np.nan).dropna().shape[0]),
        bootstrap_replicates=bootstrap,seed=SEED,primary_design="fixed_gap",
        edge_gap_days=14,icc="ICC(A,1), two-way absolute agreement, single measurement",
        structure="Complete-case first weeks; 0.5/99.5 percentile winsorization, standardized PCA and three-factor principal-axis extraction with varimax rotation; descriptive, not validation.",
        refreshed_scope=["repeatability","metric_structure","axis_correlations"],
        not_refreshed_here=["behavioral_context","laboratory_models"])
    (OUT/"repeatability_analysis.metadata.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")
    log(f"Completed reliability and structure: {len(primary):,} first weeks, {len(pairs):,} paired weeks")


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["select","grids","features","analyze","all"],default="all")
    parser.add_argument("--output-dir",type=Path,default=OUT)
    parser.add_argument("--chunk-users",type=int,default=1000)
    parser.add_argument("--bootstrap",type=int,default=500)
    parser.add_argument("--force",action="store_true")
    args=parser.parse_args();OUT=args.output_dir.resolve();cache=OUT/"cache";cache.mkdir(parents=True,exist_ok=True)
    if args.chunk_users<1 or args.bootstrap<100:raise ValueError("Positive chunk size and >=100 bootstrap replicates required")
    con=duckdb.connect(str(DB),read_only=True);con.execute("SET threads=3");con.execute("SET memory_limit='3GB'")
    con.execute("SET TimeZone='UTC'");con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory=?",[str(cache/"spill")])
    if args.stage in ["select","grids","all"]:
        windows,spans,requested,temporal_meta=select_windows(con,cache)
        selection=dict(analysis_version=VERSION,temporal_substrate=temporal_meta,
                       n_first_week=int(windows.USER_ID.nunique()),n_second_week=int(windows.window_number.eq(2).sum()),
                       n_common_42day=int(len(spans)),requested_days=int(len(requested)))
        (OUT/"repeatability_selection.json").write_text(json.dumps(selection,indent=2)+"\n")
        if args.stage=="select":con.close();return
    else:
        windows=pd.read_parquet(cache/"seven_day_windows.parquet")
        spans=pd.read_parquet(cache/"length_42day_spans.parquet")
        requested=pd.read_parquet(cache/"requested_dense_days.parquet")
    if args.stage in ["grids","all"]:
        extract_grids(con,requested,cache,args.chunk_users,args.force)
        if args.stage=="grids":con.close();return
    if args.stage in ["features","all"]:compute_features(windows,spans,cache,args.force)
    if args.stage in ["analyze","all"]:analyze(con,windows,spans,cache,args.bootstrap)
    con.close()


if __name__=="__main__":main()
