"""Helpers for single-food substrate loading and observational reliability metrics.

Provides utilities used by meal-analysis modules to load the per-food two-hour
glucose response substrate (observed-cell meals, deduplicated to one log and one
total-macro row per food, normalized by name or barcode, with isolation and CGM
coverage already enforced upstream), and to compute within-person Pearson
correlations, one-way intraclass correlation (ICC1), and the number of repeated
exposures needed to reach a target reliability.

Inputs: OUTPUTS/meals/meal_food_substrate.parquet (and its .metadata.json), produced
by analysis/meals/build_meals.py.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = OUTPUTS / 'meals'
DEFAULT_SUBSTRATE = DEFAULT_OUT / 'meal_food_substrate.parquet'
DEFAULT_DB = database_path()
DEFINITION_VERSION = 'corrected-single-food-analysis'
SUBSTRATE_VERSION = 'all-food-observed-cell-meals'


def validate_substrate(path=DEFAULT_SUBSTRATE):
    path = Path(path)
    metadata = json.loads(path.with_suffix('.metadata.json').read_text())
    expected = {'definition_version':SUBSTRATE_VERSION, 'all_food_logs_used':True,
                'noninternal_only':True, 'matched_shape_iauc_baseline':True}
    if any(metadata.get(key)!=value for key,value in expected.items()):
        raise ValueError('Food analysis requires the corrected all-food observed-cell substrate')
    if metadata.get('pre_meal_isolation_minutes',0)<120 or metadata.get('post_meal_isolation_minutes',0)<120:
        raise ValueError('Food analysis requires at least two-hour pre/post isolation')
    return metadata


def load_responses(path=DEFAULT_SUBSTRATE, identity='name'):
    validate_substrate(path)
    source = pd.read_parquet(path)
    if source.duplicated(['USER_ID','MEAL_EVENT_ID']).any():
        raise ValueError('Duplicate participant/meal food response keys')
    key = {'name':'food_key','barcode':'barcode_key'}[identity]
    source = source.loc[source.N_LOGS_IN_CLUSTER.eq(1)&source.N_MACRO_ROWS.eq(1)&source[key].notna()].copy()
    if not (source.IS_ISOLATED_2_2.fillna(False)&source.CGM_COVERAGE_OK.fillna(False)).all():
        raise ValueError('Food response row contradicts isolation/coverage provenance')
    frame = source.rename(columns={'USER_ID':'uid',key:'food','MEAL_TS':'ts','MEAL_EVENT_ID':'meal_id',
                                   'IAUC_2H':'iauc','PEAK_RISE_2H':'peak','CARBS_G':'carbs'})
    frame = frame[['uid','food','ts','meal_id','iauc','peak','carbs']]
    frame = frame.replace([np.inf,-np.inf],np.nan).dropna(subset=['iauc','peak'])
    return frame.sort_values(['uid','food','ts','meal_id'],kind='stable').reset_index(drop=True)


def pearson(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    finite=np.isfinite(a)&np.isfinite(b)
    if finite.sum()<3 or np.std(a[finite])==0 or np.std(b[finite])==0:
        return np.nan
    return float(np.corrcoef(a[finite],b[finite])[0,1])


def pairs(frame,col):
    frame=frame.sort_values(['uid','food','ts','meal_id'],kind='stable')
    position=frame.groupby(['uid','food']).cumcount()
    a=frame.loc[position.eq(0)].set_index(['uid','food'])[col]
    b=frame.loc[position.eq(1)].set_index(['uid','food'])[col]
    return pd.concat([a.rename('a'),b.rename('b')],axis=1).dropna().reset_index()


def within_person_r(halves):
    halves=halves.loc[halves.groupby('uid').a.transform('size').ge(2)]
    a=halves.a-halves.groupby('uid').a.transform('mean')
    b=halves.b-halves.groupby('uid').b.transform('mean')
    return pearson(a,b),len(halves),halves.uid.nunique()


def icc1(frame,col):
    groups=frame.groupby(['uid','food'])[col]
    counts=groups.size().to_numpy(float)
    means=groups.mean().to_numpy()
    cells,total=len(counts),counts.sum()
    if cells<2 or total<=cells:
        return np.nan
    between=np.sum(counts*(means-frame[col].mean())**2)/(cells-1)
    residual=frame[col]-groups.transform('mean')
    within=float(np.sum(residual**2)/(total-cells))
    size=(total-np.sum(counts**2)/total)/(cells-1)
    denominator=between+(size-1)*within
    return float((between-within)/denominator) if denominator>0 else np.nan


def icc1_positionwise(frame,col):
    frame=frame.sort_values(['uid','food','ts','meal_id'],kind='stable').copy()
    frame['pos']=frame.groupby(['uid','food']).cumcount()
    frame=frame.loc[frame.groupby(['uid','pos'])[col].transform('size').ge(2)].copy()
    frame['centered']=frame[col]-frame.groupby(['uid','pos'])[col].transform('mean')
    frame=frame.loc[frame.groupby(['uid','food']).centered.transform('size').ge(2)]
    return icc1(frame,'centered'),frame.groupby(['uid','food']).ngroups,frame.uid.nunique()


def repeats_needed(r,target=.7):
    if not np.isfinite(r) or r<=0:
        return np.inf
    if r>=target:
        return 1
    return int(np.ceil(target*(1-r)/(r*(1-target))))


def repeat_summary(counts):
    counts=np.asarray(counts,float)
    return {'n_pairs':len(counts),'mean_rep':float(np.mean(counts)) if len(counts) else np.nan,
            'median_rep':float(np.median(counts)) if len(counts) else np.nan,
            'p90':float(np.quantile(counts,.9)) if len(counts) else np.nan,
            **{f'f{k}':float(np.mean(counts>=k)) if len(counts) else np.nan for k in (2,5,7,12)}}


def json_safe(value):
    if isinstance(value,dict):
        return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value,(np.integer,)):
        return int(value)
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,(np.bool_,)):
        return bool(value)
    return str(value) if isinstance(value,Path) else value


def write_metadata(path, *, substrate=DEFAULT_SUBSTRATE, **details):
    document={'definition_version':DEFINITION_VERSION,'meal_substrate_metadata':validate_substrate(substrate),
              'cohort':'noninternal consenting participants; no laboratory restriction',
              'food_identity':'single log and exactly one total macro row; normalized name or exact trimmed barcode',
              'scope':'observational free-living repeats; portions/preparation and unlogged intake can differ',
              **details}
    Path(path).write_text(json.dumps(json_safe(document),indent=2,allow_nan=False)+'\n')
