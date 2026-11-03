#!/usr/bin/env python3
"""Selects the three example participants of Fig. 6c by a stated rule.

Descriptive only: the examples illustrate how the adjusted carbohydrate slope is measured; they are not evidence.
They are chosen by the stated rule below, never by appearance, and each slope is refit from the meal rows before it
is written.

Rule
  Eligible participants: the broad_mixed adjusted carbohydrate slopes (acs) of analysis/meals/meal_populations.py
  (>=30 isolated meal clusters, carbohydrate SD >=5 g, full rank), excluding every participant with any accepted
  diabetes-range laboratory value (analysis/validation/diabetes_range_exclusion.py diabetes_range_users()).
  For each of the 10th, 50th and 90th percentile of the eligible slopes, participants are ranked by
  |acs - target| / SD(acs), ties broken by USER_ID; the nearest is the example and the next five are candidates 2-6.
  Fig. 6c draws the rank-1 participant of each percentile: every fitted meal (carbohydrate, peak rise) with the
  adjusted slope drawn at the participant's mean fiber, fat and protein.

Checks: each slope is refit exactly from the meal rows (analysis/meals/_carbohydrate_analysis.py fit_slope); the
selection is run twice and must be identical; the public manifest holds no participant id.

Inputs: database tables via LEVELS_DB (diabetes-range laboratory values);
OUTPUTS/meal_populations/_cache/broad_mixed_participants.parquet (analysis/meals/meal_populations.py);
OUTPUTS/meals/meal_food_substrate.parquet (analysis/meals/build_meals.py).

Outputs: OUTPUTS/exemplars
  restricted/exemplar_selection_restricted.json   participant ids and meal values; stays private
  exemplar_manifest.json                          rule, rank and de-identified values (read by figures/fig6_meals.py)
  exemplar_selection.metadata.json

Usage: python analysis/exemplars/exemplar_candidates.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'analysis/daily_glucose'))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from analysis.meals import _carbohydrate_analysis as carb  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


s_diabetes_exclusion = _load('diabetes_exclusion_for_exemplar_candidates', 'analysis/validation/diabetes_range_exclusion.py')

VERSION = 'exemplar-selection'
PUB = OUTPUTS
OUT = PUB / 'exemplars'
RESTRICTED = OUT / 'restricted'
DB = database_path()
FOOD_SUB = PUB / 'meals/meal_food_substrate.parquet'
QSP = PUB / 'meal_populations'
BROAD = QSP / '_cache/broad_mixed_participants.parquet'

N_CAND = 6                 # exemplar + next five
PERCENTILES = (10, 50, 90)


class GuardError(SystemExit):
    pass


def guard(ok, msg):
    if not ok:
        raise GuardError(f'GUARD FAILED: {msg}')


def rank_units(df, metrics, targets, keys):
    """Standardized Euclidean distance to targets; SD over the eligible pool; stable key ties."""
    d = df.copy()
    scales = {m: float(d[m].std(ddof=1)) for m in metrics}
    guard(all(np.isfinite(v) and v > 0 for v in scales.values()), f'degenerate scales {scales}')
    d['distance'] = np.sqrt(sum(((d[m] - targets[m]) / scales[m]) ** 2 for m in metrics))
    d = d.sort_values(['distance', *keys], kind='mergesort').reset_index(drop=True)
    return d, scales


def select_carb_slope(I, guards):
    B = pd.read_parquet(BROAD)
    guards['carb_slope_broad_mixed'] = dict(n_users=int(len(B)), median_slope=float(B.acs.median()))
    E = B.loc[~B.USER_ID.isin(I['excl']) & B.n_meals.ge(carb.MIN_MEALS)].copy()
    meals = carb.load_macro_meals(FOOD_SUB)
    groups, targets = {}, {}
    for q in PERCENTILES:
        t = float(np.percentile(E.acs, q)); targets[f'p{q}'] = t
        R, scales = rank_units(E, ['acs'], {'acs': t}, ['USER_ID'])
        out = []
        for u in R.head(N_CAND).itertuples():
            m = meals.loc[meals.USER_ID.eq(u.USER_ID)]
            f = carb.fit_slope(m)
            guard(f is not None and f['n_meals'] == u.n_meals and abs(f['acs'] - u.acs) <= 1e-9, 'carbohydrate slope refit')
            x = np.column_stack([np.ones(len(m)), m[carb.MACROS].to_numpy(float)])
            beta = np.linalg.lstsq(x, m.PEAK_RISE_2H.to_numpy(float), rcond=None)[0]
            out.append(dict(rank=len(out) + 1, distance=float(u.distance), metrics=dict(acs=float(u.acs), n_meals=int(u.n_meals)),
                            restricted=dict(USER_ID=u.USER_ID),
                            trace=dict(carbs_g=[round(float(v), 2) for v in m.CARBS_G], peak_rise=[round(float(v), 3) for v in m.PEAK_RISE_2H],
                                       coef=dict(zip(['intercept', *carb.MACROS], [float(v) for v in beta])),
                                       intercept_at_mean_macros=float(f['intercept_at_mean_macros']))))
        groups[f'p{q}'] = out
    return dict(panel='carb_slope_panel', population='analysis/meals/meal_populations.py broad_mixed slopes (>=30 isolated meal clusters, carbohydrate SD >=5 g, full rank), '
                                         'excluding participants with a diabetes-range laboratory value',
                rule='participants nearest the 10th, 50th and 90th percentile of the eligible adjusted carbohydrate slopes (acs); '
                     'distance |acs - target| / SD(acs); shown: every fitted meal (carbohydrate, peak rise) with the adjusted slope drawn '
                     'at the participant\'s mean fiber, fat and protein',
                metrics=['acs'], targets=targets, scales={'acs': float(E.acs.std(ddof=1))}, n_eligible_units=int(len(E)),
                n_eligible_participants=int(len(E)), n_screened=N_CAND * 3, n_rejected_trace=0, groups=groups)


def fingerprint(res):
    return hashlib.sha256(json.dumps(res, sort_keys=True, default=str).encode()).hexdigest()


def public(res):
    """Drop restricted keys; everything else is de-identified (no ids, dates or food identities beyond generic labels)."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != 'restricted'}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return strip(res)


def leak_check(obj, ids, days):
    s = json.dumps(obj, default=str)
    guard(not any(i in s for i in ids), 'participant id in public manifest')
    guard(not any(d in s for d in days), 'date in public manifest')


def select(excl, guards):
    return {'carb_slope_panel': select_carb_slope(dict(excl=excl), guards)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv); t0 = time.monotonic()
    guards = {}
    with duckdb.connect(str(DB), read_only=True) as con:
        excl = s_diabetes_exclusion.diabetes_range_users(con)
    guards['excluded_users'] = len(excl)
    runs = []
    for k in range(2):
        print(f'selection run {k + 1}', flush=True)
        g = dict(guards); runs.append((select(excl, g), g))
    (res, guards), (res_b, _) = runs
    guards['reselection_identical'] = fingerprint(res) == fingerprint(res_b)
    guard(guards['reselection_identical'], 're-selection differs between runs')
    per = {q: len(c) for q, c in res['carb_slope_panel']['groups'].items()}
    guard(all(n == N_CAND for n in per.values()), f'fewer than {N_CAND} candidates {per}')
    guards['candidates_per_group'] = per
    OUT.mkdir(parents=True, exist_ok=True); RESTRICTED.mkdir(parents=True, exist_ok=True)
    rpath = RESTRICTED / 'exemplar_selection_restricted.json'
    rpath.write_text(json.dumps(dict(version=VERSION, note='RESTRICTED: participant ids and meal values; never track or share',
                                     panels=res), indent=1, default=str) + '\n')
    ids = sorted({c['restricted']['USER_ID'] for g in res['carb_slope_panel']['groups'].values() for c in g})
    pub = dict(version=VERSION,
               note='De-identified values of the example participants of Fig. 6c, selected by rule; no participant ids or '
                    'dates. Candidate 1 is the selected example; 2-6 are the next in order.',
               panels=public(res))
    leak_check(pub, ids, [])
    mpath = OUT / 'exemplar_manifest.json'; mpath.write_text(json.dumps(pub, indent=1, default=str) + '\n')
    stat = DB.resolve().stat()
    meta = dict(analysis='exemplar_selection', version=VERSION, status='passed',
                design='descriptive exemplar selection by stated rule; no inference', guards=guards,
                database=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns),
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'exemplar_selection.metadata.json').write_text(json.dumps(meta, indent=1, default=str) + '\n')
    print(json.dumps(dict(guards=guards), indent=1, default=str))


if __name__ == '__main__':
    main()
