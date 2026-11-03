#!/usr/bin/env python3
"""Re-estimates whole-cohort analyses after excluding participants with a diabetes-range lab value.

Whole-cohort analyses (paired-week repeatability; within-person exercise and sleep contrasts)
cannot apply the laboratory-cohort exclusion because most participants have no laboratory
values. This sensitivity analysis removes every participant with any accepted diabetes-range
value (HbA1c >=6.5% after decimal half-up rounding, or fasting glucose >=126 mg/dL) from the
shared laboratory normalization, then re-estimates each quantity with the producers' own
estimators.

  repeatability  ICC(A,1) of mean glucose across two seven-day windows (analysis/reliability/repeatability.py icc_a1)
  exercise       same-day mean glucose, any logged exercise vs none, person FE
  sleep          mean glucose, <6 h vs 7-<8 h, person FE, >=10 days per person
  food_decomp    crossed person/food/person-by-food/occasion shares of 2-h iAUC, named foods with
                 a minimum participant count (analysis/meals/food_variance.py henderson3)
  averaging      same-food split-half correlation, k=1 and k=7 per half (analysis/meals/split_half_reliability.py estimator)
The day-level substrate already excludes these participants by default (analysis/daily_glucose/build_day_panel.py),
so the first three are expected to be unchanged; the meal substrates do not.

Inputs: labs table via LEVELS_DB; OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (build_panel.py);
  OUTPUTS/reliability/cache/seven_day_windows.parquet and seven_day_cgm_features.parquet (repeatability.py);
  OUTPUTS/meals/food_general_vs_personal_variance_components.csv (food_variance.py);
  OUTPUTS/meals/same_food_averaging_observed/observed_averaging_curve.csv (same_food_averaging.py);
  daily glucose and exposures caches (build_day_panel.py).
Outputs: OUTPUTS/validation/diabetes_range_exclusion_sensitivity/exclusion_sensitivity.csv,
  summary.json (aggregate, not participant-level).

Usage: python analysis/validation/diabetes_range_exclusion.py
"""
from __future__ import annotations

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
from cgmpaper.lab_normalization import normalize_labs, round_a1c  # noqa: E402
import _day_panel as tp  # noqa: E402

VERSION = 'diabetes-range-exclusion-sensitivity'
PUB = OUTPUTS
OUT = PUB / 'validation/diabetes_range_exclusion_sensitivity'
DB = database_path()
REP = PUB / 'reliability'
GLUC, EXPOSURES = tp.CACHE / 'gluc_local_day.parquet', tp.EXPOSURES


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def diabetes_range_users(con):
    raw = con.execute("SELECT USER_ID, TYPE, DRAW_DATE, VALUE, UNIT FROM labs WHERE TYPE IN ('a1c','fasting_glucose')").fetchdf()
    daily, _, _ = normalize_labs(raw)
    a = daily.loc[daily.TYPE == 'a1c']; g = daily.loc[daily.TYPE == 'fasting_glucose']
    hi = set(a.loc[round_a1c(a.value.to_numpy()) >= 6.5, 'USER_ID']) | set(g.loc[g.value >= 126, 'USER_ID'])
    return hi


def main():
    t0 = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DB), read_only=True) as con:
        con.execute('PRAGMA threads=4')
        excl = diabetes_range_users(con)
        ex_q = f"""
          WITH gl AS (SELECT USER_ID, LOCAL_DAY, DOW, MONTH, MEAN FROM read_parquet('{GLUC}') WHERE N_READINGS >= 230)
          SELECT gl.*, COALESCE(de.EX_ANY, FALSE) AS EX_ANY FROM gl JOIN '{EXPOSURES}' de
            ON gl.USER_ID = de.USER_ID AND gl.LOCAL_DAY = CAST(de.DAY AS DATE)"""
        exd = con.execute(ex_q).fetchdf()
        sl_q = f"""
          WITH g AS (SELECT USER_ID, LOCAL_DAY, DOW, MONTH, MEAN FROM '{GLUC}' WHERE N_READINGS >= 230),
          s AS (SELECT USER_ID, CAST(DAY AS DATE) AS LOCAL_DAY, SLEEP_DURATION_MIN FROM '{EXPOSURES}' WHERE SLEEP_DURATION_MIN IS NOT NULL)
          SELECT g.*, s.SLEEP_DURATION_MIN FROM g JOIN s USING (USER_ID, LOCAL_DAY)"""
        sld = con.execute(sl_q).fetchdf()
    panel = pd.read_csv(PUB / 'canonical_panel/canonical_lab_cgm_panel.csv', usecols=['USER_ID', 'no_diabetes_range'], low_memory=False)
    pflag = set(panel.loc[panel.no_diabetes_range == False, 'USER_ID'])  # noqa: E712
    guards = dict(excluded_users=len(excl), panel_diabetes_range=len(pflag), panel_diabetes_range_captured=len(pflag & excl))
    # exercise: participants with >= 5 local days
    exd['LOCAL_DAY'] = pd.to_datetime(exd.LOCAL_DAY); exd = exd.sort_values(['USER_ID', 'LOCAL_DAY']).reset_index(drop=True)
    exd = exd.loc[exd.groupby('USER_ID').LOCAL_DAY.transform('size') >= 5].copy()
    # sleep: nights of 2-14 h, duration bins
    sld = sld.loc[sld.SLEEP_DURATION_MIN.between(120, 840)].copy()
    sld['SLEEP_HR'] = sld.SLEEP_DURATION_MIN / 60.0
    sld['DUR_BIN'] = pd.cut(sld.SLEEP_HR, bins=[0, 6, 7, 8, 9, 99], labels=['<6', '6-7', '7-8', '8-9', '>9'], right=False)
    dum = pd.get_dummies(sld.DUR_BIN, prefix='bin'); bincols = [c for c in dum.columns if c != 'bin_7-8']
    sld = pd.concat([sld, dum], axis=1)
    # repeatability: first and second seven-day windows (analysis/reliability/repeatability.py)
    s_repeatability = load('repeatability_module', 'analysis/reliability/repeatability.py')
    W = pd.read_parquet(REP / 'cache/seven_day_windows.parquet'); Fe = pd.read_parquet(REP / 'cache/seven_day_cgm_features.parquet')
    J = W.merge(Fe, on='window_id', validate='one_to_one')
    pairs = J[J.window_number.eq(1)].merge(J[J.window_number.eq(2)], on='USER_ID', suffixes=('_1', '_2'), validate='one_to_one')

    def icc(pr):
        x, y = pr.mean_glucose_1.to_numpy(float), pr.mean_glucose_2.to_numpy(float); ok = np.isfinite(x) & np.isfinite(y)
        return dict(estimate=s_repeatability.icc_a1(x[ok], y[ok]), n=int(ok.sum()))

    def exercise(d):
        f = tp.fixed_effects(d, 'MEAN', ['EX_ANY'], min_days=5).set_index('term').loc['EX_ANY']
        return dict(estimate=float(f.beta), ci_low=float(f.ci_low), ci_high=float(f.ci_high), n=int(f.n_obs), n_users=int(f.n_users))

    def sleep(d):
        f = tp.fixed_effects(d, 'MEAN', bincols, min_days=10).set_index('term').loc['bin_<6']
        return dict(estimate=float(f.beta), ci_low=float(f.ci_low), ci_high=float(f.ci_high), n=int(f.n_obs), n_users=int(f.n_users))

    full = dict(repeatability=icc(pairs), exercise=exercise(exd), sleep=sleep(sld))
    sens = dict(repeatability=icc(pairs.loc[~pairs.USER_ID.isin(excl)]), exercise=exercise(exd.loc[~exd.USER_ID.isin(excl)]),
                sleep=sleep(sld.loc[~sld.USER_ID.isin(excl)]))
    removed = dict(repeatability=int(pairs.USER_ID.isin(excl).sum()), exercise=int(exd.loc[exd.USER_ID.isin(excl)].USER_ID.nunique()),
                   sleep=int(sld.loc[sld.USER_ID.isin(excl)].USER_ID.nunique()))
    # food analyses (meal substrates include diabetes-range participants)
    from analysis.meals import _food_analysis as shared
    m_food_variance = load('food_variance_module', 'analysis/meals/food_variance.py')
    s_split_half = load('split_half_module', 'analysis/meals/split_half_reliability.py')
    fr = shared.load_responses(identity='name').replace([np.inf, -np.inf], np.nan).dropna(subset=['iauc'])

    def decomp(frame):
        cnt = frame.groupby('food').uid.nunique(); ref = frame[frame.food.isin(cnt.index[cnt.ge(50)])]
        h = m_food_variance.henderson3(ref, 'iauc', n_trace=m_food_variance.N_TRACE, seed=m_food_variance.SEED)
        return dict(estimate=float(h['pct_PF']), pct_P=float(h['pct_P']), pct_F=float(h['pct_F']), pct_E=float(h['pct_E']),
                    n=int(len(ref)), n_users=int(ref.uid.nunique()))
    comp = pd.read_csv(PUB / 'meals/food_general_vs_personal_variance_components.csv')
    c0 = comp[(comp.identity == 'name') & (comp.min_users == 50) & (comp.response == 'IAUC_2H')].iloc[0]
    fa = shared.load_responses()

    def averaging(frame):
        c = s_split_half.evaluate(frame, 7, s_split_half.N_BOOT, s_split_half.SEED)
        r1 = c[(c.split == 'interleaved') & (c.k == 1)].iloc[0]; r7 = c[(c.split == 'interleaved') & (c.k == 7)].iloc[0]
        return dict(estimate=float(r7.r_empirical), ci_low=float(r7.ci_low), ci_high=float(r7.ci_high), n=int(r7.n_cells),
                    n_users=int(r7.n_users), r_k1=float(r1.r_empirical))
    obs = pd.read_csv(PUB / 'meals/same_food_averaging_observed/observed_averaging_curve.csv')
    o7 = obs[(obs.split == 'interleaved') & (obs.k == 7)].iloc[0]
    full['food_decomp'], full['averaging'] = decomp(fr), averaging(fa)
    guards['food_decomp_reproduced'] = all(abs(full['food_decomp'][k] - float(c0[k2])) < 1e-6 for k, k2 in
                                           [('estimate', 'pct_PF'), ('pct_P', 'pct_P'), ('pct_F', 'pct_F'), ('pct_E', 'pct_E')])
    guards['averaging_reproduced'] = abs(full['averaging']['estimate'] - float(o7.r_empirical)) < 1e-12 and full['averaging']['n'] == int(o7.n_cells)
    if not (guards['food_decomp_reproduced'] and guards['averaging_reproduced']):
        raise SystemExit(f'GUARD FAILED (food): {guards} {full}')
    sens['food_decomp'], sens['averaging'] = decomp(fr[~fr.uid.isin(excl)]), averaging(fa[~fa.uid.isin(excl)])
    removed['food_decomp'] = int(fr.loc[fr.uid.isin(excl)].uid.nunique()); removed['averaging'] = int(fa.loc[fa.uid.isin(excl)].uid.nunique())
    rows = [dict(analysis=k, sample=s, **v) for k in full for s, v in (('full', full[k]), ('excluding_diabetes_range', sens[k]))]
    T = pd.DataFrame(rows); T.to_csv(OUT / 'exclusion_sensitivity.csv', index=False)
    summary = dict(removed_participants=removed, guards=guards)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    meta = dict(analysis='diabetes_range_exclusion_sensitivity', version=VERSION, status='passed', guards=guards,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'diabetes_range_exclusion_sensitivity.metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(T.to_string(index=False)); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
