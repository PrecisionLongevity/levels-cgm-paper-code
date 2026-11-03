#!/usr/bin/env python3
"""Behavior, body weight and diet in HbA1c-CGM discordance groups.

Cross-sectional associations only; no projection.

Population: canonical primary cohort (OUTPUTS/canonical_panel, no observed
diabetes-range labs). Groups: anchor HbA1c (decimal half-up) >=5.7% and mean CGM >=110 mg/dL
(both_normal, cgm_only, a1c_only, both_elevated), as in analysis/metabolic/insulin_markers.py.
Behavior (daily_exposures_unique) on dense CGM days within +/-90 local days of the HbA1c anchor:
  steps       mean daily steps over days with >0 steps, >=7 such days
  sleep_h     mean logged sleep duration (h) over nights with a sleep log and >0 min, >=7 nights
  sleep_mid   mean sleep midpoint (h) over the same nights
  ex_share    share of exposure days with logged exercise; alc_share: same for alcohol
Diet (paper isolated-meal tier, meal_shape_substrate) in the same window, >=10 meals:
  carbs_meal  carbohydrate g per meal; carb_share: mean carbohydrate share of macronutrient energy
  (4/4/9 kcal/g); fiber_meal: fiber g per meal; late_share: share of meals starting >=21:00 local
BMI from the canonical panel.

Analyses
  A  group descriptives (median, IQR, n) and coverage (n and % of the primary cohort per variable,
     overall and by group)
  B  age+sex adjusted difference vs both_normal for each variable (OLS, HC3; raw units)
  C  log(HOMA-IR) group effects vs both_normal with sequential adjustment inside fixed complete-case
     samples: sample 'bmi_steps_sleep' (age_sex, +bmi, +steps_sleep, +bmi_steps_sleep) and sample 'all'
     (additionally +diet and +all); covariates other than group/sex standardized within sample.

Inputs
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)
  OUTPUTS/daily_glucose/_cache/daily_exposures_unique.parquet (analysis/daily_glucose/build_day_panel.py)
  OUTPUTS/meals/meal_shape_substrate.parquet (analysis/meals/build_meals.py)
  OUTPUTS/metabolic/insulin_markers/discordance_group_counts.csv (analysis/metabolic/insulin_markers.py)

Outputs (OUTPUTS/metabolic/discordance_behavior/)
  descriptives.csv, coverage.csv, differences_vs_both_normal.csv, homa_ir_sequential_adjustment.csv,
  summary.json, discordance_behavior.metadata.json

Usage
  python analysis/metabolic/discordance_behavior.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from cgmpaper.lab_normalization import round_a1c  # noqa: E402

VERSION = 'discordance-behavior'
PUB = OUTPUTS
OUT = PUB / 'metabolic/discordance_behavior'
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
EXPOSURES = PUB / 'daily_glucose/_cache/daily_exposures_unique.parquet'
MEALS = PUB / 'meals/meal_shape_substrate.parquet'
S_INSULIN_MARKERS_COUNTS = PUB / 'metabolic/insulin_markers/discordance_group_counts.csv'
WINDOW_DAYS = 90
MIN_DAYS = 7
MIN_MEALS = 10
VARS = {'bmi': 'BMI (kg/m²)', 'steps': 'Daily steps', 'sleep_h': 'Sleep (h/night)', 'sleep_mid': 'Sleep midpoint (h)', 'ex_share': 'Days with logged exercise',
        'alc_share': 'Days with logged alcohol', 'carbs_meal': 'Carbohydrate per meal (g)', 'carb_share': 'Carbohydrate share of energy',
        'fiber_meal': 'Fiber per meal (g)', 'late_share': 'Meals at or after 21:00'}
GROUPS = ['both_normal', 'cgm_only', 'a1c_only', 'both_elevated']
CLAIM_NAMES = {'Concordant normal': 'both_normal', 'CGM-only elevated': 'cgm_only', 'HbA1c-only elevated': 'a1c_only',
               'Concordant elevated': 'both_elevated'}
SETS = {'age_sex': [], '+bmi': ['bmi'], '+steps_sleep': ['steps', 'sleep_h'], '+diet': ['carb_share', 'fiber_meal', 'late_share'],
        '+bmi_steps_sleep': ['bmi', 'steps', 'sleep_h'], '+all': ['bmi', 'steps', 'sleep_h', 'carb_share', 'fiber_meal', 'late_share']}
SAMPLES = [('bmi_steps_sleep', ['bmi', 'steps', 'sleep_h']), ('all', SETS['+all'])]


def design(s, cols):
    X = pd.get_dummies(s.age_bucket, prefix='age', drop_first=True, dtype=float)
    X = X.loc[:, X.sum() >= 10]
    for c in cols:
        X[c] = s[c].astype(float) if c.startswith('g_') or c == 'male' else (s[c] - s[c].mean()) / s[c].std()
    return sm.add_constant(X)


def build():
    P = pd.read_csv(PANEL, low_memory=False)
    P = P[P.no_diabetes_range == True].copy()  # noqa: E712
    P['anchor_date'] = pd.to_datetime(P.anchor_date)
    P['a1c_hi'] = round_a1c(P.a1c.to_numpy()) >= 5.7; P['cgm_hi'] = P.mean_glucose >= 110
    P['group'] = np.select([~P.a1c_hi & ~P.cgm_hi, ~P.a1c_hi & P.cgm_hi, P.a1c_hi & ~P.cgm_hi], ['both_normal', 'cgm_only', 'a1c_only'], 'both_elevated')
    P['male'] = (P.sex == 'male').astype(float)
    E = pd.read_parquet(EXPOSURES, columns=['USER_ID', 'DAY', 'TOTAL_STEPS', 'SLEEP_DURATION_MIN', 'SLEEP_MIDPOINT_HR', 'SLEEP_LOG_COUNT', 'EX_ANY', 'ALC_ANY'])
    E = E[E.USER_ID.isin(P.USER_ID)].merge(P[['USER_ID', 'anchor_date']], on='USER_ID')
    E = E[(pd.to_datetime(E.DAY) - E.anchor_date).dt.days.abs() <= WINDOW_DAYS]
    st = E[E.TOTAL_STEPS > 0].groupby('USER_ID').TOTAL_STEPS.agg(['mean', 'size'])
    sl = E[(E.SLEEP_LOG_COUNT > 0) & (E.SLEEP_DURATION_MIN > 0)].groupby('USER_ID').agg(
        sleep_h=('SLEEP_DURATION_MIN', lambda v: v.mean() / 60), sleep_mid=('SLEEP_MIDPOINT_HR', 'mean'), n=('SLEEP_DURATION_MIN', 'size'))
    B = pd.DataFrame({'steps': st.loc[st['size'] >= MIN_DAYS, 'mean']}).join(sl.loc[sl.n >= MIN_DAYS, ['sleep_h', 'sleep_mid']], how='outer')
    dd = E.groupby('USER_ID').agg(ex_share=('EX_ANY', 'mean'), alc_share=('ALC_ANY', 'mean'), cgm_days=('DAY', 'size'))
    B = B.join(dd, how='outer')
    M = pd.read_parquet(MEALS, columns=['USER_ID', 'MEAL_TS', 'tz_offset', 'MEAL_HOUR', 'CARBS_G', 'PROTEIN_G', 'FAT_G', 'FIBER_G'])
    M = M[M.USER_ID.isin(P.USER_ID)].merge(P[['USER_ID', 'anchor_date']], on='USER_ID')
    M['local_date'] = (pd.to_datetime(M.MEAL_TS, utc=True).dt.tz_localize(None) + pd.to_timedelta(M.tz_offset.fillna(0), unit='h')).dt.normalize()
    M = M[(M.local_date - M.anchor_date).dt.days.abs() <= WINDOW_DAYS].copy()
    en = 4 * M.CARBS_G.fillna(0) + 4 * M.PROTEIN_G.fillna(0) + 9 * M.FAT_G.fillna(0)
    M['carb_share'] = np.where(en > 0, 4 * M.CARBS_G.fillna(0) / en, np.nan); M['late'] = M.MEAL_HOUR >= 21
    g = M.groupby('USER_ID')
    D = pd.DataFrame({'n_meals': g.size(), 'carbs_meal': g.CARBS_G.mean(), 'carb_share': g.carb_share.mean(), 'fiber_meal': g.FIBER_G.mean(),
                      'late_share': g.late.mean()})
    D = D[D.n_meals >= MIN_MEALS].drop(columns='n_meals')
    A = P[['USER_ID', 'group', 'age_bucket', 'male', 'bmi', 'homa_ir', 'mean_glucose', 'a1c']].merge(B, left_on='USER_ID', right_index=True, how='left') \
         .merge(D, left_on='USER_ID', right_index=True, how='left')
    for gname in GROUPS[1:]:
        A['g_' + gname] = (A.group == gname).astype(float)
    return A, dict(behavior_days_in_window=int(len(E)), meals_in_window=int(len(M)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv); t0 = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    A, window_counts = build()
    GC = ['g_' + g for g in GROUPS[1:]]
    guards = dict(primary_n=int(len(A)), primary_group_sizes={g: int((A.group == g).sum()) for g in GROUPS})
    # HOMA-IR sample counts (structural check: unique participants)
    H = A[A.homa_ir > 0].copy(); H['lh'] = np.log(H.homa_ir)
    hc = H.groupby('group').agg(n=('lh', 'size'), ir=('homa_ir', lambda v: int((v >= 2.5).sum())), bmi_mean=('bmi', 'mean'), n_bmi=('bmi', 'count'))
    observed = {g: (int(hc.loc[g, 'n']), int(hc.loc[g, 'ir'])) for g in GROUPS}
    guards['homa_counts_positives'] = {g: list(v) for g, v in observed.items()}
    guards['homa_unique_participants'] = bool(A.USER_ID.is_unique and int(A.homa_ir.notna().sum()) == int(len(H)))
    s_insulin_markers = pd.read_csv(S_INSULIN_MARKERS_COUNTS).set_index('group').reindex(GROUPS)
    ours = hc.reindex(GROUPS)
    d_insulin_markers = dict(n_equal=bool((ours.n.to_numpy() == s_insulin_markers.n.to_numpy()).all()), ir_equal=bool((ours.ir.to_numpy() == s_insulin_markers.ir.to_numpy()).all()),
                n_bmi_equal=bool((ours.n_bmi.to_numpy() == s_insulin_markers.n_bmi.to_numpy()).all()),
                bmi_mean_max_abs_diff=float(np.max(np.abs(ours.bmi_mean.to_numpy() - s_insulin_markers.bmi_mean.to_numpy()))))
    guards['reproduces_insulin_markers_discordance_group_counts'] = d_insulin_markers

    # A: descriptives and coverage
    desc, cov = [], []
    for v, lab in VARS.items():
        cov.append(dict(variable=v, label=lab, group='all', n=int(A[v].notna().sum()), n_denominator=int(len(A)),
                        pct_of_denominator=float(100 * A[v].notna().mean())))
        for g in GROUPS:
            s = A.loc[A.group == g, v]
            cov.append(dict(variable=v, label=lab, group=g, n=int(s.notna().sum()), n_denominator=int(len(s)),
                            pct_of_denominator=float(100 * s.notna().mean())))
            desc.append(dict(variable=v, label=lab, group=g, n=int(s.notna().sum()), median=float(s.median()), q25=float(s.quantile(.25)),
                             q75=float(s.quantile(.75)), mean=float(s.mean())))
    for g in ['all', *GROUPS]:
        s = A if g == 'all' else A.loc[A.group == g]
        cov.append(dict(variable='homa_ir', label='HOMA-IR', group=g, n=int((s.homa_ir > 0).sum()), n_denominator=int(len(s)),
                        pct_of_denominator=float(100 * (s.homa_ir > 0).mean())))
    desc, cov = pd.DataFrame(desc), pd.DataFrame(cov)
    # B: age+sex adjusted differences vs both_normal
    diffs = []
    for v in VARS:
        s = A.dropna(subset=[v, 'age_bucket', 'male'])
        f = sm.OLS(s[v].astype(float), design(s, ['male'] + GC)).fit(cov_type='HC3')
        for gc in GC:
            lo, hi = f.conf_int().loc[gc]
            diffs.append(dict(variable=v, label=VARS[v], group=gc[2:], n=int(len(s)), n_group=int(s[gc].sum()), diff=float(f.params[gc]),
                              ci_low=float(lo), ci_high=float(hi), p=float(f.pvalues[gc])))
    diffs = pd.DataFrame(diffs)
    # C: HOMA-IR sequential adjustment in fixed complete-case samples
    att = []
    for scope, need in SAMPLES:
        Hc = H.dropna(subset=need + ['age_bucket', 'male'])
        for nm, cv in SETS.items():
            if not set(cv) <= set(need):
                continue
            f = sm.OLS(Hc.lh, design(Hc, ['male'] + GC + cv)).fit(cov_type='HC3')
            for gc in GC:
                lo, hi = f.conf_int().loc[gc]
                att.append(dict(sample=scope, n=int(len(Hc)), model=nm, group=gc[2:], pct=float(100 * np.expm1(f.params[gc])),
                                ci_low=float(100 * np.expm1(lo)), ci_high=float(100 * np.expm1(hi)), p=float(f.pvalues[gc]), n_group=int(Hc[gc].sum())))
    att = pd.DataFrame(att)
    t = att[(att['sample'] == 'bmi_steps_sleep') & (att.group == 'cgm_only')].set_index('model').pct
    guards['cgm_only_pct'] = dict(age_sex=float(t['age_sex']), bmi_steps_sleep=float(t['+bmi_steps_sleep']))

    files = {'descriptives.csv': desc, 'coverage.csv': cov, 'differences_vs_both_normal.csv': diffs, 'homa_ir_sequential_adjustment.csv': att}
    for n, fr in files.items():
        fr.to_csv(OUT / n, index=False)
    cs = att.loc[att.group == 'cgm_only'].set_index(['sample', 'model'])
    summary = dict(design='cross-sectional associations; no projection', primary_n=int(len(A)), group_sizes=guards['primary_group_sizes'],
                   homa_sample_n=int(len(H)), window_counts=window_counts,
                   coverage={r.variable: dict(n=r.n) for r in cov[cov.group == 'all'].itertuples()},
                   complete_case_samples={s: int(att.loc[att['sample'] == s, 'n'].iloc[0]) for s, _ in SAMPLES},
                   cgm_only_homa_pct={f'{s}|{m}': dict(pct=float(r.pct), ci_low=float(r.ci_low), ci_high=float(r.ci_high)) for (s, m), r in cs.iterrows()},
                   guards=guards)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    meta = dict(analysis='discordance_behavior', version=VERSION,
                design='cross-sectional associations; OLS HC3; complete-case; no imputation; no projection',
                definitions=dict(window_days=WINDOW_DAYS, min_behavior_days=MIN_DAYS, min_meals=MIN_MEALS, hba1c_cutoff=5.7, mean_cgm_cutoff=110,
                                 homa_threshold=2.5, energy='4/4/9 kcal per g carbohydrate/protein/fat', adjustment_sets=SETS,
                                 complete_case_samples=dict(SAMPLES)),
                guards=guards,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'discordance_behavior.metadata.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    print(json.dumps(guards, indent=1, default=float))


if __name__ == '__main__':
    main()
