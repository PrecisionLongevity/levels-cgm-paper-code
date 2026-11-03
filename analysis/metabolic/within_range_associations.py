#!/usr/bin/env python3
"""Within-HbA1c-range associations of each CGM and meal measure. EXPLORATORY.

For each measure, the association of the measure with HbA1c and with fasting insulin
inside each HbA1c range of the primary cohort. Cross-sectional associations only; no projection.

Population: canonical primary cohort (canonical_lab_cgm_panel.csv, no_diabetes_range == True).
Anchor HbA1c rounded decimal half-up (cgmpaper.lab_normalization.round_a1c, applied
by publication_panels.load_panel) before thresholding.
  strata  lt57   HbA1c < 5.7 %
          57_64  HbA1c 5.7-6.4 % (primary cohort; no observed diabetes-range lab)
Measures (FDR family)
  mean_glucose     average (mean) CGM glucose in the canonical window
  night_minus_day  mean 00:00-05:00 minus mean 11:00-15:00 local (analysis/metabolic/insulin_markers.py definition)
  sd_glucose, cv_glucose, modd                    canonical panel
  all_frequency, all_size, all_clearance          rise-based spike facets (analysis/meals/spike_facets.py
                   spike_facets_participants.parquet; the inputs of insulin_markers.py/hba1c_stratified_spikes.py):
                   spikes per day, mean rise size, half-recovery minutes
  carb_slope       adjusted carbohydrate slope (acs), analysis/meals/meal_populations.py window_mixed participant
                   cache (meals within +/-90 local days of the anchor; the population of insulin_markers.py
                   section D)
  peak_rise        per-person mean 2-h meal peak rise, logged meals within +/-90 days of the anchor,
                   >=10 meals (insulin_markers.py meal_responses)
Companion (reported, outside the FDR family)
  rise_beyond_content  per-person mean residual peak rise after carbohydrate, protein, fat and
                   fiber (insulin_markers.py definition, residualized in the primary cohort): peak rise at
                   the same meal content.
Outcomes: HbA1c (percentage points per SD of the measure) and log fasting insulin (% difference
per SD). Adjustment: age bucket + sex + average glucose; the average-glucose row is adjusted for age
and sex only. The measure and continuous covariates are standardized within the stratum sample
(hba1c_stratified_spikes.py fit(), reused unchanged). OLS with HC3 standard errors.
Multiplicity: Benjamini-Hochberg across the whole column (10 measures x 2 strata = 20 tests),
separately per outcome; interaction p values BH-adjusted across the 10 measures per outcome.
Range interaction (per measure and outcome): pooled lt57 + 57_64; measure standardized on the
pooled SD; common age/sex/average-glucose terms; stratum main effect; measure x stratum term (HC3).
Also reported: difference of the two within-stratum per-SD slopes with an independent-samples
z test (the strata are disjoint).

TRUNCATION: within-range HbA1c regressions select on the outcome (HbA1c is restricted to
< 5.7 or 5.7-6.4 %). Slopes are attenuated relative to the full range and can be biased when the
measurement relates to HbA1c determinants; they describe association inside a range, not a
population slope. Fasting-insulin regressions are not truncated on their outcome.

Structural consistency checks performed before new estimates are computed: strata sample sizes
match hba1c_stratified_spikes.py's strata_descriptives; the primary cohort excludes diabetes-range
HbA1c; logged-meal counts and the carbohydrate-slope population match insulin_markers.py and
meal_populations.py; insulin_markers.py fasting-insulin rows and hba1c_stratified_spikes.py model M1
rows are recomputed on this frame and compared to those modules' saved outputs.

Inputs
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)
  OUTPUTS/meals/spike_facets/spike_facets_participants.parquet (analysis/meals/spike_facets.py)
  OUTPUTS/meal_populations/_cache/window_mixed_participants.parquet and
    window_mixed_correlations.csv (analysis/meals/meal_populations.py)
  OUTPUTS/metabolic/insulin_markers/ (analysis/metabolic/insulin_markers.py)
  OUTPUTS/metabolic/hba1c_stratified_spike_recovery/ (analysis/metabolic/hba1c_stratified_spikes.py)

Outputs (OUTPUTS/metabolic/within_range_associations/, participant-level inputs are not retained)
  within_range_associations.csv, range_interactions.csv, strata_descriptives.csv,
  guard_stratified_spikes_reproduction.csv, summary.json, within_range_associations.metadata.json

Usage
  python analysis/metabolic/within_range_associations.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from cgmpaper.publication_panels import load_panel  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


s_stratified_spikes = _load('stratified_spikes_module', 'analysis/metabolic/hba1c_stratified_spikes.py')
s_insulin_markers = s_stratified_spikes.s_insulin_markers

VERSION = 'within-range-associations'
PUB = OUTPUTS
MP = PUB / 'metabolic'
OUT = MP / 'within_range_associations'
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
FACETS = PUB / 'meals/spike_facets/spike_facets_participants.parquet'
FACETS_META = PUB / 'meals/spike_facets/spike_facets.metadata.json'
MEALS = s_insulin_markers.MEALS
FPCA = s_insulin_markers.FPCA
CARB = PUB / 'meal_populations/_cache/window_mixed_participants.parquet'
CARB_CORR = PUB / 'meal_populations/window_mixed_correlations.csv'
D_INSULIN_MARKERS = MP / 'insulin_markers'
D_STRATIFIED_SPIKES = MP / 'hba1c_stratified_spike_recovery'
SCRIPT_INSULIN_MARKERS = ROOT / 'analysis/metabolic/insulin_markers.py'
SCRIPT_STRATIFIED_SPIKES = ROOT / 'analysis/metabolic/hba1c_stratified_spikes.py'
TOL = 1e-9
Z975 = float(stats.norm.ppf(0.975))
STRATA = ['lt57', '57_64']
STRATUM_LABEL = {'lt57': 'HbA1c <5.7%', '57_64': 'HbA1c 5.7-6.4%'}
BASE = ['age', 'male']
# key, label, adjustment, measure key
MEASURES = [('mean_glucose', 'Average glucose', BASE, 'mean'),
            ('night_minus_day', 'Overnight minus midday (00-05 minus 11-15)', BASE + ['mean_glucose'], 'nmd'),
            ('sd_glucose', 'SD', BASE + ['mean_glucose'], 'sd'),
            ('cv_glucose', 'CV', BASE + ['mean_glucose'], 'cv'),
            ('modd', 'MODD', BASE + ['mean_glucose'], 'modd'),
            ('all_frequency', 'Spike frequency', BASE + ['mean_glucose'], 'freq'),
            ('all_size', 'Spike size', BASE + ['mean_glucose'], 'size'),
            ('all_clearance', 'Spike recovery (half-recovery minutes)', BASE + ['mean_glucose'], 'rec'),
            ('carb_slope', 'Carbohydrate slope (adjusted)', BASE + ['mean_glucose'], 'carb'),
            ('peak_rise', 'Mean meal peak rise', BASE + ['mean_glucose'], 'food')]
COMPANION = [('rise_beyond_content', 'Mean meal peak rise at the same meal content (residual)', BASE + ['mean_glucose'], 'food')]
OUTCOMES = {'a1c': ('hba1c_points_per_sd', False), 'fasting_insulin': ('percent_per_sd', True)}
GUARD_STRATIFIED_SPIKES = ['all_clearance', 'all_size', 'all_frequency', 'peak_rise']
TRUNCATION = ('Within-range HbA1c regressions are truncated on the outcome: the sample is selected on HbA1c (<5.7% or '
              '5.7-6.4%), which restricts outcome variance, attenuates slopes relative to the full range and can bias them '
              'when the measurement relates to HbA1c determinants. They describe association inside a range, not a population '
              'slope. Fasting-insulin regressions are not truncated on their outcome.')


def fail(msg):
    raise SystemExit(f'GUARD FAILED: {msg}')


def outcome_frame(S, y, log):
    d = S.copy()
    if log:
        d = d.loc[d[y] > 0].copy(); d[y] = np.log(d[y])
    return d


def fit_raw(d, x, adj, y):
    """analysis/metabolic/hba1c_stratified_spikes.py fit() on the model scale (log for insulin), plus the model-scale SE."""
    r = s_stratified_spikes.fit(d, x, [c for c in adj if c != x], y)
    if r:
        r['se'] = (r['ci_high'] - r['ci_low']) / (2 * Z975)
    return r


def to_pct(v):
    return float(100 * (np.exp(v) - 1))


def interaction(A, x, adj, y, log):
    """Pooled lt57 + 57_64: measure (pooled SD) x stratum, common covariates, HC3."""
    covs = [c for c in adj if c not in ('age', x)]
    I = outcome_frame(A.loc[A.stratum.isin(STRATA)], y, log).dropna(subset=[x, y] + covs).reset_index(drop=True)
    X = pd.DataFrame({'z': (I[x] - I[x].mean()) / I[x].std(), 'g57': (I.stratum == '57_64').astype(float)})
    X['z_x_g57'] = X.z * X.g57
    for c in covs:
        X[c] = (I[c] - I[c].mean()) / I[c].std() if c != 'male' else I[c].astype(float)
    ages = pd.get_dummies(I.age_bucket, prefix='age', drop_first=True, dtype=float); X = X.join(ages.loc[:, ages.sum() >= 10])
    m = sm.OLS(I[y].to_numpy(float), sm.add_constant(X)).fit(cov_type='HC3'); ci = m.conf_int()
    f = to_pct if log else float
    return dict(n_pooled=int(len(I)), n_lt57=int((I.stratum == 'lt57').sum()), n_57_64=int((I.stratum == '57_64').sum()),
                slope_lt57_pooled_sd=f(m.params['z']), interaction=f(m.params['z_x_g57']),
                interaction_ci_low=f(ci.loc['z_x_g57', 0]), interaction_ci_high=f(ci.loc['z_x_g57', 1]),
                p_interaction=float(m.pvalues['z_x_g57']))


def load(guards):
    panel = load_panel(PANEL, cohort='all_lab_linked', required=['peaks_per_day'], expected_window='symmetric')
    raw = pd.read_csv(PANEL, low_memory=False, usecols=['USER_ID', 'homa_ir', 'fasting_insulin', 'age_bucket', 'sex', 'bmi', 'mean_glucose',
                                                         'mean_night', 'mean_midday', 'sd_glucose', 'cv_glucose', 'modd'])
    P = panel[['USER_ID', 'a1c', 'no_diabetes_range', 'anchor_date']].merge(raw, on='USER_ID', validate='one_to_one')
    P = P.loc[P.no_diabetes_range == True].reset_index(drop=True)  # noqa: E712
    P['male'] = (P.sex == 'male').astype(float)
    P['night_minus_day'] = P.mean_night - P.mean_midday
    P = P.merge(pd.read_parquet(FACETS, columns=['USER_ID', 'all_frequency', 'all_size', 'all_clearance']), on='USER_ID', how='left')
    guards['primary_n'] = int(len(P))
    P['stratum'] = np.where(P.a1c >= 5.7, '57_64', 'lt57')
    guards['max_hba1c_57_64'] = float(P.loc[P.stratum == '57_64', 'a1c'].max())
    if guards['max_hba1c_57_64'] >= 6.5:
        fail('diabetes-range HbA1c inside the primary cohort')
    # logged-meal responses (analysis/metabolic/insulin_markers.py definition, primary cohort)
    R, counts = s_insulin_markers.meal_responses(P)
    s_insulin_markers_c = json.loads((D_INSULIN_MARKERS / 'summary.json').read_text())['samples']
    guards['meal_counts'] = counts
    guards['meal_counts_match_insulin_markers'] = all(counts[k] == s_insulin_markers_c[k] for k in counts)
    if not guards['meal_counts_match_insulin_markers']:
        fail(f'meal counts {counts} vs {s_insulin_markers_c}')
    P = P.merge(R[['USER_ID', 'peak_rise', 'rise_beyond_content']], on='USER_ID', how='left')
    # carbohydrate slope (analysis/meals/meal_populations.py window_mixed cache)
    W = pd.read_parquet(CARB)
    pub = pd.read_csv(CARB_CORR).set_index('variable')
    s = W.dropna(subset=['homa_ir'])
    r = float(np.corrcoef(s.acs, s.homa_ir)[0, 1])
    guards['carb_n'] = int(len(W)); guards['carb_all_in_primary'] = bool(W.USER_ID.isin(P.USER_ID).all() and W.USER_ID.is_unique)
    guards['carb_homa_pearson_diff'] = abs(r - float(pub.loc['homa_ir', 'pearson_r']))
    if not (guards['carb_all_in_primary'] and guards['carb_homa_pearson_diff'] < TOL):
        fail('carbohydrate population does not reproduce meal_populations.py/insulin_markers.py')
    P = P.merge(W[['USER_ID', 'acs']].rename(columns={'acs': 'carb_slope'}), on='USER_ID', how='left')
    return P


def guard_insulin_markers(P, guards):
    A = pd.read_csv(D_INSULIN_MARKERS / 'cgm_measures_by_marker.csv'); B = pd.read_csv(D_INSULIN_MARKERS / 'meal_responses_by_marker.csv')
    cgm = [k for k, *_ in MEASURES if k in s_insulin_markers.CGM_MEASURES]
    new = pd.DataFrame(s_insulin_markers.family(P, cgm, 'fasting_insulin', True, {k: s_insulin_markers.ADJ[k] for k in ('age_sex', 'mean_glucose')}))
    newB = pd.DataFrame(s_insulin_markers.family(P.dropna(subset=['peak_rise']), ['peak_rise', 'rise_beyond_content'], 'fasting_insulin', True,
                                    {k: s_insulin_markers.ADJ[k] for k in ('age_sex', 'mean_glucose', 'mean_glucose_bmi')}))
    worst, checked = 0.0, 0
    for T, N, keys in [(A, new, [(k, 'age_sex' if k == 'mean_glucose' else 'mean_glucose') for k in cgm]),
                       (B, newB, [('peak_rise', 'mean_glucose'), ('rise_beyond_content', 'mean_glucose')])]:
        for meas, adj in keys:
            sel = lambda F: F.loc[(F.marker == 'fasting_insulin') & (F.adjustment == adj) & (F.measure == meas)]  # noqa: E731
            a, b = sel(T), sel(N)
            if len(a) != 1 or len(b) != 1 or int(a.n.iloc[0]) != int(b.n.iloc[0]):
                fail(f'insulin_markers.py row {meas}/{adj} missing or n differs')
            worst = max(worst, *(abs(float(a[c].iloc[0]) - float(b[c].iloc[0])) for c in ('estimate', 'ci_low', 'ci_high', 'p')))
            checked += 1
    guards['s_insulin_markers_rows_checked'] = checked; guards['s_insulin_markers_max_abs_diff'] = worst
    if worst > TOL:
        fail(f'insulin_markers.py fasting-insulin rows not reproduced (max diff {worst})')


def guard_stratified_spikes(P, guards):
    """Runs before any new estimate: the M1 rows of hba1c_stratified_spikes.py recomputed with this producer's stratum fits."""
    adj = {x: a for x, _, a, _ in MEASURES}
    T = pd.DataFrame([dict(stratum=st, outcome=y, measure=x, **{k: (to_pct(v) if log and k in ('estimate', 'ci_low', 'ci_high') else v)
                                                              for k, v in fit_raw(outcome_frame(P.loc[P.stratum == st], y, log), x, adj[x], y).items()})
                      for y, (_, log) in OUTCOMES.items() for st in STRATA for x in GUARD_STRATIFIED_SPIKES])
    St = pd.read_csv(D_STRATIFIED_SPIKES / 'stratified_associations.csv')
    desc = pd.read_csv(D_STRATIFIED_SPIKES / 'strata_descriptives.csv').set_index('stratum')
    for st in STRATA:
        m = P.stratum == st
        obs = dict(n=int(m.sum()), n_recovery=int(P.loc[m, 'all_clearance'].notna().sum()), n_meal_responses=int(P.loc[m, 'peak_rise'].notna().sum()))
        exp = {k: int(desc.loc[st, k]) for k in obs}
        guards[f'strata_n_{st}'] = dict(observed=obs, expected_stratified_spikes=exp)
        if obs != exp:
            fail(f'stratum {st} counts {obs} vs hba1c_stratified_spikes {exp}')
    worst, rows = 0.0, []
    for y in OUTCOMES:
        for st in STRATA:
            for x in GUARD_STRATIFIED_SPIKES:
                a = St.loc[(St.stratum == st) & (St.outcome == y) & (St.model == 'M1') & (St.measure == x)]
                b = T.loc[(T.stratum == st) & (T.outcome == y) & (T.measure == x)]
                if len(a) != 1 or len(b) != 1 or int(a.n.iloc[0]) != int(b.n.iloc[0]):
                    fail(f'hba1c_stratified_spikes.py M1 row {st}/{y}/{x} missing or n differs')
                d = max(abs(float(a[c].iloc[0]) - float(b[c].iloc[0])) for c in ('estimate', 'ci_low', 'ci_high', 'p'))
                worst = max(worst, d)
                rows.append(dict(stratum=st, outcome=y, measure=x, n=int(b.n.iloc[0]), estimate_within_range=float(b.estimate.iloc[0]),
                                 estimate_stratified_spikes=float(a.estimate.iloc[0]), max_abs_diff=d))
    guards['s_stratified_spikes_M1_rows_checked'] = len(rows); guards['s_stratified_spikes_M1_max_abs_diff'] = worst
    rec = T.loc[(T.stratum == '57_64') & (T.outcome == 'a1c') & (T.measure == 'all_clearance')].iloc[0]
    guards['recovery_57_64_a1c'] = dict(observed=float(rec.estimate), observed_3dp=round(float(rec.estimate), 3), n=int(rec.n))
    if worst > TOL:
        fail(f'hba1c_stratified_spikes.py M1 not reproduced (max diff {worst}; recovery {rec.estimate})')
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv); t0 = time.monotonic(); guards = {}
    P = load(guards)
    guard_insulin_markers(P, guards)
    G_STRATIFIED_SPIKES = guard_stratified_spikes(P, guards)
    rows = []
    for fam, specs in [('main', MEASURES), ('companion', COMPANION)]:
        for x, label, adj, row in specs:
            for y, (scale, log) in OUTCOMES.items():
                for st in STRATA:
                    d = outcome_frame(P.loc[P.stratum == st], y, log)
                    r = fit_raw(d, x, adj, y)
                    if r is None:
                        fail(f'no fit {x}/{y}/{st}')
                    sd_x = float(d.dropna(subset=[x, y] + [c for c in adj if c not in ('age', x)])[x].std())
                    out = dict(family=fam, measure_key=row, measure=x, label=label, outcome=y, stratum=st, stratum_label=STRATUM_LABEL[st],
                               adjustment='+'.join(['age_bucket', 'sex'] + [c for c in adj if c not in ('age', 'male')]), n=r['n'],
                               measure_sd_in_stratum=sd_x, coef_model_scale=r['estimate'], se_model_scale=r['se'], scale=scale,
                               estimate=to_pct(r['estimate']) if log else r['estimate'], ci_low=to_pct(r['ci_low']) if log else r['ci_low'],
                               ci_high=to_pct(r['ci_high']) if log else r['ci_high'], p=r['p'])
                    rows.append(out)
    T = pd.DataFrame(rows)
    T['q_fdr_column'] = np.nan
    for y in OUTCOMES:
        m = (T.family == 'main') & (T.outcome == y)
        T.loc[m, 'q_fdr_column'] = multipletests(T.loc[m, 'p'], method='fdr_bh')[1]
    T['ci_excludes_null'] = (T.ci_low > 0) | (T.ci_high < 0)
    T['exploratory'] = True
    # range interactions
    irows = []
    for fam, specs in [('main', MEASURES), ('companion', COMPANION)]:
        for x, label, adj, row in specs:
            for y, (scale, log) in OUTCOMES.items():
                it = interaction(P, x, adj, y, log)
                a = T.loc[(T.measure == x) & (T.outcome == y) & (T.stratum == 'lt57')].iloc[0]
                b = T.loc[(T.measure == x) & (T.outcome == y) & (T.stratum == '57_64')].iloc[0]
                diff = b.coef_model_scale - a.coef_model_scale; se = float(np.hypot(a.se_model_scale, b.se_model_scale))
                f = to_pct if log else float
                irows.append(dict(family=fam, measure_key=row, measure=x, label=label, outcome=y, **it,
                                  interaction_scale=('ratio of per-pooled-SD effects, 57_64 vs lt57, minus 1 (%)' if log
                                                     else 'difference in HbA1c points per pooled SD, 57_64 minus lt57'),
                                  stratified_slope_difference=f(diff), stratified_difference_ci_low=f(diff - Z975 * se),
                                  stratified_difference_ci_high=f(diff + Z975 * se), p_stratified_difference=float(2 * stats.norm.sf(abs(diff) / se))))
    I = pd.DataFrame(irows)
    I['q_fdr_interaction'] = np.nan
    for y in OUTCOMES:
        m = (I.family == 'main') & (I.outcome == y)
        I.loc[m, 'q_fdr_interaction'] = multipletests(I.loc[m, 'p_interaction'], method='fdr_bh')[1]
    desc = []
    for st in STRATA:
        S = P.loc[P.stratum == st]
        desc.append(dict(stratum=st, n=int(len(S)), n_fasting_insulin=int((S.fasting_insulin > 0).sum()), hba1c_median=float(S.a1c.median()),
                         hba1c_sd=float(S.a1c.std()), hba1c_min=float(S.a1c.min()), hba1c_max=float(S.a1c.max()),
                         mean_glucose_median=float(S.mean_glucose.median()),
                         **{f'n_{k}': int(S[k].notna().sum()) for k, *_ in MEASURES + COMPANION}))
    D = pd.DataFrame(desc)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {'within_range_associations.csv': T, 'range_interactions.csv': I, 'strata_descriptives.csv': D, 'guard_stratified_spikes_reproduction.csv': G_STRATIFIED_SPIKES}
    for n, f in files.items():
        f.to_csv(OUT / n, index=False)
    summary = dict(label='EXPLORATORY', truncation=TRUNCATION, fdr=dict(method='Benjamini-Hochberg', family='10 measures x 2 strata, per outcome',
                   n_tests_per_outcome=int(((T.family == 'main') & (T.outcome == 'a1c')).sum()),
                   interaction_family='10 measures, per outcome'), guards=guards)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    meta = dict(analysis='within_range_associations', version=VERSION, status='passed', label='exploratory',
                design='cross-sectional associations within HbA1c ranges; no projection', truncation_on_outcome=TRUNCATION,
                definitions=dict(population='canonical primary cohort (no_diabetes_range == True)', hba1c_rounding='decimal half-up (round_a1c) before thresholding',
                                 strata={k: v for k, v in STRATUM_LABEL.items()}, night='00:00-05:00 local', midday='11:00-15:00 local',
                                 adjustment='age bucket + sex + average glucose; average-glucose row age bucket + sex only',
                                 standardization='measure and continuous covariates standardized within stratum sample (hba1c_stratified_spikes.py fit)',
                                 se='HC3', outcomes={'a1c': 'HbA1c points per SD', 'fasting_insulin': '% difference per SD (log outcome)'},
                                 carb_slope='meal_populations.py window_mixed acs (meals within +/-90 local days of anchor)',
                                 peak_rise='insulin_markers.py per-person mean PEAK_RISE_2H, meals within +/-90 days, >=10 meals',
                                 rise_beyond_content='companion; insulin_markers.py macro-residualized peak rise (outside FDR family)',
                                 interaction='pooled lt57+57_64, measure pooled-SD, common covariates, measure x stratum (HC3); plus stratified slope difference z test'),
                guards=guards,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'within_range_associations.metadata.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    print(json.dumps(guards, indent=1, default=float))


if __name__ == '__main__':
    main()
