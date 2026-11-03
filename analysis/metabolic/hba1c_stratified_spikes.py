#!/usr/bin/env python3
"""HbA1c-stratified spike recovery and meal-response associations.

Cross-sectional associations only; no projection. Pre-specified from an exploratory
stratification done before this producer was written.

Population: canonical all-lab-linked panel. Anchor HbA1c rounded decimal half-up.
  strata  lt57          primary cohort, HbA1c <5.7
          57_64         primary cohort, HbA1c 5.7-6.4                      [PRIMARY]
          ge65          anchor HbA1c >=6.5 (diabetes range; medication unknown)  [extension]
          ge57_all      anchor HbA1c >=5.7 including all diabetes-range labs     [extension]
          57_64_anchor  anchor HbA1c 5.7-6.4 under an anchor-only rule (adds participants
                        excluded only for a diabetes-range lab at another time) [rule sensitivity]
Primary hypothesis: within 57_64, slower spike recovery (half-recovery minutes, from the
spike-detection logic in analysis/meals/spike_facets.py) is associated with higher HbA1c
at the same mean glucose, age, sex and BMI.
Secondary: FPC3 (primary cohort only; FPCA was fit there), meal-linked recovery in
consistent loggers, logged-meal response size, HOMA-IR and fasting insulin outcomes.
Models: OLS, HC3; HbA1c in percentage points per SD of the measure (SD within stratum);
M1 age+sex+mean glucose; M2 M1+BMI (complete case); M3 M2+spike size+spike frequency;
M4 M2 with seven level summaries in place of mean glucose.
Checks: interaction of recovery with stratum (M2) in lt57 + 57_64 + ge65; 200 seeded random
halves of 57_64; quarters of recovery relative to same-mean-glucose peers.

Inputs: OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py),
OUTPUTS/meals/spike_facets/spike_facets_participants.parquet and spike_exposure_daily.parquet
(analysis/meals/spike_facets.py), OUTPUTS/meals/meal_fpca_per_user.csv (analysis/meals/meal_fpca.py),
OUTPUTS/metabolic/insulin_markers/summary.json (analysis/metabolic/insulin_markers.py),
and the study database (LEVELS_DB) for spike-event re-extraction.

Outputs (OUTPUTS/metabolic/hba1c_stratified_spike_recovery/): stratified_associations.csv,
recovery_quarters.csv, strata_descriptives.csv, split_halves.csv, summary.json, and
hba1c_stratified_spike_recovery.metadata.json.

Usage: python analysis/metabolic/hba1c_stratified_spikes.py [--db PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from cgmpaper.publication_panels import load_panel  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


s_spike_facets = _load('spike_facets_module', 'analysis/meals/spike_facets.py')
s_insulin_markers = _load('insulin_markers_module', 'analysis/metabolic/insulin_markers.py')
s_spike_exposure = s_spike_facets.s_spike_exposure
ROOT = Path(__file__).resolve().parents[2]

VERSION = 'hba1c-stratified-spike-recovery'
PUB = OUTPUTS
OUT = PUB / 'metabolic/hba1c_stratified_spike_recovery'
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
FACETS = PUB / 'meals/spike_facets/spike_facets_participants.parquet'
FPCA = PUB / 'meals/meal_fpca_per_user.csv'
S_INSULIN_MARKERS_SUMMARY = PUB / 'metabolic/insulin_markers/summary.json'
SEED = 20260926
N_SPLITS = 200
N_GUARD = 300
LEVEL = s_spike_facets.LEVEL
MEASURES = ['all_clearance', 'all_size', 'all_frequency', 'meal_primary_clearance', 'fpc3_mean', 'peak_rise', 'iauc_2h',
            'rise_beyond_content', 'return_fraction']
ADJ = {'M1': ['age', 'male', 'mean_glucose'], 'M2': ['age', 'male', 'mean_glucose', 'bmi'],
       'M3': ['age', 'male', 'mean_glucose', 'bmi', 'all_size', 'all_frequency'], 'M4': ['age', 'male', *LEVEL, 'bmi']}


def facets_for(con, panel, people):
    sub = panel.loc[panel.USER_ID.isin(people)].copy(); sub.attrs = dict(panel.attrs)
    days = s_spike_exposure.dense_days(sub)
    E, D, M, mismatch = s_spike_facets.extract(con, days, np.sort(np.asarray(people)))
    if mismatch:
        raise SystemExit(f'GUARD FAILED: event extraction differs from daily_metrics on {mismatch} days')
    return s_spike_facets.participant_facets(E, D)


def logger_flags(con, panel, people):
    """Consistent-logger rule from analysis/meals/spike_facets.py (>=3 logs on >=80% of >=14 exposure dates)."""
    ex = pd.read_parquet(s_spike_facets.CACHE / 'spike_exposure_daily.parquet', columns=['USER_ID', 'DAY', 'n_logs'])
    ex = ex.loc[ex.USER_ID.isin(people)].groupby('USER_ID').n_logs
    return ((ex.apply(lambda v: (v >= 3).mean()) >= 0.8) & (ex.size() >= 14)).rename('logger_high')


def fit(s, x, adj, y):
    need = [x, y] + [c for c in adj if c != 'age']
    s = s.dropna(subset=need).reset_index(drop=True)
    if len(s) < 40 or s[x].std() == 0:
        return None
    X = s[[x] + [c for c in adj if c != 'age']].astype(float); X = (X - X.mean()) / X.std()
    if 'age' in adj:
        a = pd.get_dummies(s.age_bucket, prefix='age', drop_first=True, dtype=float); X = X.join(a.loc[:, a.sum() >= 10])
    m = sm.OLS(s[y].to_numpy(float), sm.add_constant(X)).fit(cov_type='HC3')
    b, (lo, hi) = m.params[x], m.conf_int().loc[x]
    return dict(n=int(len(s)), estimate=float(b), ci_low=float(lo), ci_high=float(hi), p=float(m.pvalues[x]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', type=Path, default=database_path())
    args = ap.parse_args(argv); t0 = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel(PANEL, cohort='all_lab_linked', required=['peaks_per_day'], expected_window='symmetric')
    raw = pd.read_csv(PANEL, low_memory=False, usecols=['USER_ID', 'homa_ir', 'fasting_insulin', 'age_bucket', 'sex', 'bmi', *LEVEL])
    P = panel[['USER_ID', 'a1c', 'no_diabetes_range', 'anchor_date']].merge(raw, on='USER_ID', validate='one_to_one')
    P['male'] = (P.sex == 'male').astype(float)
    saved = pd.read_parquet(FACETS)
    dr = P.loc[~P.no_diabetes_range, 'USER_ID'].to_numpy()
    rng = np.random.default_rng(SEED)
    probe = rng.choice(np.sort(saved.USER_ID.to_numpy()), N_GUARD, replace=False)
    with duckdb.connect(str(args.db), read_only=True) as con:
        con.execute('SET threads=6')
        Fp = facets_for(con, panel, probe)
        Fd = facets_for(con, panel, dr)
        lg = logger_flags(con, panel, dr)
    cols = [c for c in Fp.columns if c != 'USER_ID']
    chk = Fp.merge(saved[['USER_ID', *cols]], on='USER_ID', suffixes=('_new', '_saved'), validate='one_to_one')
    diffs = {c: float(np.nanmax(np.abs(chk[c + '_new'] - chk[c + '_saved']))) if chk[c + '_new'].notna().any() else 0.0 for c in cols}
    missing_ok = all((chk[c + '_new'].isna() == chk[c + '_saved'].isna()).all() for c in cols)
    guards = dict(probe_n=int(len(chk)), probe_max_abs_diff=max(diffs.values()), probe_missing_pattern_equal=bool(missing_ok))
    if not (len(chk) == N_GUARD and guards['probe_max_abs_diff'] < 1e-9 and missing_ok):
        raise SystemExit(f'GUARD FAILED: facet re-extraction {guards}')
    Fd = Fd.merge(lg.reset_index(), on='USER_ID', how='left'); Fd['logger_high'] = Fd.logger_high.fillna(False).astype(bool)
    F = pd.concat([saved[['USER_ID', 'all_frequency', 'all_size', 'all_clearance', 'meal_primary_clearance', 'logger_high']],
                   Fd[['USER_ID', 'all_frequency', 'all_size', 'all_clearance', 'meal_primary_clearance', 'logger_high']]], ignore_index=True)
    guards['diabetes_range_with_facets'] = int(Fd.all_clearance.notna().sum())
    # logged-meal responses (analysis/metabolic/insulin_markers.py definition) for all lab-linked participants
    R, counts = s_insulin_markers.meal_responses(P)
    prim_counts = s_insulin_markers.meal_responses(P.loc[P.no_diabetes_range])[1]
    s_insulin_markers_c = json.loads(S_INSULIN_MARKERS_SUMMARY.read_text())['samples']
    guards['meal_counts_primary_match_insulin_markers'] = (prim_counts['meals_in_window'] == s_insulin_markers_c['meals_in_window']
                                               and prim_counts['participants_ge_min_meals'] == s_insulin_markers_c['participants_ge_min_meals'])
    if not guards['meal_counts_primary_match_insulin_markers']:
        raise SystemExit(f'GUARD FAILED: meal counts {prim_counts} vs {s_insulin_markers_c}')
    fp = pd.read_csv(FPCA, usecols=['USER_ID', 'fpc3_mean'])
    A = P.merge(F, on='USER_ID', how='left').merge(R.drop(columns=['n_meals']), on='USER_ID', how='left').merge(fp, on='USER_ID', how='left')
    A.loc[~A.logger_high.fillna(False).astype(bool), 'meal_primary_clearance'] = np.nan  # meal-linked recovery in consistent loggers only
    A.loc[~A.no_diabetes_range, 'fpc3_mean'] = np.nan
    a = A.a1c
    ever_only = (~A.no_diabetes_range) & (a < 6.5)
    strata = {'lt57': A.no_diabetes_range & (a < 5.7), '57_64': A.no_diabetes_range & (a >= 5.7),
              'ge65': a >= 6.5, 'ge57_all': a >= 5.7, '57_64_anchor': (a >= 5.7) & (a < 6.5)}
    guards['n_excluded_only_by_other_labs'] = int(ever_only.sum())
    rows = []
    for st, mask in strata.items():
        S = A.loc[mask]
        for y in ['a1c', 'homa_ir', 'fasting_insulin']:
            d = S.copy()
            if y != 'a1c':
                d = d.loc[d[y] > 0]; d[y] = np.log(d[y])
            for mname, adj in ADJ.items():
                for x in MEASURES:
                    if mname == 'M3' and x in ('all_size', 'all_frequency'):
                        continue
                    r = fit(d, x, [c for c in adj if c != x], y)
                    if r:
                        if y != 'a1c':
                            r.update({k: float(100 * (np.exp(r[k]) - 1)) for k in ('estimate', 'ci_low', 'ci_high')})
                        rows.append(dict(stratum=st, outcome=y, model=mname, measure=x,
                                         scale='hba1c_points_per_sd' if y == 'a1c' else 'percent_per_sd', **r))
    T = pd.DataFrame(rows)
    # interaction: recovery x stratum (M2) in lt57 + 57_64 + ge65
    I = A.loc[strata['lt57'] | strata['57_64'] | strata['ge65']].dropna(subset=['all_clearance', 'a1c', 'mean_glucose', 'bmi']).copy()
    I['grp'] = np.select([strata['57_64'][I.index], strata['ge65'][I.index]], ['57_64', 'ge65'], 'lt57')
    z = (I.all_clearance - I.all_clearance.mean()) / I.all_clearance.std()
    X = pd.DataFrame({'clear': z, 'g57': (I.grp == '57_64').astype(float), 'g65': (I.grp == 'ge65').astype(float)})
    X['clear_x_g57'], X['clear_x_g65'] = X.clear * X.g57, X.clear * X.g65
    for c in ['mean_glucose', 'bmi']:
        X[c] = (I[c] - I[c].mean()) / I[c].std()
    X['male'] = I.male.to_numpy()
    ages = pd.get_dummies(I.age_bucket, prefix='age', drop_first=True, dtype=float); X = X.join(ages.loc[:, ages.sum() >= 10])
    mi = sm.OLS(I.a1c.to_numpy(float), sm.add_constant(X)).fit(cov_type='HC3'); ci = mi.conf_int()
    inter = {k: dict(estimate=float(mi.params[k]), ci_low=float(ci.loc[k, 0]), ci_high=float(ci.loc[k, 1]), p=float(mi.pvalues[k]))
             for k in ['clear', 'clear_x_g57', 'clear_x_g65']}
    inter['n'] = int(len(I)); inter['scale'] = 'hba1c points per pooled SD of recovery; clear = slope in lt57'
    # split halves of the primary stratum
    S = A.loc[strata['57_64']].dropna(subset=['all_clearance', 'a1c', 'mean_glucose', 'bmi'])
    ids = S.USER_ID.to_numpy(); hv = []
    for i in range(N_SPLITS):
        h = set(rng.choice(ids, len(ids) // 2, replace=False))
        ra, rb = fit(S.loc[S.USER_ID.isin(h)], 'all_clearance', ADJ['M2'], 'a1c'), fit(S.loc[~S.USER_ID.isin(h)], 'all_clearance', ADJ['M2'], 'a1c')
        hv.append(dict(split=i, a=ra['estimate'], a_p=ra['p'], b=rb['estimate'], b_p=rb['p'], n_a=ra['n'], n_b=rb['n']))
    H = pd.DataFrame(hv)
    split = dict(n_splits=N_SPLITS, both_positive=float(((H.a > 0) & (H.b > 0)).mean()), both_p_lt_05=float(((H.a_p < .05) & (H.b_p < .05)).mean()),
                 median=float(pd.concat([H.a, H.b]).median()), q025=float(pd.concat([H.a, H.b]).quantile(.025)), q975=float(pd.concat([H.a, H.b]).quantile(.975)))
    # quarters of recovery relative to same-mean-glucose peers (tenths of mean glucose within stratum)
    qrows = []
    for st in ['lt57', '57_64', 'ge65']:
        d = A.loc[strata[st]].dropna(subset=['all_clearance', 'a1c']).copy()
        d['bin'] = pd.qcut(d.mean_glucose, 10, labels=False, duplicates='drop')
        d['rel'] = d.all_clearance - d.groupby('bin').all_clearance.transform('median')
        d['quarter'] = pd.qcut(d.rel, 4, labels=[1, 2, 3, 4]).astype(int)
        for q, s in d.groupby('quarter'):
            qrows.append(dict(stratum=st, quarter=int(q), n=int(len(s)), recovery_median_min=float(s.all_clearance.median()),
                              mean_glucose=float(s.mean_glucose.mean()), mean_hba1c=float(s.a1c.mean()),
                              pct_hba1c_ge_6_0=float(100 * (s.a1c >= 6.0).mean()), bmi_mean=float(s.bmi.mean())))
    Q = pd.DataFrame(qrows)
    desc = pd.DataFrame([dict(stratum=st, n=int(m.sum()), n_recovery=int(A.loc[m, 'all_clearance'].notna().sum()),
                              n_bmi=int(A.loc[m, 'bmi'].notna().sum()), n_meal_responses=int(A.loc[m, 'peak_rise'].notna().sum()),
                              hba1c_median=float(A.loc[m, 'a1c'].median()), mean_glucose_median=float(A.loc[m, 'mean_glucose'].median()),
                              recovery_median_min=float(A.loc[m, 'all_clearance'].median()), recovery_sd_min=float(A.loc[m, 'all_clearance'].std()),
                              bmi_median=float(A.loc[m, 'bmi'].median())) for st, m in strata.items()])
    files = {'stratified_associations.csv': T, 'recovery_quarters.csv': Q, 'strata_descriptives.csv': desc, 'split_halves.csv': H}
    for n, f in files.items():
        f.to_csv(OUT / n, index=False)
    (OUT / 'summary.json').write_text(json.dumps(dict(interaction=inter, split_halves=split, guards=guards), indent=1) + '\n')
    meta = dict(analysis='hba1c_stratified_spike_recovery', version=VERSION,
                design='cross-sectional associations; no projection', seed=SEED, guards=guards, strata_definitions={k: str(k) for k in strata},
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'hba1c_stratified_spike_recovery.metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(dict(guards=guards, interaction=inter, split=split), indent=1))


if __name__ == '__main__':
    main()
