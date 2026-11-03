#!/usr/bin/env python3
"""HOMA-IR by quarter of mean glucose, and correlation of spike and meal measures with mean glucose.

Cross-sectional associations only; no projection. Population: canonical primary cohort
(outputs/canonical_panel, no_diabetes_range == True), exactly as
analysis/metabolic/insulin_markers.py. HOMA-IR is the primary marker (sample: HOMA-IR > 0; subset with BMI).

Sections
  A  Quarters of mean CGM glucose (cut points from the HOMA-IR participants):
     - crude % HOMA-IR >= 2.5 (n/N) and crude geometric mean HOMA-IR, in all HOMA-IR participants and in the
       subset with BMI;
     - adjusted geometric mean HOMA-IR per quarter by marginal standardization of an OLS model of
       log(HOMA-IR) on quarter + age bucket + sex (all HOMA-IR participants; and the subset with BMI) and
       + standardized BMI (BMI subset, complete case); 95% CI from 1,000 seeded participant bootstraps;
       ratio of each quarter to quarter 1 from the model (HC3).
  B  Spearman correlation of spike facets (analysis/meals/spike_facets.py: frequency, size, recovery), logged-meal
     responses (analysis/metabolic/insulin_markers.py definition: meals within +/-90 days of the HbA1c anchor, >=10 meals) and
     overnight-minus-midday glucose with mean glucose, in the primary cohort and in the HOMA-IR
     sample; 95% CI by Fisher z with the Bonett-Wright standard error.
  C  Share of HbA1c-normal (<5.7%, half-up) participants whose mean CGM glucose is >=110 mg/dL,
     in the HOMA-IR sample (reproduces analysis/metabolic/insulin_markers.py discordance counts) and in the whole primary cohort.

Inputs
  OUTPUTS/canonical_panel/... (via analysis/metabolic/insulin_markers.py): canonical lab/CGM panel
  OUTPUTS/meals/spike_facets/...: spike facet measures
  OUTPUTS/metabolic/insulin_markers/cgm_measures_by_marker.csv, summary.json: HOMA-IR associations and
    meal-response counts from analysis/metabolic/insulin_markers.py

Outputs (OUTPUTS/metabolic/level_quarters/, participant-level data not published)
  mean_glucose_quarters.csv, mean_glucose_quarters_adjusted.csv, level_correlations.csv,
  hba1c_normal_cgm_elevated.csv, summary.json, level_quarters.metadata.json

Usage
  python analysis/metabolic/level_quarters.py [--out PATH]
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from cgmpaper.lab_normalization import round_a1c  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


s_insulin_markers = _load('insulin_markers_for_level_quarters', 'analysis/metabolic/insulin_markers.py')

VERSION = 'level-quarters'
PUB = OUTPUTS
OUT = PUB / 'metabolic/level_quarters'
S_INSULIN_MARKERS_DIR = PUB / 'metabolic/insulin_markers'
S_INSULIN_MARKERS_SCRIPT = ROOT / 'analysis/metabolic/insulin_markers.py'
PANEL, FACETS, MEALS, FPCA = s_insulin_markers.PANEL, s_insulin_markers.FACETS, s_insulin_markers.MEALS, s_insulin_markers.FPCA
SEED = 20260926
N_BOOT = 1000
THRESH = 2.5
SPIKE = ['all_frequency', 'all_size', 'all_clearance']
MEAL = ['peak_rise', 'iauc_2h', 'rise_beyond_content', 'share_rise_ge30', 'time_to_peak', 'return_minutes', 'fpc3_mean']
OTHER = ['night_minus_day']


def design(s, quarter=True, bmi=False):
    X = pd.DataFrame(index=s.index)
    if quarter:
        for k in (2, 3, 4):
            X[f'q{k}'] = (s.quarter == k).astype(float)
    ages = pd.get_dummies(s.age_bucket, prefix='age', drop_first=True, dtype=float)
    X = X.join(ages.loc[:, ages.sum() >= 10])
    X['male'] = s.male.astype(float)
    if bmi:
        X['bmi'] = (s.bmi - s.bmi.mean()) / s.bmi.std()
    return sm.add_constant(X, has_constant='add')


def adjusted_gm(s, bmi):
    X = design(s, bmi=bmi)
    m = sm.OLS(np.log(s.homa_ir.to_numpy(float)), X).fit()
    out = {}
    for k in (1, 2, 3, 4):
        Xk = X.copy()
        for j in (2, 3, 4):
            Xk[f'q{j}'] = float(j == k)
        out[k] = float(np.exp(m.predict(Xk).mean()))
    return out


def section_a(H, rng):
    H = H.copy()
    H['quarter'], edges = pd.qcut(H.mean_glucose, 4, labels=[1, 2, 3, 4], retbins=True)
    H['quarter'] = H.quarter.astype(int)
    crude = []
    for sample, d in [('homa_all', H), ('homa_with_bmi', H.loc[H.bmi.notna()])]:
        for k, s in d.groupby('quarter'):
            crude.append(dict(sample=sample, quarter=int(k), n=int(len(s)), n_homa_ge_2_5=int((s.homa_ir >= THRESH).sum()),
                              pct_homa_ge_2_5=float(100 * (s.homa_ir >= THRESH).mean()),
                              geometric_mean_homa=float(np.exp(np.log(s.homa_ir).mean())),
                              mean_glucose_min=float(s.mean_glucose.min()), mean_glucose_max=float(s.mean_glucose.max()),
                              mean_glucose_median=float(s.mean_glucose.median()), bmi_mean=float(s.bmi.mean()),
                              n_bmi=int(s.bmi.notna().sum())))
    adj = []
    for name, d, bmi in [('age_sex', H, False), ('age_sex_bmi_subset', H.loc[H.bmi.notna()], False),
                         ('age_sex_bmi', H.loc[H.bmi.notna()], True)]:
        d = d.reset_index(drop=True)
        point = adjusted_gm(d, bmi)
        boots = []
        for _ in range(N_BOOT):
            b = d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True)
            boots.append(adjusted_gm(b, bmi))
        m = sm.OLS(np.log(d.homa_ir.to_numpy(float)), design(d, bmi=bmi)).fit(cov_type='HC3'); ci = m.conf_int()
        for k in (1, 2, 3, 4):
            bk = np.array([b[k] for b in boots])
            row = dict(adjustment=name, quarter=k, n=int(len(d)), n_quarter=int((d.quarter == k).sum()),
                       adjusted_geometric_mean_homa=point[k], ci_low=float(np.percentile(bk, 2.5)), ci_high=float(np.percentile(bk, 97.5)))
            if k == 1:
                row.update(ratio_vs_q1=1.0, ratio_ci_low=np.nan, ratio_ci_high=np.nan, ratio_p=np.nan)
            else:
                row.update(ratio_vs_q1=float(np.exp(m.params[f'q{k}'])), ratio_ci_low=float(np.exp(ci.loc[f'q{k}', 0])),
                           ratio_ci_high=float(np.exp(ci.loc[f'q{k}', 1])), ratio_p=float(m.pvalues[f'q{k}']))
            adj.append(row)
    return pd.DataFrame(crude), pd.DataFrame(adj), [float(e) for e in edges]


def spearman_rows(frame, sample, measures):
    rows = []
    for x in measures:
        s = frame.dropna(subset=[x, 'mean_glucose'])
        rho, p = stats.spearmanr(s[x], s.mean_glucose)
        n = len(s)
        se = np.sqrt((1 + rho ** 2 / 2) / (n - 3))
        z = np.arctanh(rho)
        rows.append(dict(sample=sample, measure=x, family='spike' if x in SPIKE else ('meal' if x in MEAL else 'day_part'), n=int(n),
                         spearman_rho=float(rho), ci_low=float(np.tanh(z - 1.96 * se)), ci_high=float(np.tanh(z + 1.96 * se)), p=float(p)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = time.monotonic(); guards = {}
    panel, P, g_insulin_markers = s_insulin_markers.load(args)
    guards['primary_n'] = int(len(P))
    H = P.loc[P.homa_ir.notna() & (P.homa_ir > 0)].copy()
    guards['homa_n'], guards['homa_with_bmi'] = int(len(H)), int(H.bmi.notna().sum())
    # reproduce analysis/metabolic/insulin_markers.py's saved HOMA-IR ~ mean glucose associations exactly, with its own fit()
    saved = pd.read_csv(S_INSULIN_MARKERS_DIR / 'cgm_measures_by_marker.csv')
    Hy = H.assign(y=np.log(H.homa_ir))
    rep = {}
    for adj in ('age_sex', 'bmi'):
        r = s_insulin_markers.fit(Hy, 'mean_glucose', s_insulin_markers.ADJ[adj])
        sv = saved.loc[(saved.marker == 'homa_ir') & (saved.measure == 'mean_glucose') & (saved.adjustment == adj)].iloc[0]
        diff = max(abs(r[k] - float(sv[k])) for k in ('estimate', 'ci_low', 'ci_high', 'p'))
        rep[adj] = dict(n=r['n'], estimate=r['estimate'], saved_estimate=float(sv.estimate), max_abs_diff=float(diff), n_matches=r['n'] == int(sv.n))
    guards['mean_glucose_homa_reproduced'] = rep
    rng = np.random.default_rng(SEED)
    A_crude, A_adj, edges = section_a(H, rng)
    # B: meal responses by analysis/metabolic/insulin_markers.py's definition
    R, counts = s_insulin_markers.meal_responses(P)
    s_insulin_markers_c = json.loads((S_INSULIN_MARKERS_DIR / 'summary.json').read_text())['samples']
    guards['meal_counts_match_insulin_markers'] = counts['meals_in_window'] == s_insulin_markers_c['meals_in_window'] and \
        counts['participants_ge_min_meals'] == s_insulin_markers_c['participants_ge_min_meals']
    PB = P.merge(R, on='USER_ID', how='left')
    HB = PB.loc[PB.USER_ID.isin(H.USER_ID)]
    B = pd.DataFrame(spearman_rows(PB, 'primary', SPIKE + MEAL + OTHER) + spearman_rows(HB, 'homa_sample', SPIKE + MEAL + OTHER))
    # C: HbA1c-normal participants with elevated mean CGM
    crows = []
    for sample, d in [('homa_sample', H), ('primary', P)]:
        d = d.dropna(subset=['a1c', 'mean_glucose'])
        norm = round_a1c(d.a1c.to_numpy()) < 5.7
        hi = d.mean_glucose.to_numpy() >= 110
        crows.append(dict(sample=sample, n_hba1c_normal=int(norm.sum()), n_hba1c_normal_cgm_ge_110=int((norm & hi).sum()),
                          pct=float(100 * (norm & hi).sum() / norm.sum())))
    C = pd.DataFrame(crows)
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    files = {'mean_glucose_quarters.csv': A_crude, 'mean_glucose_quarters_adjusted.csv': A_adj,
             'level_correlations.csv': B, 'hba1c_normal_cgm_elevated.csv': C}
    for n, f in files.items():
        f.to_csv(out / n, index=False)
    summary = dict(samples=dict(primary_n=guards['primary_n'], homa_n=guards['homa_n'], homa_with_bmi=guards['homa_with_bmi'], **counts),
                   quarter_edges_mean_glucose=edges, n_boot=N_BOOT)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    meta = dict(analysis='level_quarters', version=VERSION, status='passed',
                design='cross-sectional associations; no projection', seed=SEED, guards=guards,
                definitions=dict(homa_threshold=THRESH, quarters='pd.qcut of mean glucose in the HOMA-IR sample',
                                 adjusted_mean='marginal standardization of OLS log(HOMA-IR); participant bootstrap percentile CI',
                                 cgm_elevated='mean CGM glucose >= 110 mg/dL', hba1c_normal='HbA1c < 5.7% (decimal half-up)',
                                 spearman_ci='Fisher z, Bonett-Wright SE'),
                runtime_seconds=round(time.monotonic() - t0, 1))
    (out / 'level_quarters.metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(dict(guards=guards, summary=summary), indent=1, default=float))


if __name__ == '__main__':
    main()
