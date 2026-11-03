#!/usr/bin/env python3
"""With/without-BMI comparisons in the same participants, and unlinked spikes vs daily steps.

Cross-sectional associations only; no projection. Every with/without-BMI comparison uses the
same (BMI-complete) participants.

Population: canonical primary cohort (outputs/canonical_panel, no_diabetes_range == True).
Outcomes log(HOMA-IR) and log(fasting insulin), values >0.
Effects: % difference per SD of the measure, 100*(exp(b)-1), OLS with HC3 SEs, 95% Wald CI.
Models use analysis/metabolic/insulin_markers.py's fit() unchanged: continuous covariates
standardized within the fitted sample, age-bucket dummies (buckets with >=10 participants), male
indicator (sex == 'male').

Part 1  same_sample_bmi.csv
  (a) average (mean) glucose; without = age+sex; with = age+sex+BMI
  (b) overnight minus midday (mean 00-05 local minus mean 11-15 local); without = age+sex+average
      glucose; with = + BMI
  (c) rise-based spike frequency, size and recovery (all_frequency, all_size, all_clearance =
      half-recovery minutes; analysis/meals/spike_facets.py); without = age+sex+average glucose;
      with = + BMI
  (d) HbA1c-CGM discordance groups vs both-normal (anchor HbA1c rounded decimal half-up >=5.7%;
      mean CGM >=110 mg/dL); without = age+sex; with = age+sex+BMI (BMI standardized)
  For each row: without-BMI in the BMI-complete sample (same participants as with-BMI), with-BMI,
  and the full-sample without-BMI estimate for reference. Attenuation (%) =
  100*(1 - b_with/b_without_same) on the log-coefficient scale, flagged interpretable only when the
  same-sample without-BMI 95% CI excludes zero (>100% = sign reversal; <0 = strengthened). The per-SD scale of continuous
  measures is the SD within each fitted sample (reported as sd_x_*).

Part 2  spike_frequency_steps.json (exploratory)
  Primary cohort, consistent loggers (analysis/meals/spike_facets.py / analysis/meals/spike_exposure.py
  high loggers: >=3 food logs on >=80% of dense dates, >=14 dates). Unlinked-spike and
  meal-linked-spike frequency (primary onset window, spikes per eligible day; spike_facets.py
  participant outputs) vs average daily steps (analysis/metabolic/discordance_behavior.py definition,
  obtained by calling its build(): dense CGM days within +/-90 local days of the HbA1c anchor, days
  with >0 steps, >=7 such days). Spearman rho, and partial Spearman: Pearson correlation of
  residuals of the ranks of x and y after OLS on age-bucket dummies (>=10), male and the rank of
  average glucose. Percentile bootstrap CIs (participant resamples; ranks recomputed within each
  resample), including the paired difference unlinked minus meal-linked. All-spike frequency and
  all primary-cohort participants (any logging) are given as references.

Inputs
  outputs/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)
  outputs/meals/spike_facets/spike_facets_participants.parquet and its metadata
    (analysis/meals/spike_facets.py)
  outputs/daily_glucose/_cache/daily_exposures_unique.parquet (analysis/daily_glucose/build_day_panel.py)
  outputs/meals/meal_shape_substrate.parquet
  outputs/metabolic/insulin_markers (analysis/metabolic/insulin_markers.py) and its metadata
  outputs/metabolic/discordance_behavior (analysis/metabolic/discordance_behavior.py) and its metadata

Outputs (outputs/metabolic/bmi_same_sample/; participant-level results are aggregate statistics,
not individual records)
  same_sample_bmi.csv, spike_frequency_steps.json, summary.json, bmi_same_sample.metadata.json

Usage
  python analysis/metabolic/bmi_same_sample.py
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


s_insulin_markers = _load('insulin_markers_module', 'analysis/metabolic/insulin_markers.py')
s_discordance_behavior = _load('discordance_behavior_module', 'analysis/metabolic/discordance_behavior.py')

VERSION = 'bmi-same-sample'
PUB = OUTPUTS
OUT = PUB / 'metabolic/bmi_same_sample'
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
FACETS = PUB / 'meals/spike_facets/spike_facets_participants.parquet'
FACETS_META = PUB / 'meals/spike_facets/spike_facets.metadata.json'
EXPOSURES = PUB / 'daily_glucose/_cache/daily_exposures_unique.parquet'
MEALS = PUB / 'meals/meal_shape_substrate.parquet'
D_INSULIN_MARKERS = PUB / 'metabolic/insulin_markers'
D_DISCORDANCE_BEHAVIOR = PUB / 'metabolic/discordance_behavior'
M_INSULIN_MARKERS = D_INSULIN_MARKERS / 'insulin_markers.metadata.json'
M_DISCORDANCE_BEHAVIOR = D_DISCORDANCE_BEHAVIOR / 'discordance_behavior.metadata.json'
SEED = 20260928
N_BOOT = 2000
TOL = 1e-8
OUTCOMES = {'homa_ir': 'HOMA-IR', 'fasting_insulin': 'Fasting insulin'}
# (panel, measure, without-BMI adjustment key, with-BMI adjustment key) - keys of insulin_markers.py's ADJ
CONTINUOUS = [('a', 'mean_glucose', 'age_sex', 'bmi'), ('b', 'night_minus_day', 'mean_glucose', 'mean_glucose_bmi'),
              ('c', 'all_frequency', 'mean_glucose', 'mean_glucose_bmi'), ('c', 'all_size', 'mean_glucose', 'mean_glucose_bmi'),
              ('c', 'all_clearance', 'mean_glucose', 'mean_glucose_bmi')]
LABELS = {'mean_glucose': 'Average glucose', 'night_minus_day': 'Overnight minus midday', 'all_frequency': 'Spike frequency',
          'all_size': 'Spike size', 'all_clearance': 'Spike recovery (half-recovery minutes)', 'cgm_only': 'CGM-only elevated',
          'a1c_only': 'HbA1c-only elevated', 'both_elevated': 'Both elevated'}
GROUPS = ['both_normal', 'cgm_only', 'a1c_only', 'both_elevated']
CLAIM_NAMES = {'Concordant normal': 'both_normal', 'CGM-only elevated': 'cgm_only', 'HbA1c-only elevated': 'a1c_only',
               'Concordant elevated': 'both_elevated'}


def outcome_frame(P, marker):
    d = P.loc[P[marker].notna() & (P[marker] > 0)].copy()
    d['y'] = np.log(d[marker])
    return d


def disc_fit(s, bmi):
    """Script 140's discordance model: log outcome on group dummies (ref both_normal), age buckets, male [, std BMI]."""
    s = s.reset_index(drop=True)
    X = pd.get_dummies(s.group, dtype=float).reindex(columns=GROUPS, fill_value=0.0).drop(columns='both_normal')
    ages = pd.get_dummies(s.age_bucket, prefix='age', drop_first=True, dtype=float)
    X = X.join(ages.loc[:, ages.sum() >= 10]); X['male'] = s.male.astype(float)
    if bmi:
        X['bmi'] = (s.bmi - s.bmi.mean()) / s.bmi.std()
    m = sm.OLS(s.y.to_numpy(), sm.add_constant(X)).fit(cov_type='HC3'); ci = m.conf_int()
    return {g: dict(n=int(len(s)), n_group=int((s.group == g).sum()), b=float(m.params[g]), estimate=float(100 * np.expm1(m.params[g])),
                    ci_low=float(100 * np.expm1(ci.loc[g, 0])), ci_high=float(100 * np.expm1(ci.loc[g, 1])), p=float(m.pvalues[g]))
            for g in GROUPS[1:]}


def close(a, b):
    return abs(float(a) - float(b)) < TOL


def part1(P, guards):
    rows = []
    for marker in OUTCOMES:
        d = outcome_frame(P, marker)
        for panel, x, adj_wo, adj_w in CONTINUOUS:
            a_wo = [c for c in s_insulin_markers.ADJ[adj_wo] if c != x]; a_w = [c for c in s_insulin_markers.ADJ[adj_w] if c != x]
            full = s_insulin_markers.fit(d.dropna(subset=[x]), x, a_wo)
            S = d.dropna(subset=[x, 'y'] + [c for c in a_w if c != 'age'])
            wo, w = s_insulin_markers.fit(S, x, a_wo), s_insulin_markers.fit(S, x, a_w)
            if wo['n'] != w['n']:
                raise SystemExit(f'GUARD FAILED: same-sample n differs {marker} {x}: {wo["n"]} vs {w["n"]}')
            bw, bwo = np.log1p(w['estimate'] / 100), np.log1p(wo['estimate'] / 100)
            rows.append(dict(panel=panel, outcome=marker, term=x, label=LABELS[x], adjustment_without=adj_wo, adjustment_with=adj_w,
                             n_same=w['n'], est_without_same=wo['estimate'], ci_low_without_same=wo['ci_low'], ci_high_without_same=wo['ci_high'],
                             p_without_same=wo['p'], est_with_bmi=w['estimate'], ci_low_with_bmi=w['ci_low'], ci_high_with_bmi=w['ci_high'],
                             p_with_bmi=w['p'], n_full=full['n'], est_without_full=full['estimate'], ci_low_without_full=full['ci_low'],
                             ci_high_without_full=full['ci_high'], p_without_full=full['p'], attenuation_pct=float(100 * (1 - bw / bwo)),
                             sd_x_same=float(S[x].std()), sd_x_full=float(d.dropna(subset=[x] + [c for c in a_wo if c != 'age'])[x].std()),
                             scale='percent_per_sd'))
    return rows


def part1_discordance(P, A_DISCORDANCE_BEHAVIOR, guards):
    P = P.copy()
    a1c_hi, cgm_hi = round_a1c(P.a1c.to_numpy()) >= 5.7, P.mean_glucose.to_numpy() >= 110
    P['group'] = np.select([~a1c_hi & ~cgm_hi, ~a1c_hi & cgm_hi, a1c_hi & ~cgm_hi], ['both_normal', 'cgm_only', 'a1c_only'], 'both_elevated')
    H = outcome_frame(P, 'homa_ir')
    F = outcome_frame(P, 'fasting_insulin')
    rows = []
    for marker, d in [('homa_ir', H), ('fasting_insulin', F)]:
        S = d.loc[d.bmi.notna()]
        full, wo, w = disc_fit(d, False), disc_fit(S, False), disc_fit(S, True)
        for g in GROUPS[1:]:
            rows.append(dict(panel='d', outcome=marker, term=g, label=LABELS[g], adjustment_without='age_sex', adjustment_with='age_sex_bmi',
                             n_same=w[g]['n'], n_group_same=w[g]['n_group'], est_without_same=wo[g]['estimate'],
                             ci_low_without_same=wo[g]['ci_low'], ci_high_without_same=wo[g]['ci_high'], p_without_same=wo[g]['p'],
                             est_with_bmi=w[g]['estimate'], ci_low_with_bmi=w[g]['ci_low'], ci_high_with_bmi=w[g]['ci_high'], p_with_bmi=w[g]['p'],
                             n_full=full[g]['n'], n_group_full=full[g]['n_group'], est_without_full=full[g]['estimate'],
                             ci_low_without_full=full[g]['ci_low'], ci_high_without_full=full[g]['ci_high'], p_without_full=full[g]['p'],
                             attenuation_pct=float(100 * (1 - w[g]['b'] / wo[g]['b'])), scale='percent_vs_both_normal'))
    return rows


def _rank_resid(x, y, g, Z):
    rx, ry, rg = stats.rankdata(x), stats.rankdata(y), stats.rankdata(g)
    X = np.column_stack([np.ones(len(x)), Z, rg])
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    return float(stats.spearmanr(x, y)[0]), float(np.corrcoef(ex, ey)[0, 1])


def part2(A_DISCORDANCE_BEHAVIOR, F, rng):
    D = A_DISCORDANCE_BEHAVIOR[['USER_ID', 'age_bucket', 'male', 'mean_glucose', 'steps']].merge(
        F[['USER_ID', 'logger_high', 'all_frequency', 'meal_primary_frequency', 'unlinked_primary_frequency']], on='USER_ID', how='left')
    xs = {'unlinked': 'unlinked_primary_frequency', 'meal_linked': 'meal_primary_frequency', 'all': 'all_frequency'}
    res = {}
    for scope, mask in [('consistent_loggers', D.logger_high.fillna(False).astype(bool)), ('all_primary', pd.Series(True, index=D.index))]:
        s = D.loc[mask].dropna(subset=['steps', 'mean_glucose', *xs.values()]).reset_index(drop=True)
        Z = pd.get_dummies(s.age_bucket, prefix='age', drop_first=True, dtype=float)
        Z = Z.loc[:, Z.sum() >= 10]; Z['male'] = s.male.astype(float); Z = Z.to_numpy()
        y, g = s.steps.to_numpy(), s.mean_glucose.to_numpy()
        pt = {k: _rank_resid(s[v].to_numpy(), y, g, Z) for k, v in xs.items()}
        boot = {k: [] for k in xs}
        n = len(s)
        for _ in range(N_BOOT):
            i = rng.integers(0, n, n)
            for k, v in xs.items():
                boot[k].append(_rank_resid(s[v].to_numpy()[i], y[i], g[i], Z[i]))
        out = {}
        for k in xs:
            b = np.array(boot[k])
            out[k] = dict(spearman=pt[k][0], spearman_ci=[float(np.percentile(b[:, 0], 2.5)), float(np.percentile(b[:, 0], 97.5))],
                          partial_spearman=pt[k][1], partial_ci=[float(np.percentile(b[:, 1], 2.5)), float(np.percentile(b[:, 1], 97.5))],
                          x_median=float(s[xs[k]].median()))
        du, dm = np.array(boot['unlinked']), np.array(boot['meal_linked'])
        diff = du - dm
        out['difference_unlinked_minus_meal_linked'] = dict(
            spearman=pt['unlinked'][0] - pt['meal_linked'][0], spearman_ci=[float(np.percentile(diff[:, 0], 2.5)), float(np.percentile(diff[:, 0], 97.5))],
            partial_spearman=pt['unlinked'][1] - pt['meal_linked'][1],
            partial_ci=[float(np.percentile(diff[:, 1], 2.5)), float(np.percentile(diff[:, 1], 97.5))])
        res[scope] = dict(n=int(n), steps_median=float(s.steps.median()), **out)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv); t0 = time.monotonic()
    guards = {}
    P = pd.read_csv(PANEL)
    P = P.loc[P.no_diabetes_range == True].copy()
    if not P.USER_ID.is_unique:
        raise SystemExit('GUARD FAILED: primary cohort has duplicate USER_ID')
    guards['primary_n'] = int(len(P))
    PRIMARY_N = int(len(P))
    A_DISCORDANCE_BEHAVIOR, _ = s_discordance_behavior.build()
    rows = part1(P, guards) + part1_discordance(P, A_DISCORDANCE_BEHAVIOR, guards)
    tab = pd.DataFrame(rows)
    # attenuation is only interpretable when the same-sample without-BMI CI excludes zero
    tab['attenuation_interpretable'] = (tab.ci_low_without_same * tab.ci_high_without_same) > 0
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / 'same_sample_bmi.csv', index=False)
    F = pd.read_parquet(FACETS)
    steps = part2(A_DISCORDANCE_BEHAVIOR, F, np.random.default_rng(SEED))
    steps_doc = dict(design='exploratory; between-participant; cross-sectional', population='primary cohort',
                     definitions=dict(frequency='spikes per eligible day, primary onset window (-15, +60 min), analysis/meals/spike_facets.py',
                                      steps='mean daily steps, dense CGM days within +/-90 local days of HbA1c anchor, days with >0 steps, >=7 days (analysis/metabolic/discordance_behavior.py)',
                                      partial='Pearson r of residual ranks after OLS on age-bucket dummies, male, rank(average glucose)',
                                      consistent_loggers='analysis/meals/spike_facets.py logger_high', bootstrap=dict(replicates=N_BOOT, seed=SEED, ci='percentile 95%')),
                     results=steps)
    (OUT / 'spike_frequency_steps.json').write_text(json.dumps(steps_doc, indent=1) + '\n')
    key = tab.set_index(['outcome', 'term'])
    summary = dict(design='cross-sectional associations; same BMI-complete participants with and without BMI', primary_n=PRIMARY_N,
                   rows={f'{o}|{t}': dict(n_same=int(r.n_same), without_same=float(r.est_without_same), with_bmi=float(r.est_with_bmi),
                                          n_full=int(r.n_full), without_full=float(r.est_without_full), attenuation_pct=float(r.attenuation_pct))
                         for (o, t), r in key.iterrows()},
                   unlinked_steps_consistent_loggers={k: steps['consistent_loggers'][k] for k in ('n', 'unlinked', 'meal_linked')},
                   guards_passed=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    meta = dict(analysis='bmi_same_sample', version=VERSION,
                design='cross-sectional associations; OLS HC3; complete-case; no imputation; no projection', seed=SEED,
                definitions=dict(outcomes=list(OUTCOMES), continuous=[dict(panel=p, measure=x, without=a, with_=b) for p, x, a, b in CONTINUOUS],
                                 adjustment_sets=s_insulin_markers.ADJ, discordance_cutoffs=dict(hba1c=5.7, mean_cgm=110, hba1c_rounding='decimal half-up'),
                                 attenuation='100*(1 - b_with/b_without_same), log-coefficient scale', bootstrap_replicates=N_BOOT),
                guards=guards,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'bmi_same_sample.metadata.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    print(json.dumps(dict(primary_n=PRIMARY_N, n_rows=len(tab)), default=float))


if __name__ == '__main__':
    main()
