#!/usr/bin/env python3
"""Head-to-head comparison of average CGM glucose and HbA1c against insulin markers.

Questions.
  Q1 Do the two discordant groups differ from each other? CGM elevated only (average CGM >=110 mg/dL, HbA1c
     <5.7%) versus HbA1c elevated only (HbA1c >=5.7%, average CGM <110), with HbA1c-only as the reference.
  Q2 In the same participants, which single glycemic measure is more strongly associated with the insulin
     markers? Average CGM glucose and HbA1c are compared per SD, mutually adjusted, and by the variance they
     explain (in-sample R^2 with participant bootstrap; participant-disjoint 10-fold cross-validated R^2 as a
     sensitivity). Whole primary cohort and HbA1c-normal range (<5.7%).
Outcomes: HOMA-IR (the primary outcome), fasting insulin (larger sample; contains no glucose term, so it avoids the
shared fasting-glucose component of HOMA-IR), triglycerides (companion).

Population and definitions follow analysis/metabolic/insulin_markers.py and analysis/metabolic/bmi_same_sample.py:
canonical primary cohort (no diabetes-range labs); anchor HbA1c rounded decimal half-up (round_a1c); average CGM
glucose over dense days +/-90 days of the draw; other analytes from the same draw (+/-7 days). OLS of log marker
with HC3 intervals; age bucket and sex; BMI models run in the BMI-complete participants, and each "+ BMI" row has
a matching "same sample, without BMI" row. Cross-sectional associations only; cross-validated R^2 is a
model-comparison sensitivity, not a claim about prospective prediction.

Inputs: OUTPUTS/metabolic/insulin_markers (analysis/metabolic/insulin_markers.py) for the canonical
panel and discordance reference rows; OUTPUTS/metabolic/bmi_same_sample (analysis/metabolic/bmi_same_sample.py)
for the same-sample reference rows.
Outputs: OUTPUTS/metabolic/cgm_vs_hba1c_head_to_head/discordant_groups_head_to_head.csv,
continuous_head_to_head.csv, and summary.json. No participant identifiers are written.
Usage: python analysis/metabolic/cgm_vs_hba1c.py [--out PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from cgmpaper.lab_normalization import round_a1c  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


s_insulin_markers = _load('insulin_markers_for_cgm_vs_hba1c', 'analysis/metabolic/insulin_markers.py')
VERSION = 'cgm-vs-hba1c-head-to-head'
MP = OUTPUTS / 'metabolic'
OUT = MP / 'cgm_vs_hba1c_head_to_head'
D_INSULIN_MARKERS, D_BMI_SAME_SAMPLE = MP / 'insulin_markers', MP / 'bmi_same_sample'
OUTCOMES = ['homa_ir', 'fasting_insulin', 'triglycerides']
TOL = 1e-9
SEED = 20260928
N_BOOT = 2000
N_CV_REPEATS = 20


def pct(v):
    return 100 * (np.exp(v) - 1)


def design(s, cols, bmi):
    X = pd.DataFrame(index=s.index)
    for c in cols:
        X[c] = s[c].astype(float) if s[c].dtype != float else s[c]
    ages = pd.get_dummies(s.age_bucket, prefix='age', drop_first=True, dtype=float)
    X = X.join(ages.loc[:, ages.sum() >= 10]); X['male'] = s.male
    if bmi:
        X['bmi'] = (s.bmi - s.bmi.mean()) / s.bmi.std()
    return sm.add_constant(X)


def outcome(P, y):
    d = P.loc[P[y].notna() & (P[y] > 0)].copy()
    d['y'] = np.log(d[y])
    return d.reset_index(drop=True)


def group_fit(d, ref, bmi):
    G = pd.get_dummies(d.group, dtype=float).drop(columns=ref)
    X = design(d.join(G), list(G.columns), bmi)
    return sm.OLS(d.y.to_numpy(), X).fit(cov_type='HC3')


def continuous(d, cols, bmi):
    s = d.copy()
    for c in cols:
        s[c] = (s[c] - s[c].mean()) / s[c].std()
    return sm.OLS(s.y.to_numpy(), design(s, cols, bmi)).fit(cov_type='HC3')


def r2(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))


def cv_r2(y, X, folds):
    pred = np.empty_like(y)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        pred[te] = X[te] @ b
    return 1 - ((y - pred) @ (y - pred)) / ((y - y.mean()) @ (y - y.mean()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(argv); t0 = time.monotonic(); guards = {}
    panel, P, _ = s_insulin_markers.load(None)
    guards['primary_n'] = int(len(P))
    a1c_r = round_a1c(P.a1c.to_numpy())
    P['a1c_r'] = a1c_r
    a1c_hi, cgm_hi = a1c_r >= 5.7, P.mean_glucose.to_numpy() >= 110
    P['group'] = np.select([~a1c_hi & ~cgm_hi, ~a1c_hi & cgm_hi, a1c_hi & ~cgm_hi], ['both_normal', 'cgm_only', 'a1c_only'], 'both_elevated')
    H = outcome(P, 'homa_ir')
    counts = {g: [int((H.loc[H.group == g, 'homa_ir'] >= 2.5).sum()), int((H.group == g).sum())] for g in ['both_normal', 'cgm_only', 'a1c_only', 'both_elevated']}
    guards['discordance_counts'] = counts
    # Reproduce the insulin_markers.py discordance rows (reference both_normal)
    ref = pd.read_csv(D_INSULIN_MARKERS / 'discordance_homa_bmi.csv').set_index(['adjustment', 'group'])
    worst = 0.0
    for adj, sub, bmi in [('age_sex', H, False), ('age_sex_bmi_subset', H.loc[H.bmi.notna()].reset_index(drop=True), False),
                          ('bmi', H.loc[H.bmi.notna()].reset_index(drop=True), True)]:
        m = group_fit(sub, 'both_normal', bmi); ci = m.conf_int()
        for g in ['cgm_only', 'a1c_only', 'both_elevated']:
            q = ref.loc[(adj, g)]
            worst = max(worst, abs(pct(m.params[g]) - q.pct_vs_both_normal), abs(pct(ci.loc[g, 0]) - q.ci_low),
                        abs(pct(ci.loc[g, 1]) - q.ci_high), abs(m.pvalues[g] - q.p))
    guards['s_insulin_markers_discordance_max_abs_diff'] = float(worst)
    # Reproduce the bmi_same_sample.py same-sample rows
    r_bmi_same_sample = pd.read_csv(D_BMI_SAME_SAMPLE / 'same_sample_bmi.csv')
    r_bmi_same_sample = r_bmi_same_sample[(r_bmi_same_sample.panel == 'd') & r_bmi_same_sample.term.isin(['cgm_only', 'a1c_only'])]
    worst = 0.0
    for _, q in r_bmi_same_sample.iterrows():
        d = outcome(P, q.outcome); d = d.loc[d.bmi.notna()].reset_index(drop=True)
        for bmi, col in [(False, 'est_without_same'), (True, 'est_with_bmi')]:
            worst = max(worst, abs(pct(group_fit(d, 'both_normal', bmi).params[q.term]) - q[col]))
    guards['bmi_same_sample_max_abs_diff'] = float(worst)

    # ---------------- Q1: CGM-only vs HbA1c-only ----------------
    q1 = []
    for y in OUTCOMES:
        d = outcome(P, y)
        for label, sub, bmi in [('age_sex', d, False), ('age_sex_same_bmi_sample', d.loc[d.bmi.notna()].reset_index(drop=True), False),
                                ('age_sex_bmi', d.loc[d.bmi.notna()].reset_index(drop=True), True)]:
            m = group_fit(sub, 'a1c_only', bmi); ci = m.conf_int()
            q1.append(dict(outcome=y, adjustment=label, contrast='cgm_only vs a1c_only', n=int(len(sub)),
                           n_cgm_only=int((sub.group == 'cgm_only').sum()), n_a1c_only=int((sub.group == 'a1c_only').sum()),
                           pct_difference=float(pct(m.params['cgm_only'])), ci_low=float(pct(ci.loc['cgm_only', 0])),
                           ci_high=float(pct(ci.loc['cgm_only', 1])), p=float(m.pvalues['cgm_only']),
                           geo_mean_cgm_only=float(np.exp(sub.loc[sub.group == 'cgm_only', 'y'].mean())),
                           geo_mean_a1c_only=float(np.exp(sub.loc[sub.group == 'a1c_only', 'y'].mean()))))
    Q1 = pd.DataFrame(q1)
    # HOMA-IR >= 2.5, descriptive two-group comparison
    a, b = counts['cgm_only'], counts['a1c_only']
    rd = a[0] / a[1] - b[0] / b[1]; se = np.sqrt(a[0] / a[1] * (1 - a[0] / a[1]) / a[1] + b[0] / b[1] * (1 - b[0] / b[1]) / b[1])
    fisher = stats.fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]])
    q1_threshold = dict(cgm_only=f'{a[0]}/{a[1]}', a1c_only=f'{b[0]}/{b[1]}', risk_difference_pct_points=100 * rd,
                        ci_low=100 * (rd - 1.96 * se), ci_high=100 * (rd + 1.96 * se), fisher_odds_ratio=float(fisher[0]), fisher_p=float(fisher[1]))

    # ---------------- Q2: continuous head-to-head, same participants ----------------
    rng = np.random.default_rng(SEED)
    q2 = []
    for y in OUTCOMES:
        base = outcome(P, y).dropna(subset=['mean_glucose', 'a1c_r'])
        base['hba1c'] = base.a1c_r
        for scope, sub0 in [('all_primary', base), ('hba1c_normal_lt57', base.loc[base.hba1c < 5.7])]:
            for bmi_label, bmi in [('no_bmi_same_sample', False), ('bmi', True), ('no_bmi_full', False)]:
                sub = sub0.loc[sub0.bmi.notna()] if bmi_label != 'no_bmi_full' else sub0
                sub = sub.reset_index(drop=True)
                row = dict(outcome=y, scope=scope, adjustment=bmi_label, n=int(len(sub)))
                for name, cols in [('cgm', ['mean_glucose']), ('hba1c', ['hba1c']), ('mutual', ['mean_glucose', 'hba1c'])]:
                    m = continuous(sub, cols, bmi); ci = m.conf_int()
                    for c in cols:
                        tag = f'{name}_{"cgm" if c == "mean_glucose" else "hba1c"}' if name == 'mutual' else name
                        row.update({f'{tag}_pct_per_sd': float(pct(m.params[c])), f'{tag}_ci_low': float(pct(ci.loc[c, 0])),
                                    f'{tag}_ci_high': float(pct(ci.loc[c, 1])), f'{tag}_p': float(m.pvalues[c])})
                # R^2: covariates only, + CGM, + HbA1c, + both
                s = sub.copy()
                for c in ('mean_glucose', 'hba1c'):
                    s[c] = (s[c] - s[c].mean()) / s[c].std()
                Xs = {k: design(s, cols, bmi).to_numpy() for k, cols in
                      [('cov', []), ('cgm', ['mean_glucose']), ('hba1c', ['hba1c']), ('both', ['mean_glucose', 'hba1c'])]}
                yv = s.y.to_numpy()
                R = {k: r2(yv, X) for k, X in Xs.items()}
                row.update({f'r2_{k}': float(v) for k, v in R.items()})
                row['delta_r2_cgm'] = R['cgm'] - R['cov']; row['delta_r2_hba1c'] = R['hba1c'] - R['cov']
                row['cgm_minus_hba1c_r2'] = R['cgm'] - R['hba1c']
                row['cgm_unique_r2'] = R['both'] - R['hba1c']; row['hba1c_unique_r2'] = R['both'] - R['cgm']
                bs = np.empty(N_BOOT)
                for i in range(N_BOOT):
                    idx = rng.integers(0, len(yv), len(yv))
                    bs[i] = r2(yv[idx], Xs['cgm'][idx]) - r2(yv[idx], Xs['hba1c'][idx])
                row['cgm_minus_hba1c_r2_ci_low'], row['cgm_minus_hba1c_r2_ci_high'] = map(float, np.percentile(bs, [2.5, 97.5]))
                row['boot_share_cgm_higher'] = float((bs > 0).mean())
                cvd = []
                for rep in range(N_CV_REPEATS):
                    folds = rng.permutation(np.arange(len(yv)) % 10)
                    cvd.append(cv_r2(yv, Xs['cgm'], folds) - cv_r2(yv, Xs['hba1c'], folds))
                    if rep == 0:
                        row['cv_r2_cgm'] = float(cv_r2(yv, Xs['cgm'], folds)); row['cv_r2_hba1c'] = float(cv_r2(yv, Xs['hba1c'], folds))
                row['cv_cgm_minus_hba1c_r2_mean'] = float(np.mean(cvd)); row['cv_repeats_cgm_higher'] = int(np.sum(np.array(cvd) > 0))
                row['rho_cgm_hba1c'] = float(stats.spearmanr(sub.mean_glucose, sub.hba1c)[0])
                q2.append(row)
    Q2 = pd.DataFrame(q2)
    args.out.mkdir(parents=True, exist_ok=True)
    files = {'discordant_groups_head_to_head.csv': Q1, 'continuous_head_to_head.csv': Q2}
    for n, df in files.items():
        df.to_csv(args.out / n, index=False)
    summary = dict(version=VERSION, guards=guards, homa_ge25_cgm_only_vs_a1c_only=q1_threshold,
                   design='cross-sectional associations; same-participant comparisons; CV R^2 is a model-comparison sensitivity')
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    meta = dict(version=VERSION, seconds=round(time.monotonic() - t0, 1))
    (args.out / 'cgm_vs_hba1c_head_to_head.metadata.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(guards)[:500])


if __name__ == '__main__':
    main()
