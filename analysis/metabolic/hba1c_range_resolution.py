#!/usr/bin/env python3
"""Within-range HbA1c-CGM correlations and conversion errors, grouped by average glucose rather
than measured HbA1c.

Computes the within-range correlation between HbA1c and average CGM glucose, and the errors of
three HbA1c-from-glucose conversions, grouping by average CGM glucose rather than by measured
HbA1c (grouping by the measured value builds in regression to the mean).

Population: canonical primary cohort (no_diabetes_range == True).
HbA1c: anchor HbA1c, rounded decimal half-up with cgmpaper.lab_normalization.round_a1c before
any threshold. Ranges: <5.7% ("normal") and 5.7-6.4% ("prediabetes range"; in the primary cohort every
HbA1c >=5.7 is <6.5). Average glucose = window mean CGM glucose (mg/dL; panel mean_glucose).

Pairs are built by analysis/discordance/discordance_synthesis.py's load_panel_pairs(), for the
windows:
  symmetric_90d               canonical panel, dense days -90..+90 around the anchor   [primary]
  preceding_90d               preceding panel, dense days -89..0
  symmetric_90d_common_users  symmetric window, participants present in both windows
  preceding_90d_common_users  preceding window, same participants

Part 1  within_range_correlations.csv
  Pearson r between HbA1c and average CGM glucose (identical to r with GMI, a linear transform)
  overall, within <5.7 and within 5.7-6.4, per window. 95% CI: percentile, 2,000 participant
  bootstrap resamples of the window's participants (one pair per participant); range membership is
  re-evaluated in each resample, so the three r values and their differences (overall minus within)
  share resamples. Range descriptives (SD, IQR and distinct values of HbA1c; SD of average glucose)
  document the range restriction.
Part 2  conversion_errors_by_cgm.csv (main) and conversion_errors_by_hba1c.csv (supplement)
  Symmetric window. Error = measured HbA1c minus estimate, for
    GMI (Bergenstal et al. 2018)    3.31 + 0.02392 x average glucose
    ADAG eA1c (Nathan et al. 2008)  (average glucose + 46.7) / 28.7
    internal line                   OLS of HbA1c on average glucose in the full cohort (in-sample, descriptive)
  Summaries: n, mean and SD of the error, mean absolute error, % with |error| >=0.5 points, split into
  measured higher by >=0.5 and estimate higher by >=0.5. Groups: overall; average CGM glucose <110 vs
  >=110 mg/dL (the discordance cut-off defined in analysis/metabolic/insulin_markers.py); and, for the
  supplement only, measured HbA1c <5.7 vs 5.7-6.4. 95% CI: percentile participant bootstrap (2,000; the
  internal line is refit in every resample) and, for shares, Wilson score intervals from the counts.
  Note: grouping by the measured HbA1c selects on the quantity whose error is measured, so it builds in
  regression to the mean (low measured values carry negative errors and high values positive errors even
  if the conversion were unbiased); grouping by average CGM glucose does not select on HbA1c.

Inputs
  OUTPUTS/discordance/hba1c_cgm_discordance_synthesis/ (analysis/discordance/discordance_synthesis.py):
    discordance_synthesis.metadata.json, window_agreement_summary.csv, conversion_formula_sensitivity.csv,
    primary_symmetric_90d_pairs.parquet, primary_preceding_90d_pairs.parquet
  OUTPUTS/metabolic/insulin_markers/insulin_markers.metadata.json
    (analysis/metabolic/insulin_markers.py)
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv, canonical_lab_cgm_panel_preceding90.csv
    (analysis/canonical_panel/build_panel.py)
  LEVELS_DB (via load_panel_pairs())

Outputs (OUTPUTS/metabolic/hba1c_range_resolution/)
  within_range_correlations.csv, conversion_errors_by_cgm.csv, conversion_errors_by_hba1c.csv,
  guard_old_ed1a_rows.csv, summary.json, hba1c_range_resolution.metadata.json

Usage
  python analysis/metabolic/hba1c_range_resolution.py
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from cgmpaper.lab_normalization import round_a1c  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


S_DISCORDANCE_SYNTHESIS_PATH = 'analysis/discordance/discordance_synthesis.py'
s_discordance_synthesis = _load('discordance_synthesis_module', S_DISCORDANCE_SYNTHESIS_PATH)

VERSION = 'hba1c-range-resolution'
PUB = OUTPUTS
OUT = PUB / 'metabolic/hba1c_range_resolution'
SYN = PUB / 'discordance/hba1c_cgm_discordance_synthesis'
SYN_META = SYN / 'discordance_synthesis.metadata.json'
WINDOW_CSV = SYN / 'window_agreement_summary.csv'
CONV_CSV = SYN / 'conversion_formula_sensitivity.csv'
PAIRS = {'symmetric_90d': SYN / 'primary_symmetric_90d_pairs.parquet', 'preceding_90d': SYN / 'primary_preceding_90d_pairs.parquet'}
M_INSULIN_MARKERS = PUB / 'metabolic/insulin_markers/insulin_markers.metadata.json'
PANELS = [PUB / 'canonical_panel/canonical_lab_cgm_panel.csv', PUB / 'canonical_panel/canonical_lab_cgm_panel_preceding90.csv']
SEED = 20260928
N_BOOT = 2000
A1C_CUT = 5.7          # definition (prediabetes-range lower bound); checked against insulin_markers definitions
CGM_CUT = 110.0        # definition (CGM cut-off, mg/dL); checked against insulin_markers definitions
SHARE_CUT = 0.5        # HbA1c points
ADAG = (46.7, 28.7)
RANGE_DISCORDANCE_SYNTHESIS = {'all': 'all', 'lt_5_7': 'a1c_normal_lt_5_7', '5_7_to_6_4': 'a1c_elevated_ge_5_7'}
RANGE_LABEL = {'all': 'All HbA1c <6.5%', 'lt_5_7': 'HbA1c <5.7%', '5_7_to_6_4': 'HbA1c 5.7-6.4%'}
METHODS = {'GMI_Bergenstal_2018': 'GMI', 'eA1c_ADAG_2008': 'ADAG eA1c', 'internal_linear_calibration': 'Internal line'}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def pearson(a, g):
    if len(a) < 3 or np.std(a) == 0 or np.std(g) == 0:
        return np.nan
    return float(np.corrcoef(a, g)[0, 1])


def range_masks(a):
    return {'all': np.ones(len(a), bool), 'lt_5_7': a < A1C_CUT, '5_7_to_6_4': a >= A1C_CUT}


def windows(pairs):
    both = pairs.groupby('USER_ID').window.nunique()
    common = set(both.index[both == 2])
    out = {}
    for w in ('symmetric_90d', 'preceding_90d'):
        d = pairs.loc[pairs.window == w].sort_values('USER_ID').reset_index(drop=True)
        out[w] = d
        out[f'{w}_common_users'] = d.loc[d.USER_ID.isin(common)].reset_index(drop=True)
    return out


def estimates(a, e, fit_internal=True, coef=None):
    """Errors of the three conversions; the internal line is fit in the supplied sample."""
    gmi = s_discordance_synthesis.GMI_INTERCEPT + s_discordance_synthesis.GMI_SLOPE * e
    adag = (e + ADAG[0]) / ADAG[1]
    if fit_internal:
        slope, icpt = np.polyfit(e, a, 1)
    else:
        icpt, slope = coef
    return {'GMI_Bergenstal_2018': a - gmi, 'eA1c_ADAG_2008': a - adag, 'internal_linear_calibration': a - (icpt + slope * e)}, (icpt, slope)


def err_stats(d):
    return dict(mean_error=float(d.mean()), pct_abs_ge=float(100 * (np.abs(d) >= SHARE_CUT).mean()),
                pct_measured_higher=float(100 * (d >= SHARE_CUT).mean()), pct_estimate_higher=float(100 * (d <= -SHARE_CUT).mean()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv); t0 = time.monotonic(); guards = {}
    d_insulin_markers = json.loads(M_INSULIN_MARKERS.read_text())['definitions']['discordance_cutoffs']
    if (d_insulin_markers['hba1c'], d_insulin_markers['mean_cgm']) != (A1C_CUT, CGM_CUT):
        raise SystemExit(f'GUARD FAILED: cut-offs differ from insulin_markers definitions {d_insulin_markers}')
    guards['cutoffs_match_insulin_markers'] = dict(hba1c=d_insulin_markers['hba1c'], mean_cgm=d_insulin_markers['mean_cgm'])
    # ---- Structural check: re-built pairs equal the saved pairs; rounding idempotent
    pairs = s_discordance_synthesis.load_panel_pairs()
    pairs['a1c_rounded'] = round_a1c(pairs.a1c)
    g2 = dict(round_a1c_idempotent=bool(np.array_equal(pairs.a1c_rounded.to_numpy(), pairs.a1c.to_numpy())))
    for w, p in PAIRS.items():
        s = pd.read_parquet(p, columns=['USER_ID', 'a1c', 'mean_glucose']).sort_values('USER_ID').reset_index(drop=True)
        n = pairs.loc[pairs.window == w, ['USER_ID', 'a1c', 'mean_glucose']].sort_values('USER_ID').reset_index(drop=True)
        g2[w] = dict(n_rebuilt=int(len(n)), n_saved=int(len(s)), ids_equal=bool((s.USER_ID.to_numpy() == n.USER_ID.to_numpy()).all()) if len(s) == len(n) else False,
                     max_abs_diff_a1c=float(np.abs(s.a1c - n.a1c).max()) if len(s) == len(n) else np.inf,
                     max_abs_diff_glucose=float(np.abs(s.mean_glucose - n.mean_glucose).max()) if len(s) == len(n) else np.inf)
    guards['pairs_rebuilt'] = g2
    if not (g2['round_a1c_idempotent'] and all(g2[w]['ids_equal'] and g2[w]['max_abs_diff_a1c'] == 0 and g2[w]['max_abs_diff_glucose'] < 1e-9 for w in PAIRS)):
        raise SystemExit(f'GUARD FAILED: rebuilt pairs {g2}')
    pairs['a1c'] = pairs.a1c_rounded
    if (pairs.a1c >= 6.5).any():
        raise SystemExit('GUARD FAILED: primary pairs contain HbA1c >=6.5')
    W = windows(pairs)
    # ---- Within-range correlations vs saved synthesis
    saved = pd.read_csv(WINDOW_CSV).set_index(['window', 'cohort'])
    g3 = []
    for w, d in W.items():
        a, e = d.a1c.to_numpy(float), d.mean_glucose.to_numpy(float)
        gmi = s_discordance_synthesis.GMI_INTERCEPT + s_discordance_synthesis.GMI_SLOPE * e
        for rk, m in range_masks(a).items():
            r_glu, r_gmi = pearson(a[m], e[m]), pearson(a[m], gmi[m])
            ref = saved.loc[(w, RANGE_DISCORDANCE_SYNTHESIS[rk])]
            g3.append(dict(window=w, range=rk, n=int(m.sum()), n_saved=int(ref.n), r=r_glu, r_saved=float(ref.pearson_r),
                           abs_diff=abs(r_glu - float(ref.pearson_r)), r_gmi_minus_r_glucose=abs(r_gmi - r_glu)))
    g3 = pd.DataFrame(g3)
    ok3 = bool((g3.n == g3.n_saved).all() and (g3.abs_diff < 1e-12).all() and (g3.r_gmi_minus_r_glucose < 1e-12).all())
    guards['old_ed1a'] = dict(rows=int(len(g3)), all_rows_reproduced=ok3, max_abs_diff_r=float(g3.abs_diff.max()))
    if not ok3:
        raise SystemExit(f"GUARD FAILED: within-range correlations do not reproduce saved synthesis {guards['old_ed1a']}\n{g3}")
    # ---- Conversion error shares vs saved synthesis
    P = W['symmetric_90d']
    a, e = P.a1c.to_numpy(float), P.mean_glucose.to_numpy(float)
    E, coef = estimates(a, e)
    conv = pd.read_csv(CONV_CSV).set_index('conversion')
    g4 = {}
    for k, d in E.items():
        obs = 100 * float((np.abs(d) >= SHARE_CUT).mean())
        g4[k] = dict(n=int(len(d)), n_saved=int(conv.loc[k, 'n']), pct_observed=obs, pct_saved=float(conv.loc[k, 'pct_abs_ge_0_5']),
                     mean_error=float(d.mean()), mean_error_saved=float(conv.loc[k, 'mean_a1c_minus_estimate']))
    g4['internal_coefficients'] = dict(intercept=coef[0], slope=coef[1], intercept_saved=float(conv.loc['internal_linear_calibration', 'estimate_intercept']),
                                       slope_saved=float(conv.loc['internal_linear_calibration', 'estimate_slope_per_mgdl']))
    ok4 = all(v['n'] == v['n_saved'] and abs(v['pct_observed'] - v['pct_saved']) < 1e-9
              and abs(v['mean_error'] - v['mean_error_saved']) < 1e-9 for k, v in g4.items() if k in ('GMI_Bergenstal_2018', 'eA1c_ADAG_2008', 'internal_linear_calibration'))
    ok4 &= abs(coef[0] - g4['internal_coefficients']['intercept_saved']) < 1e-9 and abs(coef[1] - g4['internal_coefficients']['slope_saved']) < 1e-12
    guards['old_ed1b'] = dict(ok=bool(ok4), **g4)
    if not ok4:
        raise SystemExit(f"GUARD FAILED: conversion error shares do not reproduce saved synthesis {guards['old_ed1b']}")
    guards['all_passed'] = True

    # ---- Part 1: within-range correlations with participant bootstrap
    rng = np.random.default_rng(SEED)
    rows = []
    for w, d in W.items():
        a, e = d.a1c.to_numpy(float), d.mean_glucose.to_numpy(float)
        n = len(a)
        pt = {rk: pearson(a[m], e[m]) for rk, m in range_masks(a).items()}
        B = np.empty((N_BOOT, 3))
        for b in range(N_BOOT):
            i = rng.integers(0, n, n); ab, eb = a[i], e[i]
            B[b] = [pearson(ab[m], eb[m]) for m in range_masks(ab).values()]
        for j, (rk, m) in enumerate(range_masks(a).items()):
            q = np.nanpercentile(B[:, j], [2.5, 97.5]); dq = np.nanpercentile(B[:, 0] - B[:, j], [2.5, 97.5])
            aa, ee = a[m], e[m]
            rows.append(dict(window=w, range=rk, range_label=RANGE_LABEL[rk], n=int(m.sum()), pearson_r=pt[rk], ci_low=float(q[0]), ci_high=float(q[1]),
                             r_overall_minus_within=pt['all'] - pt[rk], diff_ci_low=float(dq[0]), diff_ci_high=float(dq[1]),
                             hba1c_sd=float(aa.std(ddof=1)), hba1c_q25=float(np.percentile(aa, 25)), hba1c_q75=float(np.percentile(aa, 75)),
                             hba1c_min=float(aa.min()), hba1c_max=float(aa.max()), hba1c_distinct_values=int(len(np.unique(aa))),
                             avg_glucose_sd=float(ee.std(ddof=1)), avg_glucose_median=float(np.median(ee)),
                             sensor_days_median=float(np.median(d.sensor_days.to_numpy(float)[m])),
                             n_boot=N_BOOT, n_boot_nan=int(np.isnan(B[:, j]).sum())))
    R = pd.DataFrame(rows)

    # ---- Part 2: conversion errors (symmetric window)
    a, e = P.a1c.to_numpy(float), P.mean_glucose.to_numpy(float)
    n = len(a)
    groupings = {'by_average_cgm': {'all': ('All', np.ones(n, bool)), 'cgm_lt_110': ('Average CGM glucose <110 mg/dL', e < CGM_CUT),
                                    'cgm_ge_110': ('Average CGM glucose >=110 mg/dL', e >= CGM_CUT)},
                 'by_measured_hba1c': {'all': ('All', np.ones(n, bool)), 'hba1c_lt_5_7': ('Measured HbA1c <5.7%', a < A1C_CUT),
                                       'hba1c_5_7_to_6_4': ('Measured HbA1c 5.7-6.4%', a >= A1C_CUT)}}
    boot = {g: {k: {m: [] for m in E} for k in groupings[g]} for g in groupings}
    for b in range(N_BOOT):
        i = rng.integers(0, n, n); ab, eb = a[i], e[i]
        Eb, _ = estimates(ab, eb)
        gm = {'by_average_cgm': {'all': np.ones(n, bool), 'cgm_lt_110': eb < CGM_CUT, 'cgm_ge_110': eb >= CGM_CUT},
              'by_measured_hba1c': {'all': np.ones(n, bool), 'hba1c_lt_5_7': ab < A1C_CUT, 'hba1c_5_7_to_6_4': ab >= A1C_CUT}}
        for g, ms in gm.items():
            for k, m in ms.items():
                for meth, d in Eb.items():
                    boot[g][k][meth].append(err_stats(d[m]))
    tabs = {}
    for g, groups in groupings.items():
        out = []
        for k, (lab, m) in groups.items():
            for meth, d in E.items():
                dd = d[m]; st = err_stats(dd); bs = pd.DataFrame(boot[g][k][meth])
                k_abs, k_hi, k_lo = int((np.abs(dd) >= SHARE_CUT).sum()), int((dd >= SHARE_CUT).sum()), int((dd <= -SHARE_CUT).sum())
                row = dict(grouping=g, group=k, group_label=lab, conversion=meth, conversion_label=METHODS[meth], n=int(m.sum()),
                           avg_glucose_median=float(np.median(e[m])), hba1c_mean=float(a[m].mean()),
                           mean_error=st['mean_error'], sd_error=float(dd.std(ddof=1)), mae=float(np.abs(dd).mean()),
                           n_abs_ge_0_5=k_abs, pct_abs_ge_0_5=st['pct_abs_ge'], n_measured_higher_ge_0_5=k_hi,
                           pct_measured_higher_ge_0_5=st['pct_measured_higher'], n_estimate_higher_ge_0_5=k_lo,
                           pct_estimate_higher_ge_0_5=st['pct_estimate_higher'])
                for c, key in (('mean_error', 'mean_error'), ('pct_abs_ge_0_5', 'pct_abs_ge'), ('pct_measured_higher_ge_0_5', 'pct_measured_higher'),
                               ('pct_estimate_higher_ge_0_5', 'pct_estimate_higher')):
                    q = np.percentile(bs[key].to_numpy(float), [2.5, 97.5]); row[f'{c}_boot_low'], row[f'{c}_boot_high'] = float(q[0]), float(q[1])
                row['pct_abs_ge_0_5_wilson_low'], row['pct_abs_ge_0_5_wilson_high'] = wilson(k_abs, int(m.sum()))
                row['internal_line_fit_scope'] = 'fit once in the full primary cohort (refit in every bootstrap resample)' if meth == 'internal_linear_calibration' else 'fixed published equation'
                out.append(row)
        tabs[g] = pd.DataFrame(out)
    # sanity: 'all' rows in both groupings are identical and equal guard 4
    for g in tabs:
        al = tabs[g].loc[tabs[g].group == 'all'].set_index('conversion')
        for k in ('GMI_Bergenstal_2018', 'eA1c_ADAG_2008', 'internal_linear_calibration'):
            if abs(al.loc[k, 'pct_abs_ge_0_5'] - g4[k]['pct_observed']) > 1e-12:
                raise SystemExit('GUARD FAILED: overall conversion rows differ from the earlier computation')
    OUT.mkdir(parents=True, exist_ok=True)
    files = {'within_range_correlations.csv': R, 'conversion_errors_by_cgm.csv': tabs['by_average_cgm'],
             'conversion_errors_by_hba1c.csv': tabs['by_measured_hba1c'], 'guard_old_ed1a_rows.csv': g3}
    for nme, f in files.items():
        f.to_csv(OUT / nme, index=False)
    sym = R.loc[R.window == 'symmetric_90d'].set_index('range')
    cg = tabs['by_average_cgm'].set_index(['group', 'conversion'])
    summary = dict(
        design='cross-sectional agreement; primary cohort; no projection', primary_n=int(len(P)),
        within_range_symmetric={rk: dict(n=int(r.n), r=float(r.pearson_r), ci=[float(r.ci_low), float(r.ci_high)], hba1c_sd=float(r.hba1c_sd),
                                         hba1c_distinct_values=int(r.hba1c_distinct_values), avg_glucose_sd=float(r.avg_glucose_sd))
                                for rk, r in sym.iterrows()},
        conversion_by_cgm={f'{g}|{c}': dict(n=int(r.n), mean_error=float(r.mean_error), pct_abs_ge_0_5=float(r.pct_abs_ge_0_5),
                                            pct_measured_higher=float(r.pct_measured_higher_ge_0_5), pct_estimate_higher=float(r.pct_estimate_higher_ge_0_5))
                           for (g, c), r in cg.iterrows()},
        regression_to_the_mean_note=('Grouping by measured HbA1c selects on the quantity whose error is measured: participants with low measured '
                                     'values tend to have negative errors and those with high values positive errors even for an unbiased '
                                     'conversion. conversion_errors_by_hba1c.csv is for the supplement only; the main grouping is by average CGM '
                                     'glucose, which does not select on HbA1c.'),
        range_restriction=dict(
            note=('Restricting HbA1c to a narrow range removes between-person spread (hba1c_sd, hba1c_distinct_values), which by itself lowers '
                  'a correlation; whether r is lower within a range than overall is read from r_overall_minus_within and its bootstrap CI.'),
            lower_than_overall={rk: bool(r.diff_ci_low > 0) for rk, r in sym.iterrows() if rk != 'all'},
            hba1c_sd={rk: float(r.hba1c_sd) for rk, r in sym.iterrows()}),
        guards_passed=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    md = dict(analysis='hba1c_range_resolution', version=VERSION, status='passed',
              design='cross-sectional agreement; Pearson r; participant bootstrap; no projection', seed=SEED,
              definitions=dict(hba1c_rounding='decimal half-up (round_a1c)', ranges={'lt_5_7': '<5.7', '5_7_to_6_4': '5.7-6.4 (all <6.5 in primary)'},
                               cgm_cutoff_mg_dl=CGM_CUT, share_cutoff_points=SHARE_CUT, windows=list(W),
                               conversions=dict(GMI=f'{s_discordance_synthesis.GMI_INTERCEPT} + {s_discordance_synthesis.GMI_SLOPE} x average glucose',
                                                ADAG=f'(average glucose + {ADAG[0]}) / {ADAG[1]}',
                                                internal=dict(intercept=coef[0], slope=coef[1], fit='OLS HbA1c on average glucose, full primary cohort')),
                               bootstrap=dict(replicates=N_BOOT, unit='participant', ci='percentile 95%'), wilson='score interval, z=1.96'),
              guards=guards, runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'hba1c_range_resolution.metadata.json').write_text(json.dumps(md, indent=1, default=float) + '\n')
    print(json.dumps(dict(guards={k: guards[k] for k in ('old_ed1a',)}, ed1b={k: v['pct_observed'] for k, v in g4.items() if k in ('GMI_Bergenstal_2018', 'eA1c_ADAG_2008', 'internal_linear_calibration')},
                          within=summary['within_range_symmetric']), indent=1, default=float))


if __name__ == '__main__':
    main()
