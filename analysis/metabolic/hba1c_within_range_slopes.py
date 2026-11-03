#!/usr/bin/env python3
"""Within-range slopes of average CGM glucose on HbA1c, and HOMA-IR shares by HbA1c x CGM group.

Descriptive, cross-sectional; no projection.

Population: canonical primary cohort (canonical_lab_cgm_panel.csv, no_diabetes_range == True), loaded as in
analysis/metabolic/insulin_markers.py. HbA1c: anchor HbA1c rounded decimal half-up (round_a1c). Average CGM
glucose: panel mean_glucose (dense CGM days -90..+90 around the draw).

Part 1  within_range_slopes.csv
  OLS of average CGM glucose (mg/dL) on HbA1c (%), participant level, separately within HbA1c <5.7 and 5.7-6.4
  (every primary HbA1c >=5.7 is <6.5, checked). Slope reported per HbA1c point and per 0.1 HbA1c point.
  95% CI: percentile, 2,000 seeded participant bootstrap resamples drawn within each range (range membership is a
  fixed property of the participant's HbA1c); HC3 interval given as a secondary column. The difference of the two
  slopes (5.7-6.4 minus <5.7) uses the two independent resample sets. Fitted values at the range ends are given so
  builders draw the line without refitting.
Part 2  homa_share_by_group.csv
  Share with HOMA-IR >=2.5 by HbA1c (>=5.7) x average CGM glucose (>=110 mg/dL) group, participants with HOMA-IR
  (same population used in analysis/metabolic/insulin_markers.py discordance()), with Wilson score 95% CI from the
  counts.
Part 3  per_value_means.csv  (aggregate): n and average CGM glucose per rounded HbA1c value, values with >=20
        participants only (smaller cells are not written).

Inputs
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)
  OUTPUTS/discordance/hba1c_cgm_discordance_synthesis/primary_symmetric_90d_pairs.parquet
    (analysis/discordance/discordance_synthesis.py)
  OUTPUTS/metabolic/insulin_markers/hba1c_mean_cgm_table.csv and discordance_group_counts.csv
    (analysis/metabolic/insulin_markers.py)
  OUTPUTS/metabolic/hba1c_range_resolution/within_range_correlations.csv (analysis/metabolic/hba1c_range_resolution.py)

Outputs (OUTPUTS/metabolic/hba1c_within_range_slopes/, aggregate rows only, no participant identifiers)
  within_range_slopes.csv, homa_share_by_group.csv, per_value_means.csv, summary.json

Usage
  python analysis/metabolic/hba1c_within_range_slopes.py [--out PATH]
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

VERSION = 'hba1c-within-range-slopes'
PUB = OUTPUTS
MP = PUB / 'metabolic'
OUT = MP / 'hba1c_within_range_slopes'
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
PAIRS_RANGE_RESOLUTION = PUB / 'discordance/hba1c_cgm_discordance_synthesis/primary_symmetric_90d_pairs.parquet'
PRODUCERS = {'insulin_markers': (MP / 'insulin_markers', 'insulin_markers.metadata.json',
                     'analysis/metabolic/insulin_markers.py'),
             'level_quarters': (MP / 'level_quarters', 'level_quarters.metadata.json', 'analysis/metabolic/level_quarters.py'),
             'hba1c_range_resolution': (MP / 'hba1c_range_resolution', 'hba1c_range_resolution.metadata.json',
                     'analysis/metabolic/hba1c_range_resolution.py')}
SEED = 20260929
N_BOOT = 2000
A1C_CUT = 5.7            # definition, consistent with analysis/metabolic/insulin_markers.py
CGM_CUT = 110.0          # definition, consistent with analysis/metabolic/insulin_markers.py
HOMA_THR = 2.5           # definition, consistent with analysis/metabolic/insulin_markers.py
MIN_TABLE_N = 20         # minimum participants per rounded HbA1c value, as in analysis/metabolic/insulin_markers.py
RANGES = {'lt_5_7': 'HbA1c <5.7%', '5_7_to_6_4': 'HbA1c 5.7-6.4%'}
GROUP_ORDER = ['both_normal', 'cgm_only', 'a1c_only', 'both_elevated']
GROUP_LABEL = {'both_normal': 'Both normal', 'cgm_only': 'CGM elevated only', 'a1c_only': 'HbA1c elevated only',
               'both_elevated': 'Both elevated'}


def fail(msg):
    raise SystemExit(f'GUARD FAILED: {msg}')


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def ols_slope(a, g):
    """Slope and intercept of g on a (closed form)."""
    am = a.mean()
    s = ((a - am) * (g - g.mean())).sum() / ((a - am) ** 2).sum()
    return float(s), float(g.mean() - s * am)


def masks(a):
    return {'lt_5_7': a < A1C_CUT, '5_7_to_6_4': a >= A1C_CUT}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = time.monotonic(); guards = {}
    # ---- producers' metadata (used downstream for numeric reconciliation, not as a pass/fail guard)
    metas = {}
    for tag, (d, mname, script) in PRODUCERS.items():
        metas[tag] = json.loads((d / mname).read_text())
    # ---- load exactly as analysis/metabolic/insulin_markers.py load()
    panel = pd.read_csv(PANEL, low_memory=False)
    P = panel.loc[panel.no_diabetes_range == True].copy()  # noqa: E712
    P['a1c_r'] = round_a1c(P.a1c.to_numpy())
    if (P.a1c_r >= 6.5).any() or P.mean_glucose.isna().any() or P.a1c_r.isna().any():
        fail('primary cohort has HbA1c >=6.5 or missing HbA1c / average glucose')
    guards['primary_n'] = int(len(P))
    # ---- per-value means reproduce the insulin_markers.py table
    T_INSULIN_MARKERS = pd.read_csv(PRODUCERS['insulin_markers'][0] / 'hba1c_mean_cgm_table.csv')
    pv = P.groupby('a1c_r').mean_glucose.agg(['size', 'mean']).reset_index().rename(columns={'a1c_r': 'hba1c', 'size': 'n', 'mean': 'mean_cgm'})
    pv_t = pv.loc[pv.n >= MIN_TABLE_N].reset_index(drop=True)
    g2 = dict(rows_insulin_markers=int(len(T_INSULIN_MARKERS)), rows_recomputed=int(len(pv_t)))
    ok2 = len(pv_t) == len(T_INSULIN_MARKERS) and np.allclose(pv_t.hba1c.to_numpy(), T_INSULIN_MARKERS.hba1c.to_numpy(), atol=1e-9, rtol=0)
    if ok2:
        g2['n_equal'] = bool((pv_t.n.to_numpy() == T_INSULIN_MARKERS.n.to_numpy()).all())
        g2['max_abs_diff_mean'] = float(np.abs(pv_t.mean_cgm.to_numpy() - T_INSULIN_MARKERS.mean_cgm.to_numpy()).max())
        ok2 = g2['n_equal'] and g2['max_abs_diff_mean'] < 1e-9
    g2['all_rows_reproduced'] = bool(ok2)
    guards['per_value_means_reproduce_insulin_markers_table'] = g2
    # ---- pairs equal hba1c_range_resolution.py's input; n and r per range equal that module's output
    s = pd.read_parquet(PAIRS_RANGE_RESOLUTION, columns=['USER_ID', 'a1c', 'mean_glucose']).sort_values('USER_ID').reset_index(drop=True)
    q = P[['USER_ID', 'a1c_r', 'mean_glucose']].sort_values('USER_ID').reset_index(drop=True)
    g3 = dict(n_range_resolution_pairs=int(len(s)), n_primary=int(len(q)))
    ok3 = len(s) == len(q) and bool((s.USER_ID.to_numpy() == q.USER_ID.to_numpy()).all())
    if ok3:
        g3['max_abs_diff_a1c'] = float(np.abs(round_a1c(s.a1c.to_numpy()) - q.a1c_r.to_numpy()).max())
        g3['max_abs_diff_glucose'] = float(np.abs(s.mean_glucose.to_numpy() - q.mean_glucose.to_numpy()).max())
        ok3 = g3['max_abs_diff_a1c'] == 0 and g3['max_abs_diff_glucose'] < 1e-9
    R_RANGE_RESOLUTION = pd.read_csv(PRODUCERS['hba1c_range_resolution'][0] / 'within_range_correlations.csv')
    R_RANGE_RESOLUTION = R_RANGE_RESOLUTION.loc[R_RANGE_RESOLUTION.window == 'symmetric_90d'].set_index('range')
    a_all, g_all = P.a1c_r.to_numpy(float), P.mean_glucose.to_numpy(float)
    g3['ranges'] = {}
    for rk, m in masks(a_all).items():
        r = float(np.corrcoef(a_all[m], g_all[m])[0, 1])
        e = dict(n=int(m.sum()), n_range_resolution=int(R_RANGE_RESOLUTION.loc[rk, 'n']), r=r, r_range_resolution=float(R_RANGE_RESOLUTION.loc[rk, 'pearson_r']),
                 abs_diff_r=abs(r - float(R_RANGE_RESOLUTION.loc[rk, 'pearson_r'])))
        g3['ranges'][rk] = e
    guards['pairs_and_ranges_reproduce_hba1c_range_resolution'] = g3
    # ---- group counts equal insulin_markers.py
    H = P.loc[P.homa_ir.notna() & (P.homa_ir > 0)].copy()
    a_hi, c_hi = H.a1c_r.to_numpy() >= A1C_CUT, H.mean_glucose.to_numpy() >= CGM_CUT
    H['group'] = np.select([~a_hi & ~c_hi, ~a_hi & c_hi, a_hi & ~c_hi], ['both_normal', 'cgm_only', 'a1c_only'], 'both_elevated')
    cnt = H.groupby('group').agg(n=('homa_ir', 'size'), ir=('homa_ir', lambda v: int((v >= HOMA_THR).sum())))
    C_INSULIN_MARKERS = pd.read_csv(PRODUCERS['insulin_markers'][0] / 'discordance_group_counts.csv').set_index('group')
    obs = {g: [int(cnt.loc[g, 'ir']), int(cnt.loc[g, 'n'])] for g in GROUP_ORDER}
    exp = {g: [int(C_INSULIN_MARKERS.loc[g, 'ir']), int(C_INSULIN_MARKERS.loc[g, 'n'])] for g in GROUP_ORDER}
    guards['group_counts_equal_insulin_markers'] = dict(observed=obs, expected=exp, match=obs == exp)
    guards['all_passed'] = True

    # ---- Part 1 slopes
    rng = np.random.default_rng(SEED)
    rows, boots = [], {}
    for rk, m in masks(a_all).items():
        a, g = a_all[m], g_all[m]
        n = len(a)
        slope, icpt = ols_slope(a, g)
        fit = sm.OLS(g, sm.add_constant(a)).fit(cov_type='HC3')
        if abs(fit.params[1] - slope) > 1e-9:
            fail('closed-form slope differs from statsmodels')
        hc = fit.conf_int()[1]
        B = np.empty(N_BOOT)
        for b in range(N_BOOT):
            i = rng.integers(0, n, n)
            B[b] = ols_slope(a[i], g[i])[0] if np.ptp(a[i]) > 0 else np.nan
        boots[rk] = B
        lo, hi = np.nanpercentile(B, [2.5, 97.5])
        amin, amax = float(a.min()), float(a.max())
        rows.append(dict(range=rk, range_label=RANGES[rk], n=n, hba1c_min=amin, hba1c_max=amax, hba1c_distinct_values=int(len(np.unique(a))),
                         intercept=icpt, slope_per_point=slope, slope_per_point_ci_low=float(lo), slope_per_point_ci_high=float(hi),
                         slope_per_0_1=slope / 10, slope_per_0_1_ci_low=float(lo) / 10, slope_per_0_1_ci_high=float(hi) / 10,
                         slope_per_0_1_hc3_low=float(hc[0]) / 10, slope_per_0_1_hc3_high=float(hc[1]) / 10,
                         fitted_at_min=icpt + slope * amin, fitted_at_max=icpt + slope * amax, r2=float(fit.rsquared),
                         n_boot=N_BOOT, n_boot_nan=int(np.isnan(B).sum()), ci_method='percentile participant bootstrap within range'))
    S = pd.DataFrame(rows)
    dB = boots['5_7_to_6_4'] - boots['lt_5_7']
    dlo, dhi = np.nanpercentile(dB, [2.5, 97.5])
    st = S.set_index('range')
    diff = dict(contrast='5_7_to_6_4 minus lt_5_7', slope_per_0_1_diff=float(st.loc['5_7_to_6_4', 'slope_per_0_1'] - st.loc['lt_5_7', 'slope_per_0_1']),
                ci_low=float(dlo) / 10, ci_high=float(dhi) / 10, ratio=float(st.loc['5_7_to_6_4', 'slope_per_0_1'] / st.loc['lt_5_7', 'slope_per_0_1']))
    # ---- Part 2 shares
    sh = []
    for gname in GROUP_ORDER:
        k, n = int(cnt.loc[gname, 'ir']), int(cnt.loc[gname, 'n'])
        lo, hi = wilson(k, n)
        sh.append(dict(group=gname, group_label=GROUP_LABEL[gname], n=n, n_homa_ge_2_5=k, pct=100 * k / n, wilson_low=lo, wilson_high=hi))
    Sh = pd.DataFrame(sh)
    # ---- write
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    # only values with >=20 participants are written (small cells would expose near-individual values)
    pv_out = pv_t.assign(source_rule='>=20 participants per value, as analysis/metabolic/insulin_markers.py hba1c_table()')
    files = {'within_range_slopes.csv': S, 'homa_share_by_group.csv': Sh, 'per_value_means.csv': pv_out}
    for nme, f in files.items():
        f.to_csv(out / nme, index=False)
    summary = dict(design='descriptive cross-sectional; primary cohort; participant-level OLS within HbA1c ranges; no projection',
                   primary_n=int(len(P)), slopes={r.range: dict(n=int(r.n), slope_per_0_1=r.slope_per_0_1,
                                                                 ci=[r.slope_per_0_1_ci_low, r.slope_per_0_1_ci_high]) for r in S.itertuples()},
                   slope_difference=diff, homa_share={r.group: dict(n=int(r.n), k=int(r.n_homa_ge_2_5), pct=r.pct, ci=[r.wilson_low, r.wilson_high])
                                                      for r in Sh.itertuples()},
                   guards_passed=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    meta = dict(analysis='hba1c_within_range_slopes', version=VERSION, status='passed',
                design='descriptive; OLS of average CGM glucose on HbA1c within <5.7 and 5.7-6.4; Wilson shares', seed=SEED,
                definitions=dict(population='canonical primary cohort (no_diabetes_range == True), analysis/metabolic/insulin_markers.py load()',
                                 hba1c_rounding='decimal half-up (round_a1c)', ranges={'lt_5_7': '<5.7', '5_7_to_6_4': '5.7-6.4 (all <6.5)'},
                                 outcome='average CGM glucose (panel mean_glucose, dense days -90..+90)', slope_units='mg/dL per 0.1 HbA1c point',
                                 bootstrap=dict(replicates=N_BOOT, unit='participant, resampled within range', ci='percentile 95%'),
                                 hba1c_cutoff=A1C_CUT, mean_cgm_cutoff=CGM_CUT, homa_threshold=HOMA_THR, wilson='score interval, z=1.96'),
                guards=guards,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (out / 'hba1c_within_range_slopes.metadata.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    print(json.dumps(dict(slopes=summary['slopes'], difference=diff), indent=1, default=float))


if __name__ == '__main__':
    main()
