#!/usr/bin/env python3
"""Plain (unadjusted) R^2 of insulin markers on average CGM glucose versus HbA1c, with a range-restriction check.

The HbA1c <5.7% comparison restricts the range of HbA1c by construction, which mechanically lowers HbA1c's R^2
there; a fair check stratifies on a third variable.

What.
  Part 1  unadjusted_r2.csv
    In the same participants, R^2 of log fasting insulin and of log HOMA-IR on average CGM glucose alone and on
    HbA1c alone (single-predictor OLS, i.e. squared Pearson r), whole primary cohort, HbA1c <5.7% and (for
    context) 5.7-6.4%. 95% CI: percentile, 2,000 seeded participant bootstrap resamples (same resample for both
    predictors, so the difference CI is paired). The joint model (both predictors) R^2 is given as a point value.
  Part 2  age_sex_r2_sensitivity.csv
    The age/sex version (delta R^2 over age bucket + sex model, from analysis/metabolic/cgm_vs_hba1c.py) recomputed
    with that module's own functions, side by side with the unadjusted values, with bootstrap CIs for each delta
    and for the difference.
  Part 3  discordant_groups_unadjusted.csv
    CGM elevated only (average CGM >=110 mg/dL, HbA1c <5.7%) versus HbA1c elevated only (HbA1c >=5.7%, average
    CGM <110), reference HbA1c-only; OLS of log marker on group dummies only (no covariates), HC3 95% CI, same
    participants as the age_sex rows in cgm_vs_hba1c.py. HOMA-IR and fasting insulin side by side.
  Part 4  range_restriction_strata.csv
    The same unadjusted R^2 comparison within strata defined by a variable other than HbA1c: laboratory fasting
    glucose <100 vs 100-125 mg/dL (where measured in the same draw; primary cohort has none >=126) and BMI tertiles
    (cut points from the BMI-complete primary cohort). Each stratum reports n, SD of HbA1c and SD of average glucose,
    and their ratio to the SD in the pooled comparator sample (all participants with that stratifier measured).
    The HbA1c-defined strata are listed in the same table for contrast. Note: HOMA-IR contains fasting glucose, so
    fasting-glucose strata restrict a component of that outcome; fasting insulin has no glucose term.

Population and definitions follow analysis/metabolic/insulin_markers.py, analysis/metabolic/cgm_vs_hba1c.py and
analysis/metabolic/hba1c_within_range_slopes.py: canonical primary cohort (no_diabetes_range == True); anchor HbA1c
rounded decimal half-up (round_a1c); average CGM glucose = panel mean_glucose (dense days +/-90 days of the draw);
insulin markers, fasting glucose and BMI from the panel. Cross-sectional associations only.

Structural checks (run before anything new is computed): the upstream producers' samples and reported R^2/share/
slope values are re-derived here with their own functions and compared to their output files for internal
consistency (exact counts, values within 1e-9); the cohort is confirmed to exclude HbA1c >=6.5 or fasting glucose
>=126; and the range-restriction strata are confirmed to partition their pooled comparator samples.

Inputs: database tables via LEVELS_DB (through insulin_markers.py's panel loader); OUTPUTS/metabolic/cgm_vs_hba1c_head_to_head/
(continuous_head_to_head.csv, discordant_groups_head_to_head.csv, metadata) from cgm_vs_hba1c.py; OUTPUTS/metabolic/
hba1c_within_range_slopes/ (homa_share_by_group.csv, within_range_slopes.csv, metadata) from hba1c_within_range_slopes.py;
OUTPUTS/metabolic/insulin_markers/ metadata from insulin_markers.py.
Outputs (OUTPUTS/metabolic/hba1c_unadjusted_r2/): unadjusted_r2.csv, age_sex_r2_sensitivity.csv,
discordant_groups_unadjusted.csv, range_restriction_strata.csv, summary.json, metadata. Outputs are aggregate rows
only; no participant identifiers are written.

Usage:  python analysis/metabolic/hba1c_unadjusted_r2.py
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from cgmpaper.lab_normalization import round_a1c  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


S_CGM_VS_HBA1C_PATH = 'analysis/metabolic/cgm_vs_hba1c.py'
S_WITHIN_RANGE_SLOPES_PATH = 'analysis/metabolic/hba1c_within_range_slopes.py'
s_cgm_vs_hba1c = _load('cgm_vs_hba1c_for_unadjusted_r2', S_CGM_VS_HBA1C_PATH)
s_within_range_slopes = _load('within_range_slopes_for_unadjusted_r2', S_WITHIN_RANGE_SLOPES_PATH)
VERSION = 'hba1c-unadjusted-r2'
MP = OUTPUTS / 'metabolic'
OUT = MP / 'hba1c_unadjusted_r2'
D_CGM_VS_HBA1C, D_WITHIN_RANGE_SLOPES = MP / 'cgm_vs_hba1c_head_to_head', MP / 'hba1c_within_range_slopes'
OUTCOMES = ['fasting_insulin', 'homa_ir']
SEED = 20260930
N_BOOT = 2000
TOL = 1e-9
A1C_CUT, CGM_CUT = 5.7, 110.0
FG_CUT = 100.0            # laboratory fasting glucose, mg/dL (ADA normal <100; 100-125 impaired; >=126 excluded upstream)


def fail(msg):
    raise SystemExit(f'GUARD FAILED: {msg}')


def r1(v):
    """Round half-up at one decimal for comparison with quoted values."""
    return float(np.floor(v * 10 + 0.5 + 1e-9) / 10)


def boot_r2_single(x, y, idx):
    """Squared Pearson r of y on x for each row of the index matrix idx (B x n)."""
    xs, ys = x[idx], y[idx]
    xs = xs - xs.mean(1, keepdims=True); ys = ys - ys.mean(1, keepdims=True)
    return (xs * ys).sum(1) ** 2 / ((xs ** 2).sum(1) * (ys ** 2).sum(1))


def r2_single(x, y):
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def r2_joint(y, *xs):
    X = np.column_stack([np.ones(len(y)), *xs])
    return float(s_cgm_vs_hba1c.r2(y, X))


def compare(d, rng, extra=None):
    """Unadjusted R^2 of d.y on mean_glucose vs hba1c, paired participant bootstrap."""
    x1, x2, y = d.mean_glucose.to_numpy(float), d.hba1c.to_numpy(float), d.y.to_numpy(float)
    n = len(y)
    row = dict(n=int(n), sd_hba1c=float(np.std(x2, ddof=1)), sd_mean_glucose=float(np.std(x1, ddof=1)),
               sd_log_outcome=float(np.std(y, ddof=1)), hba1c_distinct_values=int(len(np.unique(x2))),
               r2_cgm=r2_single(x1, y), r2_hba1c=r2_single(x2, y), r2_both=r2_joint(y, x1, x2),
               r_cgm_hba1c=float(np.corrcoef(x1, x2)[0, 1]))
    row['r2_diff_cgm_minus_hba1c'] = row['r2_cgm'] - row['r2_hba1c']
    B1, B2 = np.empty(N_BOOT), np.empty(N_BOOT)
    for s in range(0, N_BOOT, 250):
        idx = rng.integers(0, n, (min(250, N_BOOT - s), n))
        B1[s:s + len(idx)] = boot_r2_single(x1, y, idx); B2[s:s + len(idx)] = boot_r2_single(x2, y, idx)
    for k, B in [('r2_cgm', B1), ('r2_hba1c', B2), ('r2_diff_cgm_minus_hba1c', B1 - B2)]:
        lo, hi = np.nanpercentile(B, [2.5, 97.5]); row[f'{k}_ci_low'], row[f'{k}_ci_high'] = float(lo), float(hi)
    row['boot_share_cgm_higher'] = float(np.mean(B1 - B2 > 0)); row['n_boot'] = N_BOOT
    row['n_boot_nan'] = int(np.isnan(B2).sum())
    if extra:
        row.update(extra)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(argv); t0 = time.monotonic(); guards = {}
    # ---- load exactly as cgm_vs_hba1c.py main()
    panel, P, _ = s_cgm_vs_hba1c.s_insulin_markers.load(None)
    P['a1c_r'] = round_a1c(P.a1c.to_numpy())
    a1c_hi, cgm_hi = P.a1c_r.to_numpy() >= A1C_CUT, P.mean_glucose.to_numpy() >= CGM_CUT
    P['group'] = np.select([~a1c_hi & ~cgm_hi, ~a1c_hi & cgm_hi, a1c_hi & ~cgm_hi], ['both_normal', 'cgm_only', 'a1c_only'], 'both_elevated')
    H = s_cgm_vs_hba1c.outcome(P, 'homa_ir')
    counts = {g: [int((H.loc[H.group == g, 'homa_ir'] >= 2.5).sum()), int((H.group == g).sum())] for g in s_within_range_slopes.GROUP_ORDER}
    guards['primary_n'] = int(len(P)); guards['discordance_counts'] = counts
    if (P.a1c_r >= 6.5).any() or (P.fasting_glucose >= 126).any():
        fail('primary cohort has HbA1c >=6.5 or fasting glucose >=126')

    # ---- samples, exactly as cgm_vs_hba1c Q2 (no_bmi_full)
    base = {}
    for y in OUTCOMES:
        b = s_cgm_vs_hba1c.outcome(P, y).dropna(subset=['mean_glucose', 'a1c_r']).reset_index(drop=True)
        b['hba1c'] = b.a1c_r
        base[y] = b
    scopes = {'all_primary': lambda b: b, 'hba1c_normal_lt57': lambda b: b.loc[b.hba1c < A1C_CUT],
              'hba1c_5_7_to_6_4': lambda b: b.loc[b.hba1c >= A1C_CUT]}

    # ---- age/sex version (and Part 2 sensitivity table)
    Q_CGM_VS_HBA1C = pd.read_csv(D_CGM_VS_HBA1C / 'continuous_head_to_head.csv').set_index(['outcome', 'scope', 'adjustment'])
    rng_as = np.random.default_rng(SEED + 1)
    g3, as_rows, worst = {}, [], 0.0
    for y in OUTCOMES:
        for scope in ['all_primary', 'hba1c_normal_lt57']:
            sub = scopes[scope](base[y]).reset_index(drop=True)
            q = Q_CGM_VS_HBA1C.loc[(y, scope, 'no_bmi_full')]
            if len(sub) != int(q.n):
                fail(f'cgm_vs_hba1c n {y}/{scope}: {len(sub)} vs {int(q.n)}')
            s = sub.copy()
            for c in ('mean_glucose', 'hba1c'):
                s[c] = (s[c] - s[c].mean()) / s[c].std()
            Xs = {k: s_cgm_vs_hba1c.design(s, cols, False).to_numpy() for k, cols in [('cov', []), ('cgm', ['mean_glucose']), ('hba1c', ['hba1c'])]}
            yv = s.y.to_numpy()
            R = {k: float(s_cgm_vs_hba1c.r2(yv, X)) for k, X in Xs.items()}
            worst = max(worst, *(abs(R[k] - q[f'r2_{k}']) for k in R))
            dc, dh = 100 * (R['cgm'] - R['cov']), 100 * (R['hba1c'] - R['cov'])
            g3[f'{y}/{scope}'] = dict(n=int(len(sub)), delta_r2_cgm_pct=dc, delta_r2_hba1c_pct=dh)
            B = np.empty((N_BOOT, 3))
            for i in range(N_BOOT):
                idx = rng_as.integers(0, len(yv), len(yv))
                B[i] = [s_cgm_vs_hba1c.r2(yv[idx], Xs[k][idx]) for k in ('cov', 'cgm', 'hba1c')]
            bc, bh = B[:, 1] - B[:, 0], B[:, 2] - B[:, 0]
            un = compare(sub, np.random.default_rng(SEED + 2))  # point values only used here
            as_rows.append(dict(outcome=y, scope=scope, n=int(len(sub)), r2_age_sex_only=R['cov'],
                                delta_r2_cgm_beyond_age_sex=R['cgm'] - R['cov'], delta_r2_cgm_ci_low=float(np.percentile(bc, 2.5)),
                                delta_r2_cgm_ci_high=float(np.percentile(bc, 97.5)),
                                delta_r2_hba1c_beyond_age_sex=R['hba1c'] - R['cov'], delta_r2_hba1c_ci_low=float(np.percentile(bh, 2.5)),
                                delta_r2_hba1c_ci_high=float(np.percentile(bh, 97.5)),
                                diff_cgm_minus_hba1c=R['cgm'] - R['hba1c'], diff_ci_low=float(np.percentile(bc - bh, 2.5)),
                                diff_ci_high=float(np.percentile(bc - bh, 97.5)),
                                diff_ci_cgm_vs_hba1c_low=float(q.cgm_minus_hba1c_r2_ci_low), diff_ci_cgm_vs_hba1c_high=float(q.cgm_minus_hba1c_r2_ci_high),
                                unadjusted_r2_cgm=un['r2_cgm'], unadjusted_r2_hba1c=un['r2_hba1c'],
                                unadjusted_diff=un['r2_diff_cgm_minus_hba1c'], n_boot=N_BOOT, bootstrap_seed=SEED + 1))
    guards['age_sex_r2_max_abs_diff_vs_cgm_vs_hba1c'] = float(worst); guards['age_sex_values'] = g3
    if worst > TOL:
        fail(f'cgm_vs_hba1c R^2 rows not reproduced ({worst})')

    # ---- discordant age/sex row
    D_cgm_vs_hba1c_q = pd.read_csv(D_CGM_VS_HBA1C / 'discordant_groups_head_to_head.csv').set_index(['outcome', 'adjustment'])
    m = s_cgm_vs_hba1c.group_fit(H, 'a1c_only', False); ci = m.conf_int(); q = D_cgm_vs_hba1c_q.loc[('homa_ir', 'age_sex')]
    rec = (s_cgm_vs_hba1c.pct(m.params['cgm_only']), s_cgm_vs_hba1c.pct(ci.loc['cgm_only', 0]), s_cgm_vs_hba1c.pct(ci.loc['cgm_only', 1]), m.pvalues['cgm_only'])
    w4 = max(abs(rec[0] - q.pct_difference), abs(rec[1] - q.ci_low), abs(rec[2] - q.ci_high), abs(rec[3] - q.p))
    ok4 = int(q.n) == len(H) and w4 < TOL
    guards['discordant_age_sex_homa'] = dict(n=int(len(H)), pct=rec[0], ci=[rec[1], rec[2]], max_abs_diff=float(w4), passed=bool(ok4))
    if not ok4:
        fail(f'discordant HOMA-IR age/sex row not reproduced {guards["discordant_age_sex_homa"]}')

    # ---- Wilson shares and slopes (hba1c_within_range_slopes.py)
    Sh = pd.read_csv(D_WITHIN_RANGE_SLOPES / 'homa_share_by_group.csv').set_index('group')
    g5w, ok5 = {}, True
    for g in s_within_range_slopes.GROUP_ORDER:
        k, n = counts[g]
        lo, hi = s_within_range_slopes.wilson(k, n); p = 100 * k / n
        e = dict(k=k, n=n, pct=p, wilson=[float(lo), float(hi)])
        ok5 &= (k, n) == (int(Sh.loc[g, 'n_homa_ge_2_5']), int(Sh.loc[g, 'n'])) and max(abs(p - Sh.loc[g, 'pct']), abs(lo - Sh.loc[g, 'wilson_low']),
                                                                                        abs(hi - Sh.loc[g, 'wilson_high'])) < TOL
        g5w[g] = e
    Sl = pd.read_csv(D_WITHIN_RANGE_SLOPES / 'within_range_slopes.csv').set_index('range')
    a_all, gl_all = P.a1c_r.to_numpy(float), P.mean_glucose.to_numpy(float)
    g5s = {}
    for rk, msk in s_within_range_slopes.masks(a_all).items():
        sl, _ = s_within_range_slopes.ols_slope(a_all[msk], gl_all[msk])
        g5s[rk] = dict(n=int(msk.sum()), slope_per_0_1=sl / 10, abs_diff=abs(sl / 10 - Sl.loc[rk, 'slope_per_0_1']))
        ok5 &= int(msk.sum()) == int(Sl.loc[rk, 'n']) and g5s[rk]['abs_diff'] < TOL
    guards['wilson_shares'] = g5w; guards['within_range_slopes'] = g5s; guards['shares_slopes_passed'] = bool(ok5)
    if not ok5:
        fail(f'shares or slopes not reproduced {g5w} {g5s}')
    guards['all_passed'] = True

    # ================= Part 1 unadjusted R^2 =================
    rng = np.random.default_rng(SEED)
    rows = []
    for y in OUTCOMES:
        for scope, f in scopes.items():
            sub = f(base[y]).reset_index(drop=True)
            rows.append(compare(sub, rng, dict(outcome=y, scope=scope)))
    U = pd.DataFrame(rows)
    lead = ['outcome', 'scope']; U = U[lead + [c for c in U.columns if c not in lead]]
    for y in OUTCOMES:   # samples equal to cgm_vs_hba1c no_bmi_full
        for scope in ['all_primary', 'hba1c_normal_lt57']:
            if int(U.loc[(U.outcome == y) & (U.scope == scope), 'n'].iloc[0]) != int(Q_CGM_VS_HBA1C.loc[(y, scope, 'no_bmi_full')].n):
                fail('unadjusted sample differs from cgm_vs_hba1c')

    # ================= Part 3 discordant groups, unadjusted =================
    drows = []
    for y in OUTCOMES:
        d = s_cgm_vs_hba1c.outcome(P, y)
        G = pd.get_dummies(d.group, dtype=float).drop(columns='a1c_only')
        mu = sm.OLS(d.y.to_numpy(), sm.add_constant(G)).fit(cov_type='HC3'); ci = mu.conf_int()
        q = D_cgm_vs_hba1c_q.loc[(y, 'age_sex')]
        if int(q.n) != len(d):
            fail(f'discordant sample {y} differs from cgm_vs_hba1c')
        for g in ['cgm_only', 'both_normal', 'both_elevated']:
            drows.append(dict(outcome=y, adjustment='none', contrast=f'{g} vs a1c_only', n=int(len(d)),
                              n_group=int((d.group == g).sum()), n_a1c_only=int((d.group == 'a1c_only').sum()),
                              pct_difference=float(s_cgm_vs_hba1c.pct(mu.params[g])), ci_low=float(s_cgm_vs_hba1c.pct(ci.loc[g, 0])),
                              ci_high=float(s_cgm_vs_hba1c.pct(ci.loc[g, 1])), p=float(mu.pvalues[g]),
                              geo_mean_group=float(np.exp(d.loc[d.group == g, 'y'].mean())),
                              geo_mean_a1c_only=float(np.exp(d.loc[d.group == 'a1c_only', 'y'].mean())),
                              age_sex_pct_difference_cgm_vs_hba1c=float(q.pct_difference) if g == 'cgm_only' else np.nan,
                              ci_method='HC3'))
    Dq = pd.DataFrame(drows)

    # ================= Part 4 range-restriction strata =================
    bmi_q = np.quantile(P.bmi.dropna().to_numpy(float), [1 / 3, 2 / 3])
    strat = []
    rng_s = np.random.default_rng(SEED + 3)
    for y in OUTCOMES:
        b = base[y]
        defs = [('hba1c', 'pooled_all', b, None),
                ('hba1c', 'lt_5_7', b.loc[b.hba1c < A1C_CUT], None), ('hba1c', '5_7_to_6_4', b.loc[b.hba1c >= A1C_CUT], None)]
        fgb = b.loc[b.fasting_glucose.notna()]
        defs += [('fasting_glucose', 'pooled_measured', fgb, None),
                 ('fasting_glucose', 'lt_100', fgb.loc[fgb.fasting_glucose < FG_CUT], None),
                 ('fasting_glucose', '100_to_125', fgb.loc[fgb.fasting_glucose >= FG_CUT], None)]
        bb = b.loc[b.bmi.notna()]
        defs += [('bmi_tertile', 'pooled_measured', bb, None),
                 ('bmi_tertile', 'T1', bb.loc[bb.bmi < bmi_q[0]], f'<{bmi_q[0]:.2f}'),
                 ('bmi_tertile', 'T2', bb.loc[(bb.bmi >= bmi_q[0]) & (bb.bmi < bmi_q[1])], f'{bmi_q[0]:.2f}-<{bmi_q[1]:.2f}'),
                 ('bmi_tertile', 'T3', bb.loc[bb.bmi >= bmi_q[1]], f'>={bmi_q[1]:.2f}')]
        pooled_sd = {}
        for stratifier, stratum, sub, lab in defs:
            sub = sub.reset_index(drop=True)
            row = compare(sub, rng_s, dict(outcome=y, stratifier=stratifier, stratum=stratum, stratum_definition=lab or stratum))
            if stratum.startswith('pooled'):
                pooled_sd[stratifier] = (row['sd_hba1c'], row['sd_mean_glucose'])
            row['sd_ratio_hba1c_vs_pooled'] = row['sd_hba1c'] / pooled_sd[stratifier][0]
            row['sd_ratio_glucose_vs_pooled'] = row['sd_mean_glucose'] / pooled_sd[stratifier][1]
            strat.append(row)
    St = pd.DataFrame(strat)
    lead = ['outcome', 'stratifier', 'stratum', 'stratum_definition']; St = St[lead + [c for c in St.columns if c not in lead]]
    if not (St.groupby(['outcome', 'stratifier']).apply(lambda g: g.loc[~g.stratum.str.startswith('pooled'), 'n'].sum() == g.loc[g.stratum.str.startswith('pooled'), 'n'].iloc[0])).all():
        fail('strata do not partition the pooled samples')

    # ================= write =================
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    AS = pd.DataFrame(as_rows)
    files = {'unadjusted_r2.csv': U, 'age_sex_r2_sensitivity.csv': AS, 'discordant_groups_unadjusted.csv': Dq,
             'range_restriction_strata.csv': St}
    for n, df in files.items():
        df.to_csv(out / n, index=False)
    ns = St.loc[~St.stratum.str.startswith('pooled') & (St.stratifier != 'hba1c')]
    summary = dict(version=VERSION, design='cross-sectional associations; single-predictor OLS R^2 (squared Pearson r); same participants; paired participant bootstrap',
                   unadjusted_r2={f'{r.outcome}/{r.scope}': dict(n=int(r.n), r2_cgm=[r.r2_cgm, r.r2_cgm_ci_low, r.r2_cgm_ci_high],
                                                                   r2_hba1c=[r.r2_hba1c, r.r2_hba1c_ci_low, r.r2_hba1c_ci_high],
                                                                   diff=[r.r2_diff_cgm_minus_hba1c, r.r2_diff_cgm_minus_hba1c_ci_low, r.r2_diff_cgm_minus_hba1c_ci_high])
                                  for r in U.itertuples()},
                   discordant_unadjusted={f'{r.outcome}/{r.contrast}': [r.pct_difference, r.ci_low, r.ci_high, r.n] for r in Dq.itertuples()},
                   range_restriction=dict(bmi_tertile_cutpoints=[float(v) for v in bmi_q], fasting_glucose_cut=FG_CUT,
                                          non_hba1c_strata_with_diff_ci_above_0=int((ns.r2_diff_cgm_minus_hba1c_ci_low > 0).sum()),
                                          non_hba1c_strata_with_point_diff_above_0=int((ns.r2_diff_cgm_minus_hba1c > 0).sum()),
                                          non_hba1c_strata=int(len(ns))),
                   guards_passed=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    meta = dict(analysis='hba1c_unadjusted_r2', version=VERSION, status='passed',
                design='cross-sectional; unadjusted single-predictor R^2 of log insulin markers on average CGM glucose vs HbA1c; '
                       'age/sex sensitivity; unadjusted discordant-group contrast; range-restriction strata on fasting glucose and BMI',
                seed=SEED, seeds=dict(unadjusted=SEED, age_sex_bootstrap=SEED + 1, strata=SEED + 3),
                definitions=dict(population='canonical primary cohort (no_diabetes_range == True), insulin_markers.py load()',
                                 outcomes='log fasting insulin, log HOMA-IR (>0), samples as cgm_vs_hba1c no_bmi_full',
                                 hba1c='anchor HbA1c rounded decimal half-up (round_a1c)',
                                 average_glucose='panel mean_glucose (dense CGM days -90..+90 around the draw)',
                                 r2='single-predictor OLS R^2 = squared Pearson r; joint R^2 by least squares (point only)',
                                 bootstrap=dict(replicates=N_BOOT, unit='participant, paired across predictors, resampled within scope/stratum', ci='percentile 95%'),
                                 discordant='OLS of log marker on group dummies (ref HbA1c-only), no covariates, HC3',
                                 fasting_glucose_strata=f'<{FG_CUT:.0f} vs {FG_CUT:.0f}-125 mg/dL, panel fasting_glucose (same draw)',
                                 bmi_tertiles=dict(cutpoints=[float(v) for v in bmi_q], source='BMI-complete primary cohort, numpy linear quantiles',
                                                   n_bmi_complete=int(P.bmi.notna().sum())),
                                 hba1c_cutoff=A1C_CUT, mean_cgm_cutoff=CGM_CUT),
                guards=guards,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (out / 'hba1c_unadjusted_r2.metadata.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    print(json.dumps(dict(guards_passed=True, runtime=meta['runtime_seconds']), default=float))


if __name__ == '__main__':
    main()
