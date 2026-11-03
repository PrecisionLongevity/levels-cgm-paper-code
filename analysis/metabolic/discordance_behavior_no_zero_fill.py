#!/usr/bin/env python3
"""Discordance-group behavior and diet measures recomputed without zero-filled macronutrients.

analysis/metabolic/discordance_behavior.py builds the per-person diet measures from isolated logged meals with
`4 * CARBS_G.fillna(0) + 4 * PROTEIN_G.fillna(0) + 9 * FAT_G.fillna(0)` as meal energy, so a meal with a missing
nutrient is treated as containing none of it (carbohydrate share of energy), and it counts meals with missing
nutrients towards the >=10-meal rule and the per-person means (carbohydrate and fiber per meal, late-meal share).
The paper rule is that missing nutrients are never set to zero: meals missing any macronutrient are DROPPED, as in
analysis/metabolic/meal_rows_no_zero_fill.py. This producer recomputes the tables of discordance_behavior.py
under that rule and writes them with a side-by-side comparison (read by figures/sfig2_discordant_groups.py).

Definitions (identical to discordance_behavior.py except the one change)
  Population   canonical primary cohort (no observed diabetes-range labs), groups by anchor HbA1c (decimal
               half-up) >=5.7% and average CGM >=110 mg/dL, loaded by discordance_behavior.build().
  Behavior     unchanged (steps, sleep, logged exercise and alcohol; BMI from the canonical panel).
  CHANGE       isolated meals within +/-90 local days of the anchor that miss any of CARBS_G, PROTEIN_G, FAT_G,
               FIBER_G are dropped before every per-person diet mean (carbohydrate per meal, carbohydrate share of
               4/4/9 energy, fiber per meal, share of meals at or after 21:00) and before the >=10-meal rule.
  Models       discordance_behavior.py analyses A (descriptives, coverage), B (age+sex adjusted differences vs both
               normal, OLS HC3) and C (log HOMA-IR sequential adjustment within fixed complete-case samples),
               reproduced unchanged.

Structural checks (run before anything new is computed)
  - the reproduced build(drop_incomplete=False) equals discordance_behavior.build() (same frame, NaN pattern, values
    exact).
  - all rows of discordance_behavior's descriptives.csv, coverage.csv, differences_vs_both_normal.csv and
    homa_ir_sequential_adjustment.csv reproduced by the reproduced analyses (keys/strings equal, numbers within 1e-9);
    its summary.json counts reproduced.
  - no-zero-fill behavior columns (bmi, steps, sleep, exercise, alcohol) identical to the zero-fill frame.

Inputs
  LEVELS_DB: panel built by analysis/metabolic/discordance_behavior.py (PANEL), exposures and meals parquet files
             used there (EXPOSURES, MEALS).
  OUTPUTS/metabolic/discordance_behavior/: descriptives.csv, coverage.csv, differences_vs_both_normal.csv,
             homa_ir_sequential_adjustment.csv, summary.json, discordance_behavior.metadata.json (from
             analysis/metabolic/discordance_behavior.py).

Outputs
  OUTPUTS/metabolic/discordance_behavior_no_zero_fill/: *_no_zero_fill.csv tables,
             differences_side_by_side.csv, homa_ir_sequential_side_by_side.csv, descriptives_side_by_side.csv,
             summary.json, discordance_behavior_no_zero_fill.metadata.json.
Cross-sectional associations only. No participant identifiers are written.

Usage
  python analysis/metabolic/discordance_behavior_no_zero_fill.py [--out PATH]
"""
from __future__ import annotations

import argparse
import io
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


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


S_DISCORDANCE_BEHAVIOR_PATH = 'analysis/metabolic/discordance_behavior.py'
s_discordance_behavior = _load('discordance_behavior_for_discordance_no_zero_fill', S_DISCORDANCE_BEHAVIOR_PATH)

VERSION = 'discordance-behavior-no-zero-fill'
PUB = OUTPUTS
D_DISCORDANCE_BEHAVIOR = PUB / 'metabolic/discordance_behavior'
OUT = PUB / 'metabolic/discordance_behavior_no_zero_fill'
META_DISCORDANCE_BEHAVIOR = D_DISCORDANCE_BEHAVIOR / 'discordance_behavior.metadata.json'
MACROS = ['CARBS_G', 'PROTEIN_G', 'FAT_G', 'FIBER_G']
DIET = ['carbs_meal', 'carb_share', 'fiber_meal', 'late_share']
BEHAV = ['bmi', 'steps', 'sleep_h', 'sleep_mid', 'ex_share', 'alc_share']
TOL = 1e-9
TABLES = {'descriptives.csv': ['variable', 'group'], 'coverage.csv': ['variable', 'group'],
          'differences_vs_both_normal.csv': ['variable', 'group'], 'homa_ir_sequential_adjustment.csv': ['sample', 'model', 'group']}


def fail(msg):
    raise SystemExit(f'GUARD FAILED: {msg}')


def build(drop_incomplete):
    """Copy of s_discordance_behavior.build(); drop_incomplete=True drops meals missing any macronutrient instead of fillna(0)."""
    P = pd.read_csv(s_discordance_behavior.PANEL, low_memory=False)
    P = P[P.no_diabetes_range == True].copy()  # noqa: E712
    P['anchor_date'] = pd.to_datetime(P.anchor_date)
    P['a1c_hi'] = s_discordance_behavior.round_a1c(P.a1c.to_numpy()) >= 5.7; P['cgm_hi'] = P.mean_glucose >= 110
    P['group'] = np.select([~P.a1c_hi & ~P.cgm_hi, ~P.a1c_hi & P.cgm_hi, P.a1c_hi & ~P.cgm_hi], ['both_normal', 'cgm_only', 'a1c_only'], 'both_elevated')
    P['male'] = (P.sex == 'male').astype(float)
    E = pd.read_parquet(s_discordance_behavior.EXPOSURES, columns=['USER_ID', 'DAY', 'TOTAL_STEPS', 'SLEEP_DURATION_MIN', 'SLEEP_MIDPOINT_HR', 'SLEEP_LOG_COUNT', 'EX_ANY', 'ALC_ANY'])
    E = E[E.USER_ID.isin(P.USER_ID)].merge(P[['USER_ID', 'anchor_date']], on='USER_ID')
    E = E[(pd.to_datetime(E.DAY) - E.anchor_date).dt.days.abs() <= s_discordance_behavior.WINDOW_DAYS]
    st = E[E.TOTAL_STEPS > 0].groupby('USER_ID').TOTAL_STEPS.agg(['mean', 'size'])
    sl = E[(E.SLEEP_LOG_COUNT > 0) & (E.SLEEP_DURATION_MIN > 0)].groupby('USER_ID').agg(
        sleep_h=('SLEEP_DURATION_MIN', lambda v: v.mean() / 60), sleep_mid=('SLEEP_MIDPOINT_HR', 'mean'), n=('SLEEP_DURATION_MIN', 'size'))
    B = pd.DataFrame({'steps': st.loc[st['size'] >= s_discordance_behavior.MIN_DAYS, 'mean']}).join(sl.loc[sl.n >= s_discordance_behavior.MIN_DAYS, ['sleep_h', 'sleep_mid']], how='outer')
    dd = E.groupby('USER_ID').agg(ex_share=('EX_ANY', 'mean'), alc_share=('ALC_ANY', 'mean'), cgm_days=('DAY', 'size'))
    B = B.join(dd, how='outer')
    M = pd.read_parquet(s_discordance_behavior.MEALS, columns=['USER_ID', 'MEAL_TS', 'tz_offset', 'MEAL_HOUR', 'CARBS_G', 'PROTEIN_G', 'FAT_G', 'FIBER_G'])
    M = M[M.USER_ID.isin(P.USER_ID)].merge(P[['USER_ID', 'anchor_date']], on='USER_ID')
    M['local_date'] = (pd.to_datetime(M.MEAL_TS, utc=True).dt.tz_localize(None) + pd.to_timedelta(M.tz_offset.fillna(0), unit='h')).dt.normalize()
    M = M[(M.local_date - M.anchor_date).dt.days.abs() <= s_discordance_behavior.WINDOW_DAYS].copy()
    info = dict(meals_in_window=int(len(M)), meals_missing_any_macro=int(M[MACROS].isna().any(axis=1).sum()),
                **{f'meals_missing_{c}': int(M[c].isna().sum()) for c in MACROS})
    if drop_incomplete:
        M = M.loc[M[MACROS].notna().all(axis=1)].copy()
        en = 4 * M.CARBS_G + 4 * M.PROTEIN_G + 9 * M.FAT_G
        M['carb_share'] = np.where(en > 0, 4 * M.CARBS_G / en, np.nan)
    else:
        en = 4 * M.CARBS_G.fillna(0) + 4 * M.PROTEIN_G.fillna(0) + 9 * M.FAT_G.fillna(0)
        M['carb_share'] = np.where(en > 0, 4 * M.CARBS_G.fillna(0) / en, np.nan)
    M['late'] = M.MEAL_HOUR >= 21
    g = M.groupby('USER_ID')
    D = pd.DataFrame({'n_meals': g.size(), 'carbs_meal': g.CARBS_G.mean(), 'carb_share': g.carb_share.mean(), 'fiber_meal': g.FIBER_G.mean(),
                      'late_share': g.late.mean()})
    per_person_meals = D.n_meals.copy()
    D = D[D.n_meals >= s_discordance_behavior.MIN_MEALS].drop(columns='n_meals')
    A = P[['USER_ID', 'group', 'age_bucket', 'male', 'bmi', 'homa_ir', 'mean_glucose', 'a1c']].merge(B, left_on='USER_ID', right_index=True, how='left') \
         .merge(D, left_on='USER_ID', right_index=True, how='left')
    for gname in s_discordance_behavior.GROUPS[1:]:
        A['g_' + gname] = (A.group == gname).astype(float)
    info['meals_used'] = int(len(M))
    return A, dict(behavior_days_in_window=int(len(E)), meals_in_window=int(len(M))), info, per_person_meals


def analyses(A):
    """Analyses A, B and C of discordance_behavior.py main(), reproduced unchanged."""
    GROUPS, VARS, PRIMARY_N, design = s_discordance_behavior.GROUPS, s_discordance_behavior.VARS, s_discordance_behavior.PRIMARY_N, s_discordance_behavior.design
    GC = ['g_' + g for g in GROUPS[1:]]
    H = A[A.homa_ir > 0].copy(); H['lh'] = np.log(H.homa_ir)
    desc, cov = [], []
    for v, lab in VARS.items():
        cov.append(dict(variable=v, label=lab, group='all', n=int(A[v].notna().sum()), n_denominator=int(len(A)),
                        pct_of_denominator=float(100 * A[v].notna().mean()), pct_of_primary_cohort=float(100 * A[v].notna().sum() / PRIMARY_N)))
        for g in GROUPS:
            s = A.loc[A.group == g, v]
            cov.append(dict(variable=v, label=lab, group=g, n=int(s.notna().sum()), n_denominator=int(len(s)),
                            pct_of_denominator=float(100 * s.notna().mean()), pct_of_primary_cohort=float(100 * s.notna().sum() / PRIMARY_N)))
            desc.append(dict(variable=v, label=lab, group=g, n=int(s.notna().sum()), median=float(s.median()), q25=float(s.quantile(.25)),
                             q75=float(s.quantile(.75)), mean=float(s.mean())))
    for g in ['all', *GROUPS]:
        s = A if g == 'all' else A.loc[A.group == g]
        cov.append(dict(variable='homa_ir', label='HOMA-IR', group=g, n=int((s.homa_ir > 0).sum()), n_denominator=int(len(s)),
                        pct_of_denominator=float(100 * (s.homa_ir > 0).mean()), pct_of_primary_cohort=float(100 * (s.homa_ir > 0).sum() / PRIMARY_N)))
    desc, cov = pd.DataFrame(desc), pd.DataFrame(cov)
    diffs = []
    for v in VARS:
        s = A.dropna(subset=[v, 'age_bucket', 'male'])
        f = sm.OLS(s[v].astype(float), design(s, ['male'] + GC)).fit(cov_type='HC3')
        for gc in GC:
            lo, hi = f.conf_int().loc[gc]
            diffs.append(dict(variable=v, label=VARS[v], group=gc[2:], n=int(len(s)), n_group=int(s[gc].sum()), diff=float(f.params[gc]),
                              ci_low=float(lo), ci_high=float(hi), p=float(f.pvalues[gc])))
    diffs = pd.DataFrame(diffs)
    att = []
    for scope, need in s_discordance_behavior.SAMPLES:
        Hc = H.dropna(subset=need + ['age_bucket', 'male'])
        for nm, cv in s_discordance_behavior.SETS.items():
            if not set(cv) <= set(need):
                continue
            f = sm.OLS(Hc.lh, design(Hc, ['male'] + GC + cv)).fit(cov_type='HC3')
            for gc in GC:
                lo, hi = f.conf_int().loc[gc]
                att.append(dict(sample=scope, n=int(len(Hc)), model=nm, group=gc[2:], pct=float(100 * np.expm1(f.params[gc])),
                                ci_low=float(100 * np.expm1(lo)), ci_high=float(100 * np.expm1(hi)), p=float(f.pvalues[gc]), n_group=int(Hc[gc].sum())))
    att = pd.DataFrame(att)
    return {'descriptives.csv': desc, 'coverage.csv': cov, 'differences_vs_both_normal.csv': diffs, 'homa_ir_sequential_adjustment.csv': att}, H


def frames_equal(new, saved, keys, label):
    """Max |diff| over numeric columns after aligning on keys; string columns must be equal."""
    if len(new) != len(saved) or list(new.columns) != list(saved.columns):
        fail(f'{label}: shape/columns {new.shape} vs saved {saved.shape}')
    m = new.merge(saved, on=keys, suffixes=('_new', '_saved'), validate='one_to_one')
    if len(m) != len(saved):
        fail(f'{label}: keys do not align')
    worst = 0.0
    for c in [c for c in saved.columns if c not in keys]:
        a, b = m[c + '_new'], m[c + '_saved']
        if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
            a, b = a.astype(float), b.astype(float)
            if not (a.isna() == b.isna()).all():
                fail(f'{label}: missing pattern differs in {c}')
            worst = max(worst, float(np.nanmax(np.abs(a - b))) if a.notna().any() else 0.0)
        elif not (a.astype(str) == b.astype(str)).all():
            fail(f'{label}: column {c} differs')
    return worst


def side_by_side(old, new, keys, est, n='n'):
    m = old.merge(new, on=keys, suffixes=('_old', '_new'), validate='one_to_one')
    m['delta_estimate'] = m[f'{est}_new'] - m[f'{est}_old']
    m['n_change'] = m[f'{n}_new'] - m[f'{n}_old']
    m['sign_change'] = np.sign(m[f'{est}_old']) != np.sign(m[f'{est}_new'])
    m['ci_excludes_zero_old'] = (m.ci_low_old > 0) | (m.ci_high_old < 0)
    m['ci_excludes_zero_new'] = (m.ci_low_new > 0) | (m.ci_high_new < 0)
    m['ci_status_change'] = m.ci_excludes_zero_old != m.ci_excludes_zero_new
    m['null_to_null_sign_flip'] = m.sign_change & ~m.ci_excludes_zero_old & ~m.ci_excludes_zero_new
    m['substantive_change'] = m.ci_status_change | (m.sign_change & ~m.null_to_null_sign_flip)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(argv); t0 = time.monotonic(); guards = {}
    out = args.out
    m_discordance_behavior = json.loads(META_DISCORDANCE_BEHAVIOR.read_text())
    # structural check -------------------------------------------------------------------------
    A_DISCORDANCE_BEHAVIOR, wc_discordance_behavior = s_discordance_behavior.build()
    A_old, wc_old, info_old, ppm_old = build(drop_incomplete=False)
    try:
        pd.testing.assert_frame_equal(A_old.reset_index(drop=True), A_DISCORDANCE_BEHAVIOR.reset_index(drop=True), check_exact=True)
    except AssertionError as e:
        fail(f'reproduced build(drop_incomplete=False) differs from discordance_behavior.build(): {e}')
    guards['build_equal_discordance_behavior'] = bool(wc_old == wc_discordance_behavior)
    if not guards['build_equal_discordance_behavior']:
        fail(f'window counts differ {wc_old} vs {wc_discordance_behavior}')
    T_old, H_old = analyses(A_old)
    guards['s_discordance_behavior_rows_checked'] = {}
    worst = 0.0
    for name, keys in TABLES.items():
        saved = pd.read_csv(D_DISCORDANCE_BEHAVIOR / name)
        new = pd.read_csv(io.StringIO(T_old[name].to_csv(index=False)))  # same CSV round trip as the saved file
        worst = max(worst, frames_equal(new, saved, keys, f'discordance_behavior {name}'))
        guards['s_discordance_behavior_rows_checked'][name] = int(len(saved))
    guards['s_discordance_behavior_max_abs_diff'] = worst
    if worst > TOL:
        fail(f'discordance_behavior tables not reproduced (max abs diff {worst})')
    summ_discordance_behavior = json.loads((D_DISCORDANCE_BEHAVIOR / 'summary.json').read_text())
    att = T_old['homa_ir_sequential_adjustment.csv']
    t = att[(att['sample'] == 'bmi_steps_sleep') & (att.group == 'cgm_only')].set_index('model').pct
    hc = H_old.groupby('group').agg(n=('lh', 'size'), ir=('homa_ir', lambda v: int((v >= 2.5).sum())))
    obs = {g: [int(hc.loc[g, 'n']), int(hc.loc[g, 'ir'])] for g in s_discordance_behavior.GROUPS}
    guards['s_discordance_behavior_summary_guards_reproduced'] = bool(
        obs == summ_discordance_behavior['guards']['homa_counts_positives'] and len(A_old) == summ_discordance_behavior['primary_n']
        and {g: int((A_old.group == g).sum()) for g in s_discordance_behavior.GROUPS} == summ_discordance_behavior['group_sizes']
        and abs(t['age_sex'] - summ_discordance_behavior['guards']['cgm_only_pct']['age_sex']) < TOL
        and abs(t['+bmi_steps_sleep'] - summ_discordance_behavior['guards']['cgm_only_pct']['bmi_steps_sleep']) < TOL
        and wc_old == summ_discordance_behavior['window_counts'])
    if not guards['s_discordance_behavior_summary_guards_reproduced']:
        fail('discordance_behavior summary.json guards not reproduced')
    # ---------------- new: drop meals missing any macronutrient ----------------
    A_new, wc_new, info_new, ppm_new = build(drop_incomplete=True)
    if not (A_new.USER_ID.to_numpy() == A_old.USER_ID.to_numpy()).all():
        fail('participant order differs between the zero-fill and no-zero-fill frames')
    guards['behavior_columns_identical'] = bool(all(A_new[c].equals(A_old[c]) for c in BEHAV + ['group', 'homa_ir']))
    if not guards['behavior_columns_identical']:
        fail('non-diet columns changed when dropping meals')
    T_new, H_new = analyses(A_new)
    out.mkdir(parents=True, exist_ok=True)
    names = {}
    for name, fr in T_new.items():
        nn = name.replace('.csv', '_no_zero_fill.csv'); fr.to_csv(out / nn, index=False); names[name] = nn
    # side by side
    sbs_d = side_by_side(T_old['differences_vs_both_normal.csv'], T_new['differences_vs_both_normal.csv'], ['variable', 'label', 'group'], 'diff')
    sbs_h = side_by_side(T_old['homa_ir_sequential_adjustment.csv'], T_new['homa_ir_sequential_adjustment.csv'], ['sample', 'model', 'group'], 'pct')
    sbs_d.to_csv(out / 'differences_side_by_side.csv', index=False)
    sbs_h.to_csv(out / 'homa_ir_sequential_side_by_side.csv', index=False)
    dsc = T_old['descriptives.csv'].merge(T_new['descriptives.csv'], on=['variable', 'label', 'group'], suffixes=('_old', '_new'))
    dsc.to_csv(out / 'descriptives_side_by_side.csv', index=False)
    # participants: coverage change of the diet measures (aggregate counts only)
    pp = pd.DataFrame({'n_old': ppm_old}).join(pd.DataFrame({'n_new': ppm_new}), how='left').fillna({'n_new': 0})
    lost = dict(participants_with_any_window_meal=int(len(pp)),
                participants_ge10_old=int((pp.n_old >= s_discordance_behavior.MIN_MEALS).sum()), participants_ge10_new=int((pp.n_new >= s_discordance_behavior.MIN_MEALS).sum()),
                participants_lost_below10=int(((pp.n_old >= s_discordance_behavior.MIN_MEALS) & (pp.n_new < s_discordance_behavior.MIN_MEALS)).sum()),
                participants_with_any_dropped_meal=int((pp.n_new < pp.n_old).sum()))
    diet_rows = sbs_d.loc[sbs_d.variable.isin(DIET)]
    h_diet = sbs_h.loc[sbs_h['sample'] == 'all']
    h_other = sbs_h.loc[sbs_h['sample'] != 'all']
    if (h_other.delta_estimate.abs() > TOL).any() or (sbs_d.loc[~sbs_d.variable.isin(DIET), 'delta_estimate'].abs() > TOL).any():
        fail('rows that do not use diet measures changed')
    cc = lambda fr: fr.loc[fr.substantive_change, [c for c in fr.columns if c in ('variable', 'sample', 'model', 'group', 'delta_estimate', 'ci_excludes_zero_old', 'ci_excludes_zero_new', 'sign_change')]].to_dict(orient='records')  # noqa: E731
    cs = T_new['homa_ir_sequential_adjustment.csv']; cs = cs.loc[cs.group == 'cgm_only'].set_index(['sample', 'model'])
    summary = dict(design='cross-sectional associations; replacement of zero-filled macronutrient diet measures of analysis/metabolic/discordance_behavior.py',
                   version=VERSION, primary_n=int(len(A_new)), meal_counts=dict(zero_fill=info_old, no_zero_fill=info_new),
                   window_counts=dict(zero_fill=wc_old, no_zero_fill=wc_new), participants=lost,
                   complete_case_samples={s: int(cs.loc[s].n.iloc[0]) for s, _ in s_discordance_behavior.SAMPLES},
                   cgm_only_homa_pct={f'{s}|{m}': dict(pct=float(r.pct), ci_low=float(r.ci_low), ci_high=float(r.ci_high)) for (s, m), r in cs.iterrows()},
                   diet_differences_max_abs_delta={v: float(diet_rows.loc[diet_rows.variable == v, 'delta_estimate'].abs().max()) for v in DIET},
                   homa_all_sample_max_abs_delta_pct=float(h_diet.delta_estimate.abs().max()),
                   conclusion_changes=dict(differences=cc(diet_rows), homa_sequential=cc(h_diet),
                                           n_differences_substantive=int(diet_rows.substantive_change.sum()),
                                           n_homa_sequential_substantive=int(h_diet.substantive_change.sum())),
                   guards=guards)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    meta = dict(analysis='discordance_behavior_no_zero_fill', version=VERSION,
                design='cross-sectional associations; OLS HC3; complete-case; no imputation; no zero-filled nutrients',
                definitions=dict(**m_discordance_behavior['definitions'], macronutrients=MACROS,
                                 missing_rule='meal dropped if any macronutrient missing (never set to zero), before every per-person diet mean and the >=10-meal rule'),
                replaces={k: v for k, v in names.items()}, guards=guards,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (out / 'discordance_behavior_no_zero_fill.metadata.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    print(json.dumps(dict(guards=guards, participants=lost, meal_counts=summary['meal_counts'],
                          conclusion_changes=summary['conclusion_changes']), indent=1, default=float))


if __name__ == '__main__':
    main()
