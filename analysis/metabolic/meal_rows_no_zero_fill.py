#!/usr/bin/env python3
"""Logged-meal response rows recomputed without zero-filling missing macronutrients.

The insulin-marker meal_responses() analysis (analysis/metabolic/insulin_markers.py) residualizes
the 2-h peak rise on carbohydrate, protein, fat and fiber with
`M[[CARBS_G, PROTEIN_G, FAT_G, FIBER_G]].fillna(0)`: a meal with a missing nutrient is treated as
containing none of it. The alternative rule applied here is that missing nutrients are never set to
zero. analysis/metabolic/hba1c_stratified_spikes.py, level_quarters.py and
within_range_associations.py call meal_responses() and so inherit the zero-fill rule; observability_inputs.py
and downstream figure/table builders read meal_responses_by_marker.csv. This producer recomputes the rows
under the alternative rule, the rule used in the paper, and writes them with a side-by-side comparison;
analysis/metabolic/unadjusted_insulin_markers.py reads them.

Definitions (identical to insulin_markers.py except the one change)
  Population   canonical primary cohort (canonical_lab_cgm_panel.csv, no_diabetes_range == True),
               loaded with insulin_markers.load().
  Meals        meal_shape_substrate.parquet, meals within +/-90 local days of the HbA1c anchor.
  CHANGE       meals missing any of CARBS_G, PROTEIN_G, FAT_G, FIBER_G are DROPPED before the pooled
               OLS of PEAK_RISE_2H on the four nutrients (rise_beyond_content = residual) and before
               every per-person mean (premeal baseline, peak rise, iAUC, share rising >=30, time to
               peak, return fraction and minutes). The >=10 meals per person rule is applied after
               dropping. FPC1-3 means are read from meal_fpca_per_user.csv as in insulin_markers.py
               (not meal rows here); their rows change only through the participant set.
  Models       insulin_markers.family(): OLS of log marker (TyG untransformed) on the standardized measure,
               age bucket, sex (+ mean glucose, + BMI complete case), HC3; % difference per SD;
               BH-FDR within marker x adjustment across the 11 measures (q_fdr_family).
  Within range within_range_associations.py main-loop logic reproduced unchanged (that module is imported):
               strata of anchor HbA1c rounded decimal half-up (round_a1c via
               publication_panels.load_panel), <5.7 and 5.7-6.4; hba1c_stratified_spikes.fit(); BH across
               the measures; range interaction.
  Level corr.  level_quarters.spearman_rows() on the primary cohort and HOMA-IR sample.

Structural checks (run before the new computation; fail closed)
  - the reproduced meal_responses(drop_incomplete=False) equals insulin_markers.meal_responses() frame and
    counts exactly, and reproduces all rows of meal_responses_by_marker.csv.
  - within_range_associations.load()/guard_insulin_markers()/guard_stratified_spikes() pass, and every row of
    within_range_associations.csv and range_interactions.csv is reproduced by the reproduced loop.
  - every row of level_quarters level_correlations.csv is reproduced with spearman_rows().
  - observability_inputs.py observability_points.csv same_food_k1/k7 HOMA-IR cells equal the
    peak_rise rows they read.
Cross-sectional associations only; no projection.

Inputs
  Database (via LEVELS_DB): read indirectly through insulin_markers.load() and
  within_range_associations.load().
  OUTPUTS/metabolic/insulin_markers/ (insulin_markers.py), OUTPUTS/metabolic/level_quarters/
  (level_quarters.py), OUTPUTS/metabolic/observability_inputs/ (observability_inputs.py),
  OUTPUTS/metabolic/within_range_associations/ (within_range_associations.py).

Outputs
  OUTPUTS/metabolic/meal_rows_no_zero_fill/: meal_responses_by_marker.csv,
  meal_responses_zero_fill_and_dropped.csv, meal_responses_side_by_side.csv, pooled_macro_model.csv,
  within_range_associations_no_zero_fill.csv, range_interactions_no_zero_fill.csv,
  within_range_side_by_side.csv, range_interactions_side_by_side.csv, level_correlations_side_by_side.csv,
  participants_lost.csv (participant-level, kept private), summary.json,
  meal_rows_no_zero_fill.metadata.json.

Usage
  python analysis/metabolic/meal_rows_no_zero_fill.py [--out OUTPUT_DIR]
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
from statsmodels.stats.multitest import multipletests  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


s_within_range = _load('within_range_for_no_zero_fill', 'analysis/metabolic/within_range_associations.py')
s_stratified_spikes = s_within_range.s_stratified_spikes
s_insulin_markers = s_within_range.s_insulin_markers
s_level_quarters = _load('level_quarters_for_no_zero_fill', 'analysis/metabolic/level_quarters.py')

VERSION = 'meal-rows-no-zero-fill'
PUB = OUTPUTS
MP = PUB / 'metabolic'
OUT = MP / 'meal_rows_no_zero_fill'
D_INSULIN_MARKERS, D_LEVEL_QUARTERS, D_OBSERVABILITY, D_WITHIN_RANGE = MP / 'insulin_markers', MP / 'level_quarters', MP / 'observability_inputs', MP / 'within_range_associations'
MACROS = ['CARBS_G', 'PROTEIN_G', 'FAT_G', 'FIBER_G']
TOL = 1e-9
NUM = ['n', 'estimate', 'ci_low', 'ci_high', 'p', 'q_fdr_family']
KEY = ['marker', 'adjustment', 'measure']
CLAIM_COLS = ['marker', 'adjustment', 'measure', 'n', 'estimate', 'ci_low', 'ci_high', 'p', 'scale', 'q_fdr_family']
# downstream cells that read a 140 meal row (builder source files named in the side-by-side table)


def fail(msg):
    raise SystemExit(f'GUARD FAILED: {msg}')


def meal_responses(P, drop_incomplete):
    """Copy of s_insulin_markers.meal_responses(); drop_incomplete=True drops meals missing any macronutrient instead of fillna(0)."""
    cols = ['USER_ID', 'MEAL_TS', 'tz_offset', 'CARBS_G', 'PROTEIN_G', 'FAT_G', 'FIBER_G', 'BASELINE_PRE', 'IAUC_2H',
            'PEAK_RISE_2H', 'TIME_TO_PEAK_MIN', 'RETURN_10_MIN', 'RETURN_10_REACHED']
    M = pd.read_parquet(s_insulin_markers.MEALS, columns=cols).merge(P[['USER_ID', 'anchor_date']].assign(anchor_date=lambda d: pd.to_datetime(d.anchor_date)),
                                                        on='USER_ID')
    local = pd.to_datetime(M.MEAL_TS, utc=True).dt.tz_localize(None) + pd.to_timedelta(M.tz_offset.fillna(0), unit='h')
    M = M.loc[(local.dt.normalize() - M.anchor_date).dt.days.abs() <= s_insulin_markers.WINDOW_DAYS].copy()
    info = dict(meals_in_window=int(len(M)), meals_missing_any_macro=int(M[MACROS].isna().any(axis=1).sum()),
                **{f'meals_missing_{c}': int(M[c].isna().sum()) for c in MACROS})
    if drop_incomplete:
        M = M.loc[M[MACROS].notna().all(axis=1)].copy()
        X = sm.add_constant(M[MACROS])
    else:
        X = sm.add_constant(M[MACROS].fillna(0))
    model = sm.OLS(M.PEAK_RISE_2H, X).fit()
    M['rise_resid'] = M.PEAK_RISE_2H - model.predict(X)
    M['ret_min'] = np.where(M.RETURN_10_REACHED == True, M.RETURN_10_MIN, np.nan)  # noqa: E712
    M['rise30'] = (M.PEAK_RISE_2H >= 30).astype(float)
    g = M.groupby('USER_ID')
    R = pd.DataFrame({'n_meals': g.size(), 'premeal_baseline': g.BASELINE_PRE.mean(), 'peak_rise': g.PEAK_RISE_2H.mean(),
                      'iauc_2h': g.IAUC_2H.mean(), 'rise_beyond_content': g.rise_resid.mean(), 'time_to_peak': g.TIME_TO_PEAK_MIN.mean(),
                      'return_fraction': g.RETURN_10_REACHED.mean(), 'return_minutes': g.ret_min.mean(),
                      'share_rise_ge30': g.rise30.mean()}).reset_index()
    per_person_all = R[['USER_ID', 'n_meals']].copy()
    R = R.loc[R.n_meals >= s_insulin_markers.MIN_MEALS]
    fp = pd.read_csv(s_insulin_markers.FPCA, usecols=['USER_ID', 'fpc1_mean', 'fpc2_mean', 'fpc3_mean'])
    coef = pd.DataFrame({'term': model.params.index, 'coef': model.params.to_numpy(), 'se': model.bse.to_numpy()})
    counts = dict(meals_in_window=int(len(M)), participants_ge_min_meals=int(len(R)))
    return R.merge(fp, on='USER_ID', how='left'), counts, info, coef.assign(n_meals=int(len(M)), r2=float(model.rsquared)), per_person_all


def rows_insulin_markers(P, R):
    PB = P.merge(R, on='USER_ID')
    rows = []
    for marker, log in s_insulin_markers.MARKERS.items():
        rows += s_insulin_markers.family(PB, s_insulin_markers.MEAL_MEASURES, marker, log, {k: s_insulin_markers.ADJ[k] for k in ('age_sex', 'mean_glucose', 'mean_glucose_bmi')})
    return pd.DataFrame(rows)


def within_range(P):
    """main() estimation loop of within_range_associations.py, reproduced unchanged (T and I tables)."""
    MEASURES, COMPANION, OUTCOMES, STRATA = s_within_range.MEASURES, s_within_range.COMPANION, s_within_range.OUTCOMES, s_within_range.STRATA
    rows = []
    for fam, specs in [('main', MEASURES), ('companion', COMPANION)]:
        for x, label, adj, row in specs:
            for y, (scale, log) in OUTCOMES.items():
                for st in STRATA:
                    d = s_within_range.outcome_frame(P.loc[P.stratum == st], y, log)
                    r = s_within_range.fit_raw(d, x, adj, y)
                    if r is None:
                        fail(f'no fit {x}/{y}/{st}')
                    sd_x = float(d.dropna(subset=[x, y] + [c for c in adj if c not in ('age', x)])[x].std())
                    rows.append(dict(family=fam, measure_key=row, measure=x, label=label, outcome=y, stratum=st, stratum_label=s_within_range.STRATUM_LABEL[st],
                                     adjustment='+'.join(['age_bucket', 'sex'] + [c for c in adj if c not in ('age', 'male')]), n=r['n'],
                                     measure_sd_in_stratum=sd_x, coef_model_scale=r['estimate'], se_model_scale=r['se'], scale=scale,
                                     estimate=s_within_range.to_pct(r['estimate']) if log else r['estimate'], ci_low=s_within_range.to_pct(r['ci_low']) if log else r['ci_low'],
                                     ci_high=s_within_range.to_pct(r['ci_high']) if log else r['ci_high'], p=r['p']))
    T = pd.DataFrame(rows)
    T['q_fdr_column'] = np.nan
    for y in OUTCOMES:
        m = (T.family == 'main') & (T.outcome == y)
        T.loc[m, 'q_fdr_column'] = multipletests(T.loc[m, 'p'], method='fdr_bh')[1]
    T['ci_excludes_null'] = (T.ci_low > 0) | (T.ci_high < 0)
    T['exploratory'] = True
    irows = []
    for fam, specs in [('main', MEASURES), ('companion', COMPANION)]:
        for x, label, adj, row in specs:
            for y, (scale, log) in OUTCOMES.items():
                it = s_within_range.interaction(P, x, adj, y, log)
                a = T.loc[(T.measure == x) & (T.outcome == y) & (T.stratum == 'lt57')].iloc[0]
                b = T.loc[(T.measure == x) & (T.outcome == y) & (T.stratum == '57_64')].iloc[0]
                diff = b.coef_model_scale - a.coef_model_scale; se = float(np.hypot(a.se_model_scale, b.se_model_scale))
                f = s_within_range.to_pct if log else float
                irows.append(dict(family=fam, measure_key=row, measure=x, label=label, outcome=y, **it,
                                  interaction_scale=('ratio of per-pooled-SD effects, 57_64 vs lt57, minus 1 (%)' if log
                                                     else 'difference in HbA1c points per pooled SD, 57_64 minus lt57'),
                                  stratified_slope_difference=f(diff), stratified_difference_ci_low=f(diff - s_within_range.Z975 * se),
                                  stratified_difference_ci_high=f(diff + s_within_range.Z975 * se), p_stratified_difference=float(2 * s_within_range.stats.norm.sf(abs(diff) / se))))
    I = pd.DataFrame(irows)
    I['q_fdr_interaction'] = np.nan
    for y in OUTCOMES:
        m = (I.family == 'main') & (I.outcome == y)
        I.loc[m, 'q_fdr_interaction'] = multipletests(I.loc[m, 'p_interaction'], method='fdr_bh')[1]
    return T, I


def frames_equal(new, saved, keys, label):
    """Max |diff| over numeric columns after aligning on keys; string columns must be equal."""
    if len(new) != len(saved):
        fail(f'{label}: {len(new)} rows vs saved {len(saved)}')
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


def sig_flags(est, lo, hi, q):
    return dict(ci_excludes_zero=bool((lo > 0) | (hi < 0)), q_lt_05=(None if q is None or pd.isna(q) else bool(q < 0.05)))


def side_by_side(old, new, keys, qcol, pcol='p'):
    m = old.merge(new, on=keys, suffixes=('_old', '_new'), validate='one_to_one')
    m['delta_estimate'] = m.estimate_new - m.estimate_old
    m['n_change'] = m.n_new - m.n_old
    m['sign_change'] = np.sign(m.estimate_old) != np.sign(m.estimate_new)
    m['ci_excludes_zero_old'] = (m.ci_low_old > 0) | (m.ci_high_old < 0)
    m['ci_excludes_zero_new'] = (m.ci_low_new > 0) | (m.ci_high_new < 0)
    m['ci_status_change'] = m.ci_excludes_zero_old != m.ci_excludes_zero_new
    if qcol:
        m['q_lt_05_old'] = m[f'{qcol}_old'] < 0.05
        m['q_lt_05_new'] = m[f'{qcol}_new'] < 0.05
        has_q = m[f'{qcol}_old'].notna()
        m['q_status_change'] = has_q & (m.q_lt_05_old != m.q_lt_05_new)
    else:
        m['q_status_change'] = False
    m['conclusion_change'] = m.sign_change | m.ci_status_change | m.q_status_change
    # a sign flip between two estimates whose CIs both include zero is a null-to-null flip, not a changed conclusion
    m['null_to_null_sign_flip'] = m.sign_change & ~m.ci_excludes_zero_old & ~m.ci_excludes_zero_new
    m['substantive_change'] = m.ci_status_change | m.q_status_change | (m.sign_change & ~m.null_to_null_sign_flip)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(argv); t0 = time.monotonic(); guards = {}
    # the reproduced meal_responses() matches insulin_markers.py exactly
    panel, P, g_insulin_markers = s_insulin_markers.load(None)
    guards['s_insulin_markers_load'] = g_insulin_markers; guards['primary_n'] = int(len(P))
    R_INSULIN_MARKERS, c_insulin_markers = s_insulin_markers.meal_responses(P)
    R_old, c_old, info, coef_old, pp_old = meal_responses(P, drop_incomplete=False)
    pd.testing.assert_frame_equal(R_old.reset_index(drop=True), R_INSULIN_MARKERS.reset_index(drop=True), check_exact=True)
    guards['meal_responses_equal_insulin_markers'] = bool(c_old == c_insulin_markers)
    if not guards['meal_responses_equal_insulin_markers']:
        fail(f'meal_responses counts {c_old} vs insulin_markers.py {c_insulin_markers}')
    s_insulin_markers_c = json.loads((D_INSULIN_MARKERS / 'summary.json').read_text())['samples']
    guards['meal_counts'] = dict(observed=c_old, expected_insulin_markers_summary={k: s_insulin_markers_c[k] for k in c_old})
    if any(c_old[k] != s_insulin_markers_c[k] for k in c_old):
        fail(f'meal counts {c_old} vs insulin_markers summary')
    # all meal rows reproduced
    B_old = rows_insulin_markers(P, R_old)
    B_saved = pd.read_csv(D_INSULIN_MARKERS / 'meal_responses_by_marker.csv')
    if not (B_old.n.astype(int).to_numpy() == B_saved.n.astype(int).to_numpy()).all():
        fail('insulin_markers meal rows: n differs')
    guards['s_insulin_markers_meal_rows_checked'] = int(len(B_saved))
    guards['s_insulin_markers_meal_rows_max_abs_diff'] = frames_equal(B_old, B_saved, KEY, 'insulin_markers meal_responses_by_marker')
    if guards['s_insulin_markers_meal_rows_max_abs_diff'] > TOL:
        fail('insulin_markers meal rows not reproduced')
    # within_range_associations.py rows reproduced
    g_within_range = {}
    P_WITHIN_RANGE = s_within_range.load(g_within_range); s_within_range.guard_insulin_markers(P_WITHIN_RANGE, g_within_range); s_within_range.guard_stratified_spikes(P_WITHIN_RANGE, g_within_range)
    guards['s_within_range_load_and_guards'] = {k: v for k, v in g_within_range.items() if k in ('primary_n', 'meal_counts_match_insulin_markers', 's_insulin_markers_max_abs_diff',
                                                                           's_stratified_spikes_M1_max_abs_diff', 'carb_homa_pearson_diff')}
    T_old, I_old = within_range(P_WITHIN_RANGE)
    T_saved, I_saved = pd.read_csv(D_WITHIN_RANGE / 'within_range_associations.csv'), pd.read_csv(D_WITHIN_RANGE / 'range_interactions.csv')
    tk, ik = ['family', 'measure', 'outcome', 'stratum'], ['family', 'measure', 'outcome']
    guards['s_within_range_rows_checked'] = int(len(T_saved) + len(I_saved))
    guards['s_within_range_max_abs_diff'] = max(frames_equal(T_old, T_saved, tk, 'within_range_associations within_range_associations'),
                                      frames_equal(I_old, I_saved, ik, 'within_range_associations range_interactions'))
    if guards['s_within_range_max_abs_diff'] > TOL:
        fail(f"within_range_associations not reproduced ({guards['s_within_range_max_abs_diff']})")
    # level_quarters.py level correlations reproduced
    H = P.loc[P.homa_ir.notna() & (P.homa_ir > 0)]

    def corr(R):
        PB = P.merge(R, on='USER_ID', how='left'); HB = PB.loc[PB.USER_ID.isin(H.USER_ID)]
        fam = s_level_quarters.SPIKE + s_level_quarters.MEAL + s_level_quarters.OTHER
        return pd.DataFrame(s_level_quarters.spearman_rows(PB, 'primary', fam) + s_level_quarters.spearman_rows(HB, 'homa_sample', fam))
    L_old, L_saved = corr(R_old), pd.read_csv(D_LEVEL_QUARTERS / 'level_correlations.csv')
    guards['s_level_quarters_rows_checked'] = int(len(L_saved))
    guards['s_level_quarters_max_abs_diff'] = frames_equal(L_old, L_saved, ['sample', 'measure'], 'level_quarters level_correlations')
    if guards['s_level_quarters_max_abs_diff'] > TOL:
        fail('level_quarters level correlations not reproduced')
    # observability_inputs.py same-food cells read the insulin_markers.py peak_rise rows
    Mp = pd.read_csv(D_OBSERVABILITY / 'observability_points.csv').set_index('signal')
    pr = B_saved.set_index(KEY)
    d_observability = 0.0
    for sig in ('same_food_k1', 'same_food_k7'):
        a, b = pr.loc[('homa_ir', 'mean_glucose_bmi', 'peak_rise')], pr.loc[('homa_ir', 'mean_glucose', 'peak_rise')]
        d_observability = max(d_observability, abs(Mp.loc[sig, 'homa_estimate'] - a.estimate), abs(Mp.loc[sig, 'homa_ci_low'] - a.ci_low),
                   abs(Mp.loc[sig, 'homa_ci_high'] - a.ci_high), abs(Mp.loc[sig, 'homa_no_bmi_estimate'] - b.estimate))
    guards['same_food_cells_max_abs_diff'] = float(d_observability)
    if d_observability > TOL:
        fail('observability_inputs same-food cells do not equal the insulin_markers peak_rise rows')

    # ---------------- new: drop meals missing any macronutrient ----------------
    R_new, c_new, _, coef_new, pp_new = meal_responses(P, drop_incomplete=True)
    B_new = rows_insulin_markers(P, R_new)
    lost = sorted(set(R_old.USER_ID) - set(R_new.USER_ID))
    gained = sorted(set(R_new.USER_ID) - set(R_old.USER_ID))
    kept = set(R_old.USER_ID) & set(R_new.USER_ID)
    changed = pp_old.merge(pp_new, on='USER_ID', how='left', suffixes=('_old', '_new')).fillna({'n_meals_new': 0})
    Hs, Fs = set(H.USER_ID), set(P.loc[P.fasting_insulin > 0, 'USER_ID'])
    drops = dict(**info, meals_used_old=c_old['meals_in_window'], meals_used_new=c_new['meals_in_window'],
                 meals_dropped=c_old['meals_in_window'] - c_new['meals_in_window'],
                 pct_meals_dropped=float(100 * (c_old['meals_in_window'] - c_new['meals_in_window']) / c_old['meals_in_window']),
                 participants_with_meals=int(len(pp_old)), participants_with_any_meal_dropped=int((changed.n_meals_new < changed.n_meals_old).sum()),
                 participants_with_all_meals_dropped=int((changed.n_meals_new == 0).sum()),
                 participants_ge_min_meals_old=c_old['participants_ge_min_meals'], participants_ge_min_meals_new=c_new['participants_ge_min_meals'],
                 participants_lost=len(lost), participants_gained=len(gained),
                 participants_kept_with_meals_dropped=int(changed.loc[changed.USER_ID.isin(kept)].pipe(lambda d: (d.n_meals_new < d.n_meals_old).sum())),
                 homa_sample_lost=int(len(set(lost) & Hs)), homa_sample_with_bmi_lost=int(len(set(lost) & set(H.loc[H.bmi.notna(), 'USER_ID']))),
                 fasting_insulin_sample_lost=int(len(set(lost) & Fs)))
    if gained:
        fail('dropping meals cannot add participants')
    S = side_by_side(B_old, B_new, KEY, 'q_fdr_family')
    S['scale'] = S.scale_old
    # decomposition: the zero-fill rule of insulin_markers.py restricted to the participants retained under the drop rule
    B_sp = rows_insulin_markers(P, R_old.loc[R_old.USER_ID.isin(R_new.USER_ID)])
    S = S.merge(B_sp[KEY + ['n', 'estimate', 'ci_low', 'ci_high', 'q_fdr_family']].rename(
        columns={c: f'{c}_zero_fill_same_participants' for c in ['n', 'estimate', 'ci_low', 'ci_high', 'q_fdr_family']}), on=KEY, validate='one_to_one')
    S = S[KEY + ['scale', 'n_old', 'n_new', 'n_change', 'estimate_old', 'ci_low_old', 'ci_high_old', 'p_old', 'q_fdr_family_old',
                 'estimate_new', 'ci_low_new', 'ci_high_new', 'p_new', 'q_fdr_family_new', 'delta_estimate', 'sign_change',
                 'ci_excludes_zero_old', 'ci_excludes_zero_new', 'ci_status_change', 'q_lt_05_old', 'q_lt_05_new', 'q_status_change',
                 'conclusion_change', 'null_to_null_sign_flip', 'substantive_change', 'n_zero_fill_same_participants',
                 'estimate_zero_fill_same_participants', 'ci_low_zero_fill_same_participants', 'ci_high_zero_fill_same_participants',
                 'q_fdr_family_zero_fill_same_participants']]
    Rep = B_new[CLAIM_COLS].assign(zero_fill=False)
    Both = pd.concat([B_old[CLAIM_COLS].assign(zero_fill=True), Rep], ignore_index=True)
    coef = coef_old.merge(coef_new, on='term', suffixes=('_zero_fill', '_dropped'))
    # within_range_associations within range (rows for peak_rise and rise_beyond_content change; q across the column recomputed)
    Rn_within_range = meal_responses(P_WITHIN_RANGE.drop(columns=['peak_rise', 'rise_beyond_content']), drop_incomplete=True)[0]
    P_within_range_n = P_WITHIN_RANGE.drop(columns=['peak_rise', 'rise_beyond_content']).merge(Rn_within_range[['USER_ID', 'peak_rise', 'rise_beyond_content']], on='USER_ID', how='left')
    T_new, I_new = within_range(P_within_range_n)
    ST = side_by_side(T_old, T_new, tk + ['label', 'scale', 'measure_key'], 'q_fdr_column')
    ST['changed_input'] = ST.measure.isin(['peak_rise', 'rise_beyond_content'])
    ST = ST[tk + ['measure_key', 'label', 'scale', 'changed_input', 'n_old', 'n_new', 'n_change', 'estimate_old', 'ci_low_old', 'ci_high_old',
                  'p_old', 'q_fdr_column_old', 'estimate_new', 'ci_low_new', 'ci_high_new', 'p_new', 'q_fdr_column_new', 'delta_estimate',
                  'sign_change', 'ci_excludes_zero_old', 'ci_excludes_zero_new', 'ci_status_change', 'q_lt_05_old', 'q_lt_05_new',
                  'q_status_change', 'conclusion_change', 'null_to_null_sign_flip', 'substantive_change']]
    SI = I_old.merge(I_new, on=ik + ['label', 'measure_key', 'interaction_scale'], suffixes=('_old', '_new'), validate='one_to_one')
    SI['changed_input'] = SI.measure.isin(['peak_rise', 'rise_beyond_content'])
    SI['q_int_lt_05_old'], SI['q_int_lt_05_new'] = SI.q_fdr_interaction_old < .05, SI.q_fdr_interaction_new < .05
    SI['p_int_lt_05_old'], SI['p_int_lt_05_new'] = SI.p_interaction_old < .05, SI.p_interaction_new < .05
    SI['conclusion_change'] = (np.sign(SI.interaction_old) != np.sign(SI.interaction_new)) | \
        (SI.q_int_lt_05_old != SI.q_int_lt_05_new) & SI.q_fdr_interaction_old.notna() | (SI.p_int_lt_05_old != SI.p_int_lt_05_new)
    SI['substantive_change'] = ((SI.q_int_lt_05_old != SI.q_int_lt_05_new) & SI.q_fdr_interaction_old.notna()) | (SI.p_int_lt_05_old != SI.p_int_lt_05_new)
    SI = SI[ik + ['label', 'changed_input', 'n_pooled_old', 'n_pooled_new', 'interaction_old', 'interaction_ci_low_old', 'interaction_ci_high_old',
                  'p_interaction_old', 'q_fdr_interaction_old', 'interaction_new', 'interaction_ci_low_new', 'interaction_ci_high_new',
                  'p_interaction_new', 'q_fdr_interaction_new', 'q_int_lt_05_old', 'q_int_lt_05_new', 'p_int_lt_05_old', 'p_int_lt_05_new',
                  'conclusion_change', 'substantive_change', 'interaction_scale']]
    # level_quarters level correlations
    L_new = corr(R_new)
    SL = L_old.merge(L_new, on=['sample', 'measure', 'family'], suffixes=('_old', '_new'), validate='one_to_one')
    SL['changed_input'] = SL.family == 'meal'
    SL['ci_excludes_zero_old'] = (SL.ci_low_old > 0) | (SL.ci_high_old < 0)
    SL['ci_excludes_zero_new'] = (SL.ci_low_new > 0) | (SL.ci_high_new < 0)
    SL['conclusion_change'] = (np.sign(SL.spearman_rho_old) != np.sign(SL.spearman_rho_new)) | (SL.ci_excludes_zero_old != SL.ci_excludes_zero_new)
    SL['substantive_change'] = SL.ci_excludes_zero_old != SL.ci_excludes_zero_new
    SL = SL.loc[SL.changed_input]
    # same-food cells (observability_inputs.py) and derived values
    bn = B_new.set_index(KEY)
    d_observability_new = {sig: dict(homa_estimate=float(bn.loc[('homa_ir', 'mean_glucose_bmi', 'peak_rise')].estimate),
                          homa_ci_low=float(bn.loc[('homa_ir', 'mean_glucose_bmi', 'peak_rise')].ci_low),
                          homa_ci_high=float(bn.loc[('homa_ir', 'mean_glucose_bmi', 'peak_rise')].ci_high),
                          homa_n=int(bn.loc[('homa_ir', 'mean_glucose_bmi', 'peak_rise')].n),
                          homa_no_bmi_estimate=float(bn.loc[('homa_ir', 'mean_glucose', 'peak_rise')].estimate),
                          homa_no_bmi_n=int(bn.loc[('homa_ir', 'mean_glucose', 'peak_rise')].n)) for sig in ('same_food_k1', 'same_food_k7')}
    d_observability_old = {sig: {k: (int(Mp.loc[sig, k]) if k.endswith('_n') else float(Mp.loc[sig, k])) for k in d_observability_new[sig]} for sig in d_observability_new}

    def derived(B):
        fi = B.loc[(B.marker == 'fasting_insulin') & (B.adjustment == 'mean_glucose_bmi')]
        return dict(meal_response_n=dict(mean_glucose=int(B.loc[(B.marker == 'homa_ir') & (B.adjustment == 'mean_glucose')].n.max()),
                                         mean_glucose_bmi=int(B.loc[(B.marker == 'homa_ir') & (B.adjustment == 'mean_glucose_bmi')].n.max())),
                    fasting_insulin_meal_bmi_max_abs=float(fi.estimate.abs().max()))
    der_old, der_new = derived(B_old), derived(B_new)
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    files = {'meal_responses_by_marker.csv': Rep, 'meal_responses_zero_fill_and_dropped.csv': Both, 'meal_responses_side_by_side.csv': S,
             'pooled_macro_model.csv': coef, 'within_range_associations_no_zero_fill.csv': T_new, 'range_interactions_no_zero_fill.csv': I_new,
             'within_range_side_by_side.csv': ST, 'range_interactions_side_by_side.csv': SI, 'level_correlations_side_by_side.csv': SL,
             'participants_lost.csv': pd.DataFrame({'USER_ID': lost})}
    for n, f in files.items():
        f.to_csv(out / n, index=False)
    chg = lambda D: D.loc[D.substantive_change]  # noqa: E731
    summary = dict(
        rule='meals missing any of carbohydrate, protein, fat or fiber dropped before the pooled macronutrient model and before per-person means; >=10 meals after dropping',
        drops=drops, new_counts=c_new, old_counts=c_old, derived_values=dict(old=der_old, new=der_new), same_food_cells=dict(old=d_observability_old, new=d_observability_new),
        conclusion_changes=dict(
            definition='substantive = 95% CI excludes-zero status changes, or q<0.05 status changes, or sign changes with at least one CI excluding zero; '
                       'null-to-null sign flips counted separately',
            s_insulin_markers_rows_total=int(len(S)), s_insulin_markers_any_flag=int(S.conclusion_change.sum()), s_insulin_markers_substantive=int(S.substantive_change.sum()),
            s_insulin_markers_null_to_null_sign_flips=int(S.null_to_null_sign_flip.sum()),
            s_insulin_markers_changed=chg(S)[KEY + ['estimate_old', 'ci_low_old', 'ci_high_old', 'q_fdr_family_old', 'estimate_new', 'ci_low_new', 'ci_high_new',
                                       'q_fdr_family_new', 'ci_status_change', 'q_status_change']].to_dict(orient='records'),
            s_within_range_rows_total=int(len(ST)), s_within_range_any_flag=int(ST.conclusion_change.sum()), s_within_range_substantive=int(ST.substantive_change.sum()),
            s_within_range_changed=chg(ST)[tk + ['estimate_old', 'estimate_new', 'sign_change', 'ci_status_change', 'q_status_change']].to_dict(orient='records'),
            s_within_range_interactions_any_flag=int(SI.conclusion_change.sum()), s_within_range_interactions_substantive=int(SI.substantive_change.sum()),
            s_level_quarters_rows_total=int(len(SL)), s_level_quarters_any_flag=int(SL.conclusion_change.sum()), s_level_quarters_substantive=int(SL.substantive_change.sum()),
            s_level_quarters_changed=chg(SL)[['sample', 'measure', 'spearman_rho_old', 'ci_low_old', 'ci_high_old', 'spearman_rho_new', 'ci_low_new', 'ci_high_new']].to_dict(orient='records')),
        max_abs_delta=dict(s_insulin_markers_estimate=float(S.delta_estimate.abs().max()), s_within_range_estimate=float(ST.delta_estimate.abs().max()),
                           s_level_quarters_rho=float((SL.spearman_rho_new - SL.spearman_rho_old).abs().max())),
        guards=guards)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    meta = dict(analysis='meal_rows_no_zero_fill', version=VERSION, status='passed',
                design='cross-sectional associations; no projection; replacement of zero-filled macronutrient meal rows',
                definitions=dict(population='canonical primary cohort (no_diabetes_range == True)', meal_window_days=s_insulin_markers.WINDOW_DAYS,
                                 min_meals=s_insulin_markers.MIN_MEALS, macronutrients=MACROS,
                                 missing_rule='meal dropped if any macronutrient missing (never set to zero), before pooled model and per-person means',
                                 hba1c_rounding='decimal half-up (round_a1c) before thresholding (within_range_associations strata via load_panel)',
                                 models='family() of insulin_markers.py: % per SD, HC3, BH within marker x adjustment; within_range_associations.py loop; level_quarters.py spearman_rows()',
                                 zero_fill_flag='True = zero-fill rule of insulin_markers.py (fillna(0)); False = this module'),
                guards=guards, runtime_seconds=round(time.monotonic() - t0, 1))
    (out / 'meal_rows_no_zero_fill.metadata.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    print(json.dumps(dict(guards=guards, drops=drops, conclusion_changes={k: v for k, v in summary['conclusion_changes'].items() if not k.endswith('changed')},
                          max_abs_delta=summary['max_abs_delta']), indent=1, default=float))


if __name__ == '__main__':
    main()
