#!/usr/bin/env python3
"""Unadjusted insulin-marker associations of CGM and meal measures, with sensitivity tables.

Primary estimates are UNADJUSTED; the only main-text adjustment is "at the same average glucose" (covariate
mean_glucose); age and sex are reported in a sensitivity table; BMI appears only as same-participant without/with
rows; the pre-meal comparison is reported in a supplementary table. The CGM-marker and meal-response association
modules run these associations with age bucket + sex in every model, and the carbohydrate-sensitivity
and glucose-quarter analyses are age+sex adjusted; this module issues the corresponding unadjusted/partially-adjusted
versions from the same underlying functions.

Population: canonical primary cohort, canonical_lab_cgm_panel.csv no_diabetes_range == True, loaded via
analysis/metabolic/insulin_markers.py load().

Outputs (OUTPUTS/metabolic/unadjusted_insulin_markers/, aggregate rows only, no participant identifiers):
  1 cgm_measures_unadjusted.csv   family() on the CGM measures (BH-FDR family) for HOMA-IR, fasting insulin,
                                  triglycerides, TG:HDL (log, % per SD) and TyG (units per SD).
  2 meal_responses_unadjusted.csv family() on the meal measures, per-person meal frame from meal_responses(P,
                                  drop_incomplete=True) (meals missing any macronutrient dropped, never zero-filled;
                                  minimum meal count; +/-90 days of the HbA1c anchor), merged on USER_ID.
     Adjustment sets (names in the 'adjustment' column):
       none                          no covariates (PRIMARY)
       mean_glucose_only             + standardized mean_glucose ("at the same average glucose")
       none_bmi_subset / bmi_only    BMI-complete participants: without / with standardized BMI (same participants)
       mean_glucose_only_bmi_subset / mean_glucose_bmi_only   same pair at the same average glucose
  3 headline_homa_insulin.csv     HOMA-IR and fasting insulin side by side for the paper measures (none, mean_glucose_only).
  4 bmi_same_participant.csv      without/with BMI pairs, attenuation % (only where the without-BMI CI excludes 0).
  5 age_sex_sensitivity.csv       none vs age_sex, mean_glucose_only vs age+sex+mean glucose; change in points;
                                  summary max |change|.
  6 premeal_comparison.csv        meal measures (premeal_baseline itself excluded): none vs premeal_only and
                                  mean_glucose_only vs mean_glucose_premeal_only (covariate premeal_baseline).
  7 carb_sensitivity_unadjusted.csv  carbohydrate() contrast (window_mixed slope above vs at/below the median)
                                  without age/sex: geometric-mean ratio high/low (log markers) or difference
                                  (HbA1c), HC3. 'none' uses every participant with the marker; 'none_age_sample'
                                  uses the age-bucket (>=10 per bucket) sample so it pairs with 'age_sex'.
  8 homa_by_glucose_quarter_unadjusted.csv  glucose-quarter analysis (qcut of mean glucose in the HOMA-IR sample):
                                  crude share HOMA-IR >= 2.5 (Wilson CI), crude geometric mean (t CI on the log scale),
                                  unadjusted ratio vs Q1 (OLS on quarter dummies, HC3; equals the crude geometric-mean
                                  ratio), plus the BMI same-participant pair.

Inputs: canonical_lab_cgm_panel.csv (via analysis/metabolic/insulin_markers.py load()); OUTPUTS/metabolic/
insulin_markers/cgm_measures_by_marker.csv and carb_sensitivity_high_low.csv; OUTPUTS/metabolic/
level_quarters/mean_glucose_quarters.csv and mean_glucose_quarters_adjusted.csv; OUTPUTS/metabolic/
meal_rows_no_zero_fill/meal_responses_by_marker.csv and summary.json.

Internal consistency checks: CGM and meal association rows are recomputed and
compared against the saved insulin-marker outputs; meal counts are checked against the no-zero-fill summary; the
glucose-quarter crude shares/ratios are checked against the level-quarters outputs; same-participant pairs (e.g.
without/with BMI, without/with age+sex) are checked to share identical sample sizes; the unadjusted quarter ratio
is checked to equal the crude geometric-mean ratio.

Cross-sectional associations only; no projection.
Usage: python analysis/metabolic/unadjusted_insulin_markers.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402
from cgmpaper.paths import OUTPUTS  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


s_no_zero_fill = _load('no_zero_fill_for_unadjusted', 'analysis/metabolic/meal_rows_no_zero_fill.py')
s_insulin_markers = s_no_zero_fill.s_insulin_markers  # insulin_markers module as loaded by meal_rows_no_zero_fill (same source file)
s_level_quarters = s_no_zero_fill.s_level_quarters

VERSION = 'unadjusted-insulin-markers'
PUB = OUTPUTS
MP = PUB / 'metabolic'
OUT = MP / 'unadjusted_insulin_markers'
D_INSULIN_MARKERS, D_LEVEL_QUARTERS, D_NO_ZERO_FILL = MP / 'insulin_markers', MP / 'level_quarters', MP / 'meal_rows_no_zero_fill'
SEED = None  # no resampling in this producer
TOL = 1e-9
KEY = ['marker', 'adjustment', 'measure']
PAPER_CGM = ['mean_glucose', 'sd_glucose', 'cv_glucose', 'modd', 'mage', 'all_frequency', 'all_size', 'all_clearance']
PAPER_MEAL = ['iauc_2h', 'peak_rise', 'premeal_baseline']
NEW_ADJ = {'none': [], 'mean_glucose_only': ['mean_glucose']}
BMI_ADJ = {'none_bmi_subset': [], 'bmi_only': ['bmi'], 'mean_glucose_only_bmi_subset': ['mean_glucose'],
           'mean_glucose_bmi_only': ['mean_glucose', 'bmi']}
PREMEAL_ADJ = {'none': [], 'premeal_only': ['premeal_baseline'], 'mean_glucose_only': ['mean_glucose'],
               'mean_glucose_premeal_only': ['mean_glucose', 'premeal_baseline']}
CARB_MARKERS = [('homa_ir', True), ('fasting_insulin', True), ('fasting_glucose', True), ('a1c', False),
                ('triglycerides', True), ('tg_hdl_ratio', True), ('hdl_cholesterol', True), ('apo_b', True)]


def fail(msg):
    raise SystemExit(f'GUARD FAILED: {msg}')


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(100 * (c - h)), float(100 * (c + h))


def run_families(frame, measures, adj_sets, markers=None):
    rows = []
    for marker, log in (markers or s_insulin_markers.MARKERS).items():
        rows += s_insulin_markers.family(frame, measures, marker, log, adj_sets)
    return pd.DataFrame(rows)


def reproduce(new, saved, label):
    worst = s_no_zero_fill.frames_equal(new, saved, KEY, label)
    if worst >= TOL:
        fail(f'{label}: max |diff| {worst}')
    return dict(rows=int(len(saved)), max_abs_diff=float(worst))


def issue(frame, measures):
    """Unadjusted and same-average-glucose rows, plus the BMI same-participant pairs."""
    A = run_families(frame, measures, NEW_ADJ)
    B = run_families(frame.loc[frame.bmi.notna()], measures, BMI_ADJ)
    return pd.concat([A, B], ignore_index=True)


def pair(T, a, b):
    """Rows of set a (suffix _without) beside set b (suffix _with) for the same marker x measure."""
    x = T.loc[T.adjustment == a].drop(columns='adjustment'); y = T.loc[T.adjustment == b].drop(columns='adjustment')
    m = x.merge(y, on=['marker', 'measure'], suffixes=('_without', '_with'), validate='one_to_one')
    m.insert(0, 'set_with', b); m.insert(0, 'set_without', a)
    m['change_points'] = m['estimate_with'] - m['estimate_without']
    return m.reset_index(drop=True)


def same_n(m, label):
    bad = m.loc[m['n_without'] != m['n_with']]
    if len(bad):
        fail(f'{label}: n differs within a same-participant pair ({len(bad)} rows)')


def carbohydrate_unadjusted():
    W = pd.read_parquet(s_insulin_markers.CARB)
    W['high'] = (W.acs > W.acs.median()).astype(int)
    rows = []
    for marker, log in CARB_MARKERS:
        for name, adj, subset in [('none', [], None), ('none_age_sample', [], 'age'), ('mean_glucose_only', ['mean_glucose'], None),
                                  ('none_bmi_subset', [], 'bmi'), ('bmi_only', ['bmi'], 'bmi'),
                                  ('mean_glucose_only_bmi_subset', ['mean_glucose'], 'bmi'), ('mean_glucose_bmi_only', ['mean_glucose', 'bmi'], 'bmi')]:
            need = [marker] + adj + (['bmi'] if subset == 'bmi' else [])
            d = W.dropna(subset=need)
            d = d.loc[d[marker] > 0] if log else d
            if subset == 'age':  # the sample of insulin_markers carbohydrate() age_sex rows
                d = d.loc[d.age_bucket.map(d.age_bucket.value_counts()) >= 10]
            d = d.reset_index(drop=True)
            y = np.log(d[marker]) if log else d[marker]
            X = pd.DataFrame({'high': d.high.astype(float)})
            for c in adj:
                X[c] = d[c].astype(float)
            m = sm.OLS(y.to_numpy(), sm.add_constant(X)).fit(cov_type='HC3')
            pred = {g: float(np.mean(m.predict(sm.add_constant(X.assign(high=float(g)), has_constant='add')))) for g in (0, 1)}
            f = np.exp if log else (lambda v: v)
            lo, hi = m.conf_int().loc['high']
            rows.append(dict(marker=marker, adjustment=name, n=int(len(d)), n_high=int(d.high.sum()), n_low=int((1 - d.high).sum()),
                             adjusted_mean_low=float(f(pred[0])), adjusted_mean_high=float(f(pred[1])), effect=float(f(m.params['high'])),
                             ci_low=float(f(lo)), ci_high=float(f(hi)), effect_scale='ratio_high_low' if log else 'difference_high_low',
                             p=float(m.pvalues['high'])))
    return pd.DataFrame(rows)


def quarters_unadjusted(H):
    H = H.copy()
    H['quarter'] = pd.qcut(H.mean_glucose, 4, labels=[1, 2, 3, 4]).astype(int)
    rows = []
    for name, d in [('none', H), ('none_bmi_subset', H.loc[H.bmi.notna()])]:
        d = d.reset_index(drop=True)
        X = sm.add_constant(pd.DataFrame({f'q{k}': (d.quarter == k).astype(float) for k in (2, 3, 4)}))
        m = sm.OLS(np.log(d.homa_ir.to_numpy(float)), X).fit(cov_type='HC3'); ci = m.conf_int()
        for k in (1, 2, 3, 4):
            s = d.loc[d.quarter == k]; ly = np.log(s.homa_ir); se = ly.std() / np.sqrt(len(s)); t = stats.t.ppf(0.975, len(s) - 1)
            kk = int((s.homa_ir >= s_level_quarters.THRESH).sum()); wl, wh = wilson(kk, len(s))
            row = dict(adjustment=name, quarter=k, n=int(len(d)), n_quarter=int(len(s)), mean_glucose_min=float(s.mean_glucose.min()),
                       mean_glucose_max=float(s.mean_glucose.max()), n_homa_ge_2_5=kk, pct_homa_ge_2_5=float(100 * kk / len(s)),
                       pct_ci_low=wl, pct_ci_high=wh, geometric_mean_homa=float(np.exp(ly.mean())),
                       gm_ci_low=float(np.exp(ly.mean() - t * se)), gm_ci_high=float(np.exp(ly.mean() + t * se)))
            if k == 1:
                row.update(ratio_vs_q1=1.0, ratio_ci_low=np.nan, ratio_ci_high=np.nan, ratio_p=np.nan)
            else:
                row.update(ratio_vs_q1=float(np.exp(m.params[f'q{k}'])), ratio_ci_low=float(np.exp(ci.loc[f'q{k}', 0])),
                           ratio_ci_high=float(np.exp(ci.loc[f'q{k}', 1])), ratio_p=float(m.pvalues[f'q{k}']))
            rows.append(row)
    # BMI same-participant pair for the ratio (log HOMA-IR on quarters + standardized BMI, 742)
    d = H.loc[H.bmi.notna()].reset_index(drop=True)
    X = sm.add_constant(pd.DataFrame({**{f'q{k}': (d.quarter == k).astype(float) for k in (2, 3, 4)},
                                      'bmi': (d.bmi - d.bmi.mean()) / d.bmi.std()}))
    m = sm.OLS(np.log(d.homa_ir.to_numpy(float)), X).fit(cov_type='HC3'); ci = m.conf_int()
    for k in (1, 2, 3, 4):
        row = dict(adjustment='bmi_only', quarter=k, n=int(len(d)), n_quarter=int((d.quarter == k).sum()))
        if k == 1:
            row.update(ratio_vs_q1=1.0, ratio_ci_low=np.nan, ratio_ci_high=np.nan, ratio_p=np.nan)
        else:
            row.update(ratio_vs_q1=float(np.exp(m.params[f'q{k}'])), ratio_ci_low=float(np.exp(ci.loc[f'q{k}', 0])),
                       ratio_ci_high=float(np.exp(ci.loc[f'q{k}', 1])), ratio_p=float(m.pvalues[f'q{k}']))
        rows.append(row)
    return pd.DataFrame(rows), H


def guard_level_quarters(H, Q, guards):
    crude = pd.read_csv(D_LEVEL_QUARTERS / 'mean_glucose_quarters.csv'); crude = crude.loc[crude['sample'] == 'homa_all'].set_index('quarter')
    q = Q.loc[Q.adjustment == 'none'].set_index('quarter')
    ok = all(int(q.loc[k, 'n_quarter']) == int(crude.loc[k, 'n']) and int(q.loc[k, 'n_homa_ge_2_5']) == int(crude.loc[k, 'n_homa_ge_2_5'])
             and abs(q.loc[k, 'pct_homa_ge_2_5'] - crude.loc[k, 'pct_homa_ge_2_5']) < TOL
             and abs(q.loc[k, 'geometric_mean_homa'] - crude.loc[k, 'geometric_mean_homa']) < TOL for k in (1, 2, 3, 4))
    guards['G6_crude_quarters_match_level_quarters'] = dict(ok=bool(ok), pct=[float(q.loc[k, 'pct_homa_ge_2_5']) for k in (1, 2, 3, 4)],
                                                  n=[int(q.loc[k, 'n_quarter']) for k in (1, 2, 3, 4)])
    if not ok:
        fail('crude HOMA-IR quarters do not reproduce 146 mean_glucose_quarters.csv')
    adj = pd.read_csv(D_LEVEL_QUARTERS / 'mean_glucose_quarters_adjusted.csv'); adj = adj.loc[adj.adjustment == 'age_sex'].set_index('quarter')
    d = H.reset_index(drop=True)
    m = sm.OLS(np.log(d.homa_ir.to_numpy(float)), s_level_quarters.design(d)).fit(cov_type='HC3'); ci = m.conf_int()
    worst = max(max(abs(np.exp(m.params[f'q{k}']) - adj.loc[k, 'ratio_vs_q1']), abs(np.exp(ci.loc[f'q{k}', 0]) - adj.loc[k, 'ratio_ci_low']),
                    abs(np.exp(ci.loc[f'q{k}', 1]) - adj.loc[k, 'ratio_ci_high'])) for k in (2, 3, 4))
    guards['G6_age_sex_ratio_reproduced_level_quarters'] = dict(q4_vs_q1=float(np.exp(m.params['q4'])), max_abs_diff=float(worst))
    if worst >= TOL:
        fail(f'level_quarters age+sex quarter ratios not reproduced ({worst})')
    gm = Q.loc[Q.adjustment == 'none'].set_index('quarter').geometric_mean_homa
    r = Q.loc[(Q.adjustment == 'none') & (Q.quarter == 4)].ratio_vs_q1.iloc[0]
    guards['G7_unadjusted_ratio_equals_crude_gm_ratio'] = abs(r - gm[4] / gm[1]) < TOL
    if not guards['G7_unadjusted_ratio_equals_crude_gm_ratio']:
        fail('unadjusted quarter ratio differs from the crude geometric-mean ratio')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args(argv)
    t0 = time.monotonic(); guards = {}
    panel, P, g_insulin_markers = s_insulin_markers.load(args)
    guards['G2_insulin_markers_load'] = dict(g_insulin_markers, primary_n=int(len(P)))
    H = P.loc[P.homa_ir.notna() & (P.homa_ir > 0)].copy()
    guards['G2_samples'] = dict(primary_n=int(len(P)), homa_n=int(len(H)), homa_with_bmi=int(H.bmi.notna().sum()))
    # G3: 140 CGM rows, every adjustment set
    A_INSULIN_MARKERS = run_families(P, s_insulin_markers.CGM_MEASURES, s_insulin_markers.ADJ)
    guards['G3_insulin_markers_cgm_rows_reproduced'] = reproduce(A_INSULIN_MARKERS, pd.read_csv(D_INSULIN_MARKERS / 'cgm_measures_by_marker.csv'), 'insulin_markers cgm_measures_by_marker')
    # G4: 163 meal rows
    R, counts, _info, _coef, _pp = s_no_zero_fill.meal_responses(P, drop_incomplete=True)
    new_no_zero_fill = json.loads((D_NO_ZERO_FILL / 'summary.json').read_text())['new_counts']
    if counts != new_no_zero_fill:
        fail(f'meal counts {counts} vs meal_rows_no_zero_fill {new_no_zero_fill}')
    PB = P.merge(R, on='USER_ID')
    B_NO_ZERO_FILL = run_families(PB, s_insulin_markers.MEAL_MEASURES, {k: s_insulin_markers.ADJ[k] for k in ('age_sex', 'mean_glucose', 'mean_glucose_bmi')}).assign(zero_fill=False)
    guards['G4_meal_rows_no_zero_fill_meal_rows_reproduced'] = dict(reproduce(B_NO_ZERO_FILL, pd.read_csv(D_NO_ZERO_FILL / 'meal_responses_by_marker.csv'), 'meal_rows_no_zero_fill meal_responses_by_marker'),
                                                 counts=counts)
    # G5: 140 carbohydrate()
    g5 = {}
    D_insulin_markers_c, _ = s_insulin_markers.carbohydrate(g5)
    savedD = pd.read_csv(D_INSULIN_MARKERS / 'carb_sensitivity_high_low.csv')
    guards['G5_insulin_markers_carbohydrate_reproduced'] = dict(ok=s_no_zero_fill.frames_equal(D_insulin_markers_c, savedD, ['marker', 'adjustment'], 'insulin_markers carb_sensitivity_high_low') < TOL
                                                    and bool(g5['carb_homa_pearson_matches']), pearson=g5['carb_homa_pearson'])
    if not guards['G5_insulin_markers_carbohydrate_reproduced']['ok']:
        fail('insulin_markers carbohydrate() not reproduced')
    # G6/G7 quarters
    Q, HQ = quarters_unadjusted(H)
    guard_level_quarters(HQ, Q, guards)

    # ---------- new estimates
    CGM = issue(P, s_insulin_markers.CGM_MEASURES)
    MEAL = issue(PB, s_insulin_markers.MEAL_MEASURES)
    CGM.insert(0, 'family', 'cgm'); MEAL.insert(0, 'family', 'meal')
    both = pd.concat([CGM, MEAL], ignore_index=True)
    # age/sex sensitivity (insulin_markers CGM rows and 163 meal rows are the age_sex versions)
    agesex = pd.concat([A_INSULIN_MARKERS.assign(family='cgm'), B_NO_ZERO_FILL.drop(columns='zero_fill').assign(family='meal')], ignore_index=True)
    allrows = pd.concat([both, agesex.loc[agesex.adjustment.isin(['age_sex', 'mean_glucose'])]], ignore_index=True)
    S = []
    for fam, keep in [('cgm', PAPER_CGM), ('meal', s_insulin_markers.MEAL_MEASURES)]:
        T = allrows.loc[allrows.family == fam].drop(columns='family')
        for a, b, lab in [('none', 'age_sex', 'unadjusted vs age+sex'), ('mean_glucose_only', 'mean_glucose', 'mean glucose vs mean glucose+age+sex')]:
            m = pair(T, a, b); m.insert(0, 'comparison', lab); m.insert(0, 'family', fam); same_n(m, f'age/sex pair {a}'); S.append(m)
    AS = pd.concat(S, ignore_index=True)
    AS['abs_change_points'] = AS.change_points.abs()
    AS['in_paper_set'] = ((AS.family == 'cgm') & AS.measure.isin(PAPER_CGM)) | ((AS.family == 'meal') & AS.measure.isin(PAPER_MEAL))
    # BMI pairs
    BM = []
    for fam in ('cgm', 'meal'):
        T = both.loc[both.family == fam].drop(columns='family')
        for a, b, lab in [('none_bmi_subset', 'bmi_only', 'without vs with BMI'),
                          ('mean_glucose_only_bmi_subset', 'mean_glucose_bmi_only', 'mean glucose without vs with BMI')]:
            m = pair(T, a, b); m.insert(0, 'comparison', lab); m.insert(0, 'family', fam); same_n(m, f'BMI pair {a}'); BM.append(m)
    BMI = pd.concat(BM, ignore_index=True)
    # attenuation only where the without-BMI estimate is detected (95% CI excludes 0); otherwise the ratio is not interpretable
    det = (BMI.ci_low_without > 0) | (BMI.ci_high_without < 0)
    BMI['attenuation_pct'] = np.where(det, 100 * (1 - BMI.estimate_with / BMI.estimate_without), np.nan)
    # pre-meal comparison
    meal_no_pre = [m for m in s_insulin_markers.MEAL_MEASURES if m != 'premeal_baseline']
    PM = run_families(PB, meal_no_pre, PREMEAL_ADJ)
    pm = [pair(PM, 'none', 'premeal_only').assign(comparison='without vs with pre-meal glucose'),
          pair(PM, 'mean_glucose_only', 'mean_glucose_premeal_only').assign(comparison='mean glucose without vs with pre-meal glucose')]
    for m in pm:
        same_n(m, 'pre-meal pair')
    PRE = pd.concat(pm, ignore_index=True)
    PRE['in_paper_set'] = PRE.measure.isin(PAPER_MEAL)
    # guard: premeal families' none/mean_glucose_only rows equal the main issue (different FDR family only)
    chk = PM.loc[PM.adjustment.isin(['none', 'mean_glucose_only'])].merge(MEAL, on=KEY, suffixes=('_a', '_b'))
    guards['G7_premeal_rows_consistent'] = float(max((chk.estimate_a - chk.estimate_b).abs().max(), (chk.n_a - chk.n_b).abs().max())) < TOL
    if not guards['G7_premeal_rows_consistent']:
        fail('pre-meal family rows differ from the main issue')
    guards['G7_same_participant_pairs_n_equal'] = True
    # headline (HOMA-IR and fasting insulin side by side)
    hl = both.loc[both.adjustment.isin(['none', 'mean_glucose_only']) & both.marker.isin(['homa_ir', 'fasting_insulin'])
                  & (((both.family == 'cgm') & both.measure.isin(PAPER_CGM)) | ((both.family == 'meal') & both.measure.isin(PAPER_MEAL)))]
    cols = ['n', 'estimate', 'ci_low', 'ci_high', 'p', 'q_fdr_family']
    HL = hl.loc[hl.marker == 'homa_ir', ['family', 'adjustment', 'measure'] + cols].merge(
        hl.loc[hl.marker == 'fasting_insulin', ['family', 'adjustment', 'measure'] + cols], on=['family', 'adjustment', 'measure'],
        suffixes=('_homa_ir', '_fasting_insulin'), validate='one_to_one')
    order = {m: i for i, m in enumerate(PAPER_CGM + PAPER_MEAL)}
    HL = HL.sort_values(['adjustment', 'family', 'measure'], key=lambda s: s.map(order) if s.name == 'measure' else s).reset_index(drop=True)
    HL['ci_includes_zero_homa_ir'] = (HL.ci_low_homa_ir <= 0) & (HL.ci_high_homa_ir >= 0)
    HL['ci_includes_zero_fasting_insulin'] = (HL.ci_low_fasting_insulin <= 0) & (HL.ci_high_fasting_insulin >= 0)
    # carbohydrate
    C = carbohydrate_unadjusted()
    ref = savedD.loc[savedD.adjustment == 'age_sex'].set_index('marker')
    C = C.merge(ref[['n', 'effect', 'ci_low', 'ci_high']].add_prefix('age_sex_').reset_index(), on='marker', how='left')
    ca = C.loc[C.adjustment == 'none_age_sample']
    if not (ca.n == ca.age_sex_n).all():
        fail('carbohydrate none_age_sample n differs from insulin_markers age_sex n')
    guards['G7_carb_age_sample_n_equal'] = True

    out = args.out; out.mkdir(parents=True, exist_ok=True)
    files = {'cgm_measures_unadjusted.csv': CGM, 'meal_responses_unadjusted.csv': MEAL, 'headline_homa_insulin.csv': HL,
             'bmi_same_participant.csv': BMI, 'age_sex_sensitivity.csv': AS, 'premeal_comparison.csv': PRE,
             'carb_sensitivity_unadjusted.csv': C, 'homa_by_glucose_quarter_unadjusted.csv': Q}
    for nme, f in files.items():
        f.to_csv(out / nme, index=False)
    paper = AS.loc[AS.in_paper_set & AS.marker.isin(['homa_ir', 'fasting_insulin'])]
    mx = paper.loc[paper.abs_change_points.idxmax()]
    mx_all = AS.loc[AS.abs_change_points.idxmax()]
    q4 = Q.loc[(Q.adjustment == 'none') & (Q.quarter == 4)].iloc[0]
    ch = C.loc[(C.marker == 'homa_ir') & (C.adjustment == 'none')].iloc[0]
    summary = dict(design='cross-sectional associations; unadjusted primary; no projection',
                   samples=dict(guards['G2_samples'], meal_participants_ge_10=counts['participants_ge_min_meals'], meals_in_window=counts['meals_in_window'],
                                carbohydrate_population=3497),
                   age_sex_max_abs_change_points=dict(paper_measures_homa_insulin=dict(value=float(mx.abs_change_points), family=mx.family, marker=mx.marker,
                                                                                       measure=mx.measure, comparison=mx.comparison),
                                                      all_rows=dict(value=float(mx_all.abs_change_points), family=mx_all.family, marker=mx_all.marker,
                                                                    measure=mx_all.measure, comparison=mx_all.comparison)),
                   homa_q4_vs_q1_unadjusted=dict(ratio=float(q4.ratio_vs_q1), ci=[float(q4.ratio_ci_low), float(q4.ratio_ci_high)], n=int(q4.n)),
                   carb_homa_ratio_unadjusted=dict(ratio=float(ch.effect), ci=[float(ch.ci_low), float(ch.ci_high)], n=int(ch.n)),
                   guards_passed=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float) + '\n')
    meta = dict(analysis='unadjusted_insulin_markers', version=VERSION, status='passed',
                design='cross-sectional associations; OLS (HC3) of log marker (TyG untransformed) on standardized measure; no projection',
                seed=SEED, definitions=dict(population='canonical primary cohort (no_diabetes_range == True), analysis/metabolic/insulin_markers.py load()',
                                            meal_rows='analysis/metabolic/meal_rows_no_zero_fill.py meal_responses(P, drop_incomplete=True)', scale='% difference per SD (log markers); units per SD (TyG)',
                                            adjustment_sets=dict(NEW_ADJ, **BMI_ADJ, **PREMEAL_ADJ), fdr='BH within marker x adjustment set over the insulin_markers.py measure list',
                                            quarters='pd.qcut of mean glucose in the HOMA-IR sample (analysis/metabolic/level_quarters.py)', homa_threshold=float(s_level_quarters.THRESH),
                                            carbohydrate='window_mixed slope > median vs <= median (analysis/metabolic/insulin_markers.py carbohydrate())'),
                guards=guards, runtime_seconds=round(time.monotonic() - t0, 1))
    (out / 'unadjusted_insulin_markers.metadata.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    print(json.dumps(dict(summary=summary), indent=1, default=float))


if __name__ == '__main__':
    main()
