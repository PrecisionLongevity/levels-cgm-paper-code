"""Slope of measured HbA1c on average CGM glucose, compared with the published GMI calibration.

GMI (Bergenstal et al., Diabetes Care 2018) was derived by regressing laboratory HbA1c on mean CGM glucose:
GMI (%) = 3.31 + 0.02392 x mean glucose (mg/dL). The same regression direction is fitted here across every paired
participant (primary cohort plus those with a diabetes-range laboratory value), with a participant bootstrap 95% CI.
Also compares, at matched mean glucose, measured HbA1c against GMI and against the ADAG glucose-on-HbA1c line
(Nathan et al., Diabetes Care 2008: estimated average glucose (mg/dL) = 28.7 x HbA1c (%) - 46.7), and produces binned
summaries (mean glucose per HbA1c value, mean HbA1c per 5 mg/dL glucose bin) suppressing any bin with fewer than
MIN_CELL participants.

Inputs
- OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)
- OUTPUTS/metabolic/hba1c_within_range_slopes/within_range_slopes.csv (analysis/metabolic/hba1c_within_range_slopes.py)

Outputs (OUTPUTS/clinical/gmi_calibration/)
- gmi_calibration.json: full-panel and primary-cohort HbA1c-on-glucose slopes with bootstrap CIs
- gmi_fits.json: both regression directions, published GMI/ADAG lines, binned-cell suppression counts
- gmi_bins_by_hba1c.csv, gmi_bins_by_glucose.csv: binned means with confidence intervals
- gmi_offset.json: measured HbA1c vs. GMI at matched glucose (participant-level summaries only)

Usage: python analysis/clinical/gmi_calibration.py
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cgmpaper.paths import OUTPUTS  # noqa: E402

OUT = OUTPUTS; dst = OUT / 'clinical' / 'gmi_calibration'; dst.mkdir(parents=True, exist_ok=True)
GMI_SLOPE = 0.02392
P = pd.read_csv(OUT / 'canonical_panel/canonical_lab_cgm_panel.csv')
P = P[P.a1c.notna() & P.mean_glucose.notna()].copy()
pri = P.no_diabetes_range.astype(bool)
rng = np.random.default_rng(20261028)
def slope(d): return float(np.polyfit(d.mean_glucose.to_numpy(float), d.a1c.to_numpy(float), 1)[0])
res = {}
for nm, d in (('full_panel', P), ('primary_cohort', P[pri])):
    b = slope(d); x, y = d.mean_glucose.to_numpy(float), d.a1c.to_numpy(float); n = len(d)
    bs = [np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, n, n) for _ in range(2000))]
    res[nm] = dict(n=n, slope_pct_per_mgdl=b, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                   ratio_to_gmi=b / GMI_SLOPE, r=float(np.corrcoef(x, y)[0, 1]))
res['n_diabetes_range'] = int((~pri).sum()); res['gmi_slope'] = GMI_SLOPE
s_within_range_slopes = pd.read_csv(OUT / 'metabolic/hba1c_within_range_slopes/within_range_slopes.csv').set_index('range')
res['within_range_avg_on_a1c_per_0_1'] = {k: float(s_within_range_slopes.loc[k, 'slope_per_0_1']) for k in ('lt_5_7', '5_7_to_6_4')}
(dst / 'gmi_calibration.json').write_text(json.dumps(res, indent=1) + '\n')
print(json.dumps(res, indent=1))

# Aggregate rows for the GMI calibration supplementary figure. Binned means only, every bin with >= MIN_CELL participants
# (smaller bins are not written; their count is recorded); no individual points.
# Published lines (constants as published; GMI's are tied to the panel's own gmi column below):
#   GMI  (Bergenstal et al., Diabetes Care 2018):  GMI (%) = 3.31 + 0.02392 x mean glucose (mg/dL)   [HbA1c-on-glucose direction]
#   ADAG (Nathan et al., Diabetes Care 2008):      estimated average glucose (mg/dL) = 28.7 x HbA1c (%) - 46.7   [glucose-on-HbA1c]
import math
from scipy import stats
GMI_INTERCEPT, ADAG_SLOPE, ADAG_INTERCEPT = 3.31, 28.7, -46.7
MIN_CELL, GLU_BIN = 10, 5.0
G2 = {}
def _guard(k, ok, detail=None):
    G2[k] = dict(status='passed' if ok else 'failed', detail=detail)
    assert ok, (k, detail)
_panel = OUT / 'canonical_panel/canonical_lab_cgm_panel.csv'
_g = pd.read_csv(_panel, usecols=['USER_ID', 'gmi']).set_index('USER_ID').loc[P.USER_ID, 'gmi'].to_numpy(float)
_d = float(np.max(np.abs(GMI_INTERCEPT + GMI_SLOPE * P.mean_glucose.to_numpy(float) - _g)))
_guard('gmi_formula_reproduces_panel_gmi_column', _d < 1e-9, _d)
x_a1c, y_glu = P.a1c.to_numpy(float), P.mean_glucose.to_numpy(float)
_guard('a1c_stored_at_0_1', bool(np.all(np.abs(x_a1c * 10 - np.round(x_a1c * 10)) < 1e-9)))
# fits, both directions, all paired participants
b_ga, a_ga = np.polyfit(x_a1c, y_glu, 1)          # glucose on HbA1c: mg/dL per HbA1c point
b_ag, a_ag = np.polyfit(y_glu, x_a1c, 1)          # HbA1c on glucose: % per mg/dL
_guard('hba1c_on_glucose_slope_equals_json', abs(b_ag - res['full_panel']['slope_pct_per_mgdl']) < 1e-12, b_ag)
rng2 = np.random.default_rng(20261029); n_ = len(P)
bs_ga = []   # the HbA1c-on-glucose CI is the one already in gmi_calibration.json (same slope; one interval per estimate)
for _ in range(2000):
    i = rng2.integers(0, n_, n_)
    bs_ga.append(np.polyfit(x_a1c[i], y_glu[i], 1)[0])
r_ = float(np.corrcoef(x_a1c, y_glu)[0, 1])
_guard('slope_product_equals_r2', abs(b_ga * b_ag - r_ ** 2) < 1e-12, dict(product=b_ga * b_ag, r2=r_ ** 2))
fits = dict(n=n_, n_primary=int(pri.sum()), n_diabetes_range=int((~pri).sum()), r=r_,
            glucose_on_hba1c=dict(intercept_mg_dl=float(a_ga), slope_mg_dl_per_pct=float(b_ga),
                                  slope_ci=[float(np.percentile(bs_ga, 2.5)), float(np.percentile(bs_ga, 97.5))]),
            hba1c_on_glucose=dict(intercept_pct=float(a_ag), slope_pct_per_mg_dl=float(b_ag),
                                  slope_ci=list(res['full_panel']['ci']), slope_ci_source='gmi_calibration.json full_panel.ci',
                                  implied_mg_dl_per_pct=float(1 / b_ag)),
            gmi_published=dict(intercept_pct=GMI_INTERCEPT, slope_pct_per_mg_dl=GMI_SLOPE, implied_mg_dl_per_pct=1 / GMI_SLOPE,
                               source='Bergenstal et al., Diabetes Care 2018'),
            adag_published=dict(slope_mg_dl_per_pct=ADAG_SLOPE, intercept_mg_dl=ADAG_INTERCEPT, source='Nathan et al., Diabetes Care 2008'),
            bootstrap=dict(replicates=2000, seed=20261029, unit='participant'), min_cell=MIN_CELL)
# mean glucose per HbA1c value
tb = pd.DataFrame(dict(h=np.round(x_a1c, 1), g=y_glu)).groupby('h').g.agg(['size', 'mean', 'std']).reset_index()
keep = tb['size'] >= MIN_CELL
rows = []
for r in tb.loc[keep].itertuples():
    se = r.std / math.sqrt(r.size); t = stats.t.ppf(0.975, r.size - 1)
    rows.append(dict(hba1c=float(r.h), n=int(r.size), mean_glucose=float(r.mean), ci_lo=float(r.mean - t * se), ci_hi=float(r.mean + t * se)))
B1 = pd.DataFrame(rows)
_guard('hba1c_bins_ge_min_cell', bool((B1.n >= MIN_CELL).all()) and int(B1.n.sum() + tb.loc[~keep, 'size'].sum()) == n_)
# mean HbA1c per 5 mg/dL glucose bin
lo_edge = np.floor(y_glu / GLU_BIN) * GLU_BIN
tg = pd.DataFrame(dict(b=lo_edge, g=y_glu, h=x_a1c)).groupby('b').agg(n=('h', 'size'), mean_glucose=('g', 'mean'), mean_hba1c=('h', 'mean'),
                                                                     sd_hba1c=('h', 'std')).reset_index()
keep2 = tg.n >= MIN_CELL
rows = []
for r in tg.loc[keep2].itertuples():
    se = r.sd_hba1c / math.sqrt(r.n); t = stats.t.ppf(0.975, r.n - 1)
    rows.append(dict(glucose_bin_lo=float(r.b), glucose_bin_hi=float(r.b + GLU_BIN), n=int(r.n), mean_glucose=float(r.mean_glucose),
                     mean_hba1c=float(r.mean_hba1c), ci_lo=float(r.mean_hba1c - t * se), ci_hi=float(r.mean_hba1c + t * se)))
B2 = pd.DataFrame(rows)
_guard('glucose_bins_ge_min_cell', bool((B2.n >= MIN_CELL).all()) and int(B2.n.sum() + tg.loc[~keep2, 'n'].sum()) == n_)
fits['suppressed'] = dict(hba1c_values_lt_min_cell=int((~keep).sum()), participants_in_them=int(tb.loc[~keep, 'size'].sum()),
                          glucose_bins_lt_min_cell=int((~keep2).sum()), participants_in_them_glucose=int(tg.loc[~keep2, 'n'].sum()))
fits['drawn_ranges'] = dict(hba1c=[float(B1.hba1c.min()), float(B1.hba1c.max())], glucose_bins=[float(B2.glucose_bin_lo.min()), float(B2.glucose_bin_hi.max())])
B1.to_csv(dst / 'gmi_bins_by_hba1c.csv', index=False)
B2.to_csv(dst / 'gmi_bins_by_glucose.csv', index=False)
(dst / 'gmi_fits.json').write_text(json.dumps(fits, indent=1) + '\n')
# Measured HbA1c vs GMI at the same glucose, for Supplementary Fig. 1 in GMI's orientation.
# Participant-level differences are summarised only (share below GMI, mean and median difference); the fitted line minus GMI is
# reported at the ends of the drawn glucose range. gmi_fits.json and the bin files above are unchanged by this block.
_gmi_p = GMI_INTERCEPT + GMI_SLOPE * y_glu
_dif = x_a1c - _gmi_p
_lo, _hi = float(B2.glucose_bin_lo.min()), float(B2.glucose_bin_hi.max())
_fit_minus = {f'{g:g}': float(a_ag + b_ag * g - (GMI_INTERCEPT + GMI_SLOPE * g)) for g in (_lo, float(np.mean(y_glu)), _hi)}
_guard('fit_below_gmi_over_drawn_range', all(v < 0 for v in _fit_minus.values()), _fit_minus)
_guard('fit_minus_gmi_at_mean_equals_mean_difference', abs(_fit_minus[f'{float(np.mean(y_glu)):g}'] - float(np.mean(_dif))) < 1e-9)
off = dict(n=n_, share_hba1c_below_gmi=float(np.mean(_dif < 0)), mean_difference_pct_points=float(np.mean(_dif)),
           median_difference_pct_points=float(np.median(_dif)), mean_hba1c=float(np.mean(x_a1c)), mean_gmi=float(np.mean(_gmi_p)),
           mean_glucose=float(np.mean(y_glu)), fit_minus_gmi_at_glucose=_fit_minus, drawn_glucose_range=[_lo, _hi],
           primary_cohort=dict(n=int(pri.sum()), share_hba1c_below_gmi=float(np.mean(_dif[pri.to_numpy()] < 0)),
                               mean_difference_pct_points=float(np.mean(_dif[pri.to_numpy()]))),
           definition='difference = measured HbA1c minus GMI (3.31 + 0.02392 x the same average CGM glucose), percentage points')
(dst / 'gmi_offset.json').write_text(json.dumps(off, indent=1) + '\n')
(dst / 'metadata.json').write_text(json.dumps(dict(
    analysis='gmi_calibration', version='gmi-calibration', status='passed', guards=G2,
    script='analysis/clinical/gmi_calibration.py'), indent=1) + '\n')
print(json.dumps({k: v['status'] for k, v in G2.items()}), json.dumps(fits['suppressed']), len(B1), len(B2))
