#!/usr/bin/env python3
"""Tests whether a macro-adjusted meal-response SHAPE (duration/broadness) tracks insulin markers where
carbohydrate sensitivity (peak-rise height per gram) does not.

Uses the same meals as the carbohydrate-sensitivity slope (window_mixed: isolated meals with macros, within
+/-90 days of the anchor, primary cohort; >=30 meals and carb SD >=5 g per person). For each person and each
meal outcome, fits the same per-person OLS used for the slope: outcome ~ carbs + fiber + fat + protein.
Person features per outcome:
  slope      coefficient on carbohydrate (per g), holding fiber, fat and protein constant
  std_meal   predicted outcome at one fixed standard meal (cohort-median macros), i.e. the person's response to the
             same meal, adjusted for what they usually eat
Outcomes: PEAK_RISE_2H (height; slope = carbohydrate sensitivity), IAUC_2H (area),
          AUC_SHAPE_RATIO (iAUC / (peak rise x 120 min): how 'broad' the response is, independent of height),
          TIME_TO_PEAK_MIN, RETURN_10_MIN (peak to within 10 mg/dL of baseline; meals where it was reached).
Associations: log marker on standardized feature (OLS, HC3), unadjusted / + average glucose / + average glucose and
habitual carbohydrate (person mean carbs). Split-half (odd/even meals) repeatability of each feature.

Inputs:
  OUTPUTS/meals/meal_food_substrate.parquet (analysis/meals/build_meals.py)
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py)
  OUTPUTS/meal_populations/_cache/window_mixed_participants.parquet (analysis/meals/meal_populations.py)

Outputs (OUTPUTS/meals/meal_shape_macro_adjusted/; person_features.parquet is participant-level and stays private):
  meal_shape_macro_adjusted_associations.csv, person_features.parquet

Usage: python analysis/meals/meal_shape_macro_adjusted.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from analysis.meals import _carbohydrate_analysis as carb  # noqa: E402
from analysis.meals import _shape_analysis as shape  # noqa: E402
PUB = OUTPUTS
OUT = OUTPUTS / 'meals/meal_shape_macro_adjusted'; OUT.mkdir(parents=True, exist_ok=True)
MAC = carb.MACROS
OUTCOMES = ['PEAK_RISE_2H', 'IAUC_2H', 'AUC_SHAPE_RATIO', 'TIME_TO_PEAK_MIN', 'RETURN_10_MIN']

meals = carb.load_macro_meals(PUB / 'meals/meal_food_substrate.parquet')
panel = shape.load_panel(PUB / 'canonical_panel/canonical_lab_cgm_panel.csv')
W = shape.window_meals(meals, panel)
cached = pd.read_parquet(PUB / 'meal_populations/_cache/window_mixed_participants.parquet')
std = W[MAC].median().to_numpy(float)

def fit(frame, y):
    f = frame.dropna(subset=[y])
    if y == 'RETURN_10_MIN': f = f.loc[f.RETURN_10_REACHED.fillna(False)]
    if len(f) < carb.MIN_MEALS or f.CARBS_G.std() < carb.MIN_CARB_SD: return None
    x = np.column_stack([np.ones(len(f)), f[MAC].to_numpy(float)])
    b, _, rank, _ = np.linalg.lstsq(x, f[y].to_numpy(float), rcond=None)
    if rank != 5: return None
    return b[1], b[0] + b[1:] @ std

def person_features(frame_all):
    rows = []
    for u, f in frame_all.groupby('USER_ID', sort=True):
        if len(f) < carb.MIN_MEALS or f.CARBS_G.std() < carb.MIN_CARB_SD: continue
        r = {'USER_ID': u, 'n_meals': len(f), 'habitual_carbs': f.CARBS_G.mean()}
        for y in OUTCOMES:
            o = fit(f, y)
            r[f'{y}_slope'], r[f'{y}_std_meal'] = (o if o else (np.nan, np.nan))
            r[f'{y}_mean'] = f[y].mean()
        rows.append(r)
    return pd.DataFrame(rows)

F = person_features(W)
chk = F.merge(cached[['USER_ID', 'acs']], on='USER_ID', how='inner')
assert len(chk) == len(cached) and np.allclose(chk.PEAK_RISE_2H_slope, chk.acs, atol=1e-9), 'slope does not reproduce cached acs'

W = W.sort_values(['USER_ID', 'MEAL_TS', 'MEAL_EVENT_ID'], kind='stable'); W['half'] = W.groupby('USER_ID').cumcount() % 2
H = [person_features(W.loc[W.half.eq(h)]) for h in (0, 1)]
feats = [c for c in F.columns if c.endswith(('_slope', '_std_meal', '_mean'))]
rel = {c: H[0][['USER_ID', c]].merge(H[1][['USER_ID', c]], on='USER_ID').dropna().pipe(lambda d: d.iloc[:, 1].corr(d.iloc[:, 2])) for c in feats}

D = F.merge(cached[['USER_ID', 'fasting_insulin', 'homa_ir', 'mean_glucose', 'bmi']], on='USER_ID')
def assoc(marker, feat, adj):
    d = D[[marker, feat] + adj].replace([np.inf, -np.inf], np.nan).dropna(); d = d.loc[d[marker] > 0]
    lo, hi = d[feat].quantile([.01, .99]); d = d.loc[d[feat].between(lo, hi)]
    z = lambda s: (s - s.mean()) / s.std(ddof=1)
    X = sm.add_constant(pd.concat([z(d[feat])] + [z(d[a]) for a in adj], axis=1))
    m = sm.OLS(np.log(d[marker]), X).fit(cov_type='HC3'); b = m.params.iloc[1]; l, h = m.conf_int().iloc[1]
    return dict(marker=marker, feature=feat, adjustment='+'.join(adj) or 'none', n=len(d),
                pct_per_sd=100 * (np.exp(b) - 1), ci_low=100 * (np.exp(l) - 1), ci_high=100 * (np.exp(h) - 1), p=m.pvalues.iloc[1])
R = pd.DataFrame([assoc(mk, f, a) for mk in ('fasting_insulin', 'homa_ir') for f in feats
                  for a in ([], ['mean_glucose'], ['mean_glucose', 'habitual_carbs'])])
R['split_half_r'] = R.feature.map(rel)
R.to_csv(OUT / 'meal_shape_macro_adjusted_associations.csv', index=False)
F.to_parquet(OUT / 'person_features.parquet', index=False)
print(len(F), 'participants;', int(D.fasting_insulin.notna().sum()), 'with fasting insulin;', int(D.homa_ir.notna().sum()), 'with HOMA-IR')
