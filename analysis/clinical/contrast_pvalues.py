"""P values for the between-group contrasts quoted in the HbA1c-discordance analysis.

Reads existing producer outputs only (no database access). Where a producer reports a
bootstrap 95% CI but no P value, P is derived from the estimate and its CI (Altman &
Bland, BMJ 2011;343:d2090): SE = (upper - lower) / (2 x 1.96), z = est / SE, on the log
scale for ratios. Correlations are compared with Fisher's z (independent ranges).
Proportions are compared with a two-sided chi-squared test.

Inputs:
  OUTPUTS/metabolic/hba1c_within_range_slopes/summary.json (analysis/metabolic/hba1c_within_range_slopes.py)
  OUTPUTS/metabolic/hba1c_range_resolution/within_range_correlations.csv (analysis/metabolic/hba1c_range_resolution.py)
  OUTPUTS/clinical/markers/discordance_high_share.csv, OUTPUTS/clinical/markers/discordance_markers.csv (analysis/clinical/markers.py)
  OUTPUTS/clinical/hba1c_resolution.json

Outputs:
  OUTPUTS/clinical/contrast_pvalues/contrast_pvalues.csv

Usage: python analysis/clinical/contrast_pvalues.py
"""
import json, math
from pathlib import Path
import pandas as pd
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cgmpaper.paths import OUTPUTS  # noqa: E402

OUT = OUTPUTS
MP = OUT / 'metabolic'
dst = OUT / 'clinical' / 'contrast_pvalues'; dst.mkdir(parents=True, exist_ok=True)

def p_from_ci(est, lo, hi, log=False):
    if log: est, lo, hi = map(math.log, (est, lo, hi))
    se = (hi - lo) / (2 * 1.959964)
    return 2 * stats.norm.sf(abs(est / se))

rows = []
s = json.load(open(MP / 'hba1c_within_range_slopes/summary.json'))['slope_difference']
rows.append(('a1c_slope_diff', 'slope per 0.1 HbA1c point, 5.7-6.4% minus <5.7% (mg/dL)', s['slope_per_0_1_diff'], s['ci_low'], s['ci_high'], p_from_ci(s['slope_per_0_1_diff'], s['ci_low'], s['ci_high']), 'from bootstrap CI'))

c = pd.read_csv(MP / 'hba1c_range_resolution/within_range_correlations.csv').query("window=='symmetric_90d'").set_index('range')
r1, n1, r2, n2 = c.loc['lt_5_7', 'pearson_r'], c.loc['lt_5_7', 'n'], c.loc['5_7_to_6_4', 'pearson_r'], c.loc['5_7_to_6_4', 'n']
z = (math.atanh(r2) - math.atanh(r1)) / math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
rows.append(('a1c_r_diff', 'Pearson r, 5.7-6.4% minus <5.7%', r2 - r1, None, None, 2 * stats.norm.sf(abs(z)), "Fisher z, independent ranges"))

h = pd.read_csv(OUT / 'clinical/markers/discordance_high_share.csv').query("marker=='fasting_insulin'").set_index('group')
a, b = h.loc['CGM elevated only'], h.loc['HbA1c elevated only']
chi2, p, _, _ = stats.chi2_contingency([[a.k, a.n - a.k], [b.k, b.n - b.k]], correction=False)
rows.append(('fi10_share_cgm_vs_a1c', 'share fasting insulin >=10, CGM-only minus HbA1c-only (pct points)', a.pct - b.pct, None, None, p, 'chi-squared, 1 df'))

pr = json.load(open(OUT / 'clinical/hba1c_resolution.json'))['prevalence_ratio']
rows.append(('fi10_pr_normal_a1c', 'prevalence ratio fasting insulin >=10, avg >=110 vs <110 (normal HbA1c)', pr['rr'], *pr['ci'], p_from_ci(pr['rr'], *pr['ci'], log=True), 'from CI, log scale'))

d = pd.read_csv(OUT / 'clinical/markers/discordance_markers.csv').set_index('marker')
for m in ['fasting_insulin', 'hscrp', 'homa_ir']:
    rows.append((f'disc_{m}', f'{m} geometric mean, CGM-only vs HbA1c-only (%)', d.loc[m, 'cgm_vs_a1c_only_pct'], d.loc[m, 'lo'], d.loc[m, 'hi'], d.loc[m, 'p'], 'producer (OLS on log scale)'))

df = pd.DataFrame(rows, columns=['key', 'contrast', 'estimate', 'ci_low', 'ci_high', 'p', 'method'])
df.to_csv(dst / 'contrast_pvalues.csv', index=False)
print(df[['key', 'estimate', 'p']].to_string(index=False))
