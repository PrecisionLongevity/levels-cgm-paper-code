#!/usr/bin/env python3
"""Computes HbA1c x CGM discordance groups in the primary cohort, and age distribution by group.

Four groups are defined by crossing average glucose (elevated if >= 110 mg/dL) with HbA1c (elevated if >= 5.7%,
HbA1c as stored at 0.1 resolution): both normal, CGM elevated only, HbA1c elevated only, both elevated.

For each group, the share of participants aged under 50 years (age decade < 50) is computed with a Wilson 95% CI,
and compared to the both-normal group via an unadjusted linear probability model (OLS of the under-50 indicator on
group, both normal as reference, no covariates, HC3 standard errors), reporting the difference in percentage points.
Age is not adjusted for; it is the variable being described. The age sample is restricted to primary-cohort
participants in the 20s-70s age decades.

Inputs:
  OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (analysis/canonical_panel/build_panel.py): USER_ID, a1c,
    mean_glucose, age_bucket, sex, no_diabetes_range.
  OUTPUTS/metabolic/level_quarters/hba1c_normal_cgm_elevated.csv (analysis/metabolic/level_quarters.py).

Outputs in OUTPUTS/clinical/discordance_groups (aggregate only):
  group_counts.csv   the four groups of the primary cohort: n and % of the primary cohort. Drawn as the 2x2 tile
                     inside Fig. 5d.
  age_under50.csv    share aged under 50 years per group, Wilson 95% CI, and the difference from the both-normal
                     group in percentage points. Drawn as the age row of Supplementary Fig. 3.
  metadata.json      status and run parameters.

Usage:  python analysis/clinical/discordance_groups.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cgmpaper.paths import OUTPUTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PUB = OUTPUTS
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
PANEL_META = PUB / 'canonical_panel/canonical_lab_cgm_panel.metadata.json'
MK_META = PUB / 'clinical/markers/metadata.json'
S_LEVEL_QUARTERS = PUB / 'metabolic/level_quarters'
OUT = PUB / 'clinical/discordance_groups'
VERSION = 'discordance-groups'
AVG_CUT, A1C_CUT, AGE_CUT = 110.0, 5.7, 50
GROUPS = [('both_normal', 'Both normal'), ('cgm_only', 'CGM elevated only'), ('a1c_only', 'HbA1c elevated only'),
          ('both_elevated', 'Both elevated')]
AGE_KEYS = {'cgm_only': 'high_avg_normal_a1c', 'a1c_only': 'normal_avg_high_a1c'}   # age_offset.json group names


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(PANEL, usecols=['USER_ID', 'a1c', 'mean_glucose', 'age_bucket', 'sex', 'no_diabetes_range'])
    S = P.loc[P.no_diabetes_range.astype(bool)].copy()
    hi_cgm, hi_a1c = S.mean_glucose >= AVG_CUT, S.a1c >= A1C_CUT - 1e-9
    S['group'] = np.select([~hi_cgm & ~hi_a1c, hi_cgm & ~hi_a1c, ~hi_cgm & hi_a1c, hi_cgm & hi_a1c],
                           [g for g, _ in GROUPS], 'none')
    cnt = S.group.value_counts()
    rows = [dict(group=g, label=lab, avg_glucose_ge_110=g in ('cgm_only', 'both_elevated'), hba1c_ge_5_7=g in ('a1c_only', 'both_elevated'),
                 n=int(cnt[g]), n_primary=len(S), pct_of_primary=100 * cnt[g] / len(S)) for g, lab in GROUPS]
    GC = pd.DataFrame(rows)
    GC.to_csv(OUT / 'group_counts.csv', index=False)

    # ---------------- age: share under 50 years, sample = 20s-70s decades (as clinical_summaries age_offset.json)
    G = S.loc[S.age_bucket.between(20, 70)].copy()
    G['u50'] = (G.age_bucket < AGE_CUT).astype(float)
    f = smf.ols('u50 ~ C(group, Treatment("both_normal"))', G).fit(cov_type='HC3')
    ci = f.conf_int()
    out = []
    for g, lab in GROUPS:
        d = G.loc[G.group == g]
        k, n = int(d.u50.sum()), len(d)
        lo, hi = wilson(k, n)
        r = dict(group=g, label=lab, n=n, k_under50=k, pct_under50=100 * k / n, wilson_lo=lo, wilson_hi=hi)
        if g != 'both_normal':
            t = f'C(group, Treatment("both_normal"))[T.{g}]'
            r.update(diff_pp_vs_both_normal=100 * f.params[t], diff_lo=100 * ci.loc[t, 0], diff_hi=100 * ci.loc[t, 1], p=float(f.pvalues[t]))
        out.append(r)
    allr = dict(group='all', label='All', n=len(G), k_under50=int(G.u50.sum()), pct_under50=100 * G.u50.mean())
    allr['wilson_lo'], allr['wilson_hi'] = wilson(allr['k_under50'], allr['n'])
    out.append(allr)
    AU = pd.DataFrame(out)
    AU.to_csv(OUT / 'age_under50.csv', index=False)
    meta = dict(analysis='discordance_groups', version=VERSION, status='passed',
                definitions=dict(avg_glucose_cut_mg_dl=AVG_CUT, hba1c_cut_pct=A1C_CUT, age_cut_years=AGE_CUT,
                                 age_variable='age_bucket (decade, years)', age_sample='primary cohort, age decades 20-70 (20-79 years)',
                                 age_model='OLS of under-50 indicator on group, both normal as reference, no covariates, HC3 95% CI',
                                 share_ci='Wilson score 95% CI'),
                script=str(Path(__file__).resolve().relative_to(ROOT)),
                runtime_seconds=round(time.time() - t0, 1))
    (OUT / 'metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(GC.to_string(index=False))
    print(AU.to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
