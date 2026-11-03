#!/usr/bin/env python3
"""Carbohydrate sensitivity vs insulin markers as a continuous measure and by quintile.

Complements the median-split comparison in analysis/metabolic/unadjusted_insulin_markers.py by asking the same
question without the split, using the same participants, slopes and markers (analysis/meals/meal_populations.py
window_mixed slopes within +/-90 days of the laboratory draw, loaded exactly as unadjusted_insulin_markers.py
loads them):
  continuous   OLS (HC3) of log marker on the standardized slope: % difference per SD, unadjusted;
               the same after winsorizing the slope at its 1st/99th percentiles (robustness to extreme slopes)
  rank         Spearman correlation of slope with marker
  quintiles    geometric-mean marker by slope quintile (t-based 95% CI on the log scale) and a linear trend across
               quintile index 1-5 (OLS, HC3), % difference per quintile step
Markers: HOMA-IR and fasting insulin. Associations between people, not causal effects.

Inputs: OUTPUTS/metabolic/unadjusted_insulin_markers/carb_sensitivity_unadjusted.csv and the carbohydrate frame loaded
by analysis/metabolic/unadjusted_insulin_markers.py (which reads the insulin_markers CARB parquet via LEVELS_DB-derived paths).
Outputs: OUTPUTS/metabolic/carb_sensitivity_continuous/carb_sensitivity_continuous.csv and
carb_sensitivity_quintiles.csv (participant-level slope/marker summaries; aggregate statistics only).
Usage: python analysis/metabolic/carb_sensitivity_continuous.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
VERSION = 'carb-sensitivity-continuous'
MP = OUTPUTS / 'metabolic'
OUT = MP / 'carb_sensitivity_continuous'
S_UNADJUSTED = ROOT / 'analysis/metabolic/unadjusted_insulin_markers.py'
R_UNADJUSTED = MP / 'unadjusted_insulin_markers'
MARKERS = ['homa_ir', 'fasting_insulin']
TOL = 1e-9


def load_unadjusted():
    spec = importlib.util.spec_from_file_location('unadjusted_for_carb_continuous', S_UNADJUSTED)
    mod = importlib.util.module_from_spec(spec); sys.modules['unadjusted_for_carb_continuous'] = mod; spec.loader.exec_module(mod)
    return mod


def per_sd(x, y):
    z = ((x - x.mean()) / x.std(ddof=1)).to_numpy()
    m = sm.OLS(y, sm.add_constant(z)).fit(cov_type='HC3')
    lo, hi = m.conf_int()[1]
    return dict(pct_per_sd=100 * (np.exp(m.params[1]) - 1), ci_low=100 * (np.exp(lo) - 1), ci_high=100 * (np.exp(hi) - 1), p=float(m.pvalues[1]))


def main():
    t0 = time.monotonic()
    u = load_unadjusted()
    W = pd.read_parquet(u.s_insulin_markers.CARB)
    C = u.carbohydrate_unadjusted().query("adjustment == 'none'").set_index('marker')
    saved = pd.read_csv(R_UNADJUSTED / 'carb_sensitivity_unadjusted.csv').query("adjustment == 'none'").set_index('marker')
    checks = {m: bool(abs(C.loc[m, 'effect'] - saved.loc[m, 'effect']) < TOL and C.loc[m, 'n'] == saved.loc[m, 'n'])
              for m in MARKERS}
    if not all(checks.values()):
        raise SystemExit(f'CHECK FAILED: median-split reproduction mismatch: {checks}')
    rows, qrows = [], []
    for mk in MARKERS:
        d = W.dropna(subset=[mk]).loc[lambda x: x[mk] > 0].reset_index(drop=True)
        if len(d) != int(saved.loc[mk, 'n']):
            raise SystemExit(f'CHECK FAILED: {mk} n {len(d)} != {saved.loc[mk, "n"]}')
        y = np.log(d[mk].to_numpy()); x = d.acs.astype(float)
        lo1, hi99 = x.quantile([0.01, 0.99])
        rho, prho = stats.spearmanr(x, d[mk])
        base = dict(marker=mk, n=len(d), median_split_ratio=float(saved.loc[mk, 'effect']),
                    median_split_ci_low=float(saved.loc[mk, 'ci_low']), median_split_ci_high=float(saved.loc[mk, 'ci_high']))
        rows.append(dict(base, model='per_sd', **per_sd(x, y)))
        rows.append(dict(base, model='per_sd_winsorized_1_99', **per_sd(x.clip(lo1, hi99), y)))
        rows.append(dict(base, model='spearman', rho=float(rho), p=float(prho)))
        q = pd.qcut(x, 5, labels=False) + 1
        mt = sm.OLS(y, sm.add_constant(q.astype(float).to_numpy())).fit(cov_type='HC3')
        lo, hi = mt.conf_int()[1]
        rows.append(dict(base, model='quintile_trend_per_step', pct_per_sd=100 * (np.exp(mt.params[1]) - 1),
                         ci_low=100 * (np.exp(lo) - 1), ci_high=100 * (np.exp(hi) - 1), p=float(mt.pvalues[1])))
        for k in range(1, 6):
            v = y[q.to_numpy() == k]; se = v.std(ddof=1) / np.sqrt(len(v)); tq = stats.t.ppf(0.975, len(v) - 1)
            qrows.append(dict(marker=mk, quintile=k, n=len(v), slope_min=float(x[q == k].min()), slope_max=float(x[q == k].max()),
                              geometric_mean=float(np.exp(v.mean())), ci_low=float(np.exp(v.mean() - tq * se)), ci_high=float(np.exp(v.mean() + tq * se))))
    T, Q = pd.DataFrame(rows), pd.DataFrame(qrows)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / 'carb_sensitivity_continuous.csv', index=False); Q.to_csv(OUT / 'carb_sensitivity_quintiles.csv', index=False)
    md = dict(analysis='carb_sensitivity_continuous', version=VERSION, status='passed', checks=checks,
              population='carbohydrate frame from analysis/metabolic/unadjusted_insulin_markers.py (meal_populations window_mixed slopes; 140 CARB parquet)',
              runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'carb_sensitivity_continuous.metadata.json').write_text(json.dumps(md, indent=1, default=float) + '\n')
    print(T.round(4).to_string()); print(Q.round(3).to_string())


if __name__ == '__main__':
    main()
