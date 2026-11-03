#!/usr/bin/env python3
"""Observed (non-projected) same-food averaging reliability, up to eight meals per half.

Re-runs the split-half reliability estimator from analysis/meals/split_half_reliability.py
(halves(), within_person_r, cluster_boot, seed) with K_MAX=8 on raw 2-h iAUC, for both the
interleaved (alternating-meal) and consecutive (first k vs next k in time) splits, plus a
matched-cell variant where the same cells (>=16 responses) are used at every k. Reports
observed within-person correlations only; no Spearman-Brown projection.

Inputs
------
- Meal response data via analysis/meals/_food_analysis.load_responses().
- OUTPUTS/meals/empirical_split_half_reliability.csv (analysis/meals/split_half_reliability.py).

Outputs
-------
- OUTPUTS/meals/same_food_averaging_observed/observed_averaging_curve.csv
- OUTPUTS/meals/same_food_averaging_observed/observed_averaging_matched_cells.csv
- OUTPUTS/meals/same_food_averaging_observed/summary.json
- OUTPUTS/meals/same_food_averaging_observed/same_food_averaging_observed.metadata.json

Usage
-----
python analysis/meals/same_food_averaging.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from analysis.meals import _food_analysis as shared# noqa: E402

VERSION = 'same-food-averaging-observed'
PP = OUTPUTS / 'meals'
OUT = PP / 'same_food_averaging_observed'
SAVED = PP / 'empirical_split_half_reliability.csv'
S_SPLIT_HALF = ROOT / 'analysis/meals/split_half_reliability.py'
K_MAX = 8
MATCHED_MIN = 2 * K_MAX


def main():
    t0 = time.monotonic()
    spec = importlib.util.spec_from_file_location('s_split_half', S_SPLIT_HALF)
    s_split_half = importlib.util.module_from_spec(spec); spec.loader.exec_module(s_split_half)
    frame = shared.load_responses()
    curve = s_split_half.evaluate(frame, K_MAX, s_split_half.N_BOOT, s_split_half.SEED)
    saved = pd.read_csv(SAVED)
    g = saved.merge(curve, on=['split', 'k'], suffixes=('_saved', '_new'))
    guard = dict(rows=int(len(g)), max_abs_r_diff=float((g.r_empirical_saved - g.r_empirical_new).abs().max()),
                 cells_equal=bool((g.n_cells_saved == g.n_cells_new).all()))
    curve = curve[['split', 'k', 'n_cells', 'n_users', 'r_empirical', 'ci_low', 'ci_high']]
    n_cell = frame.groupby(['uid', 'food']).iauc.transform('size')
    sub = frame.loc[n_cell.ge(MATCHED_MIN)]
    rng = np.random.default_rng(s_split_half.SEED); rows = []
    for split in ('interleaved', 'consecutive'):
        for k in range(1, K_MAX + 1):
            h = s_split_half.halves(sub, k, split)
            r, nc, nu = shared.within_person_r(h)
            lo, hi = s_split_half.cluster_boot(h, rng, s_split_half.N_BOOT)
            rows.append(dict(split=split, k=k, n_cells=nc, n_users=nu, r_empirical=r, ci_low=lo, ci_high=hi))
    matched = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    curve.to_csv(OUT / 'observed_averaging_curve.csv', index=False)
    matched.to_csv(OUT / 'observed_averaging_matched_cells.csv', index=False)
    first_070 = {s: (int(c.loc[c.r_empirical >= .70, 'k'].min()) if (c.r_empirical >= .70).any() else None)
                 for s, c in curve.groupby('split')}
    summary = dict(first_k_reaching_0_70=first_070, matched_cells_min_responses=MATCHED_MIN,
                   note='r = within-person (participant-demeaned) correlation between the means of two disjoint sets of k responses '
                        'to the same food; observed, not projected.')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    meta = dict(analysis='same_food_averaging_observed', version=VERSION, guard=guard,
                estimator='analysis/meals/split_half_reliability.py (unchanged)', k_max=K_MAX, seed=int(s_split_half.SEED), n_boot=int(s_split_half.N_BOOT),
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'same_food_averaging_observed.metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(dict(guard=guard, summary=summary), indent=1))


if __name__ == '__main__':
    main()
