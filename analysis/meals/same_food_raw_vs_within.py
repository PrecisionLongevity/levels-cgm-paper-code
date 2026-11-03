#!/usr/bin/env python3
"""Compares the within-person same-food correlation with direct (raw) correlation variants.

Asks whether the within-person r for repeated same-food responses changes if the two sets of meals are compared
directly, without first subtracting each person's average over their foods.

Frame and selection match analysis/meals/same_food_averaging_preventive.py: analysis/meals/_food_analysis.py
load_responses() (isolated single-food meals, raw 2-h iAUC), participants in analysis/validation/diabetes_range_exclusion.py
diabetes_range_users() excluded, analysis/meals/split_half_reliability.py halves() for the k-meal sets (interleaved =
alternate meals of the first 2k; consecutive = first k vs next k, reported as 'time-separated').

Variants, all Pearson r across person-food pairs:
  within_person   set means minus the person's mean over their foods; persons with >= 2 foods
  raw_same_pairs  the same pairs, no demeaning (adds the between-person level)
  raw_all_pairs   every pair, including persons with one food, no demeaning (the direct comparison)
  personal_only   within_person after also removing each food's average across people (food centred on its mean over
                  all pairs of that food; foods with >= 2 persons), i.e. the person-specific part of the food response

95% CI: participant-clustered percentile bootstrap, 2,000 resamples (food means held fixed for personal_only).

Inputs: study database (via LEVELS_DB) through diabetes_range_exclusion.py; OUTPUTS/meals/meal_food_substrate.parquet
through _food_analysis.load_responses().
Outputs: OUTPUTS/meals/same_food_raw_r/{same_food_r_variants.csv, summary.json}. Aggregate rows only.
Usage: python analysis/meals/same_food_raw_vs_within.py
"""
import importlib.util, json, sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from analysis.meals import _food_analysis as shared# noqa: E402

PUB = OUTPUTS
S_SPLIT_HALF = ROOT / 'analysis/meals/split_half_reliability.py'
S_DIABETES_EXCLUSION = ROOT / 'analysis/validation/diabetes_range_exclusion.py'
OUT = OUTPUTS / 'meals/same_food_raw_r'
N_BOOT, SEED = 2000, 20261029


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m


def r_from(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def variant(h, kind, food_means=None):
    """Return a frame with columns uid, x, y for the variant (rows = pairs)."""
    h = h.copy()
    if kind in ('within_person', 'raw_same_pairs', 'personal_only'):
        h = h.loc[h.groupby('uid').a.transform('size').ge(2)].copy()
    if kind == 'personal_only':
        h = h.merge(food_means, on='food', how='inner')
        h = h.loc[h.groupby('uid').a.transform('size').ge(2)].copy()
        h['a'] = h.a - h.fa; h['b'] = h.b - h.fb
    if kind in ('within_person', 'personal_only'):
        h['a'] = h.a - h.groupby('uid').a.transform('mean'); h['b'] = h.b - h.groupby('uid').b.transform('mean')
    return h[['uid', 'a', 'b']].rename(columns={'a': 'x', 'b': 'y'})


def boot(v, rng):
    g = v.assign(xx=v.x ** 2, yy=v.y ** 2, xy=v.x * v.y, n=1).groupby('uid')[['n', 'x', 'y', 'xx', 'yy', 'xy']].sum().to_numpy()
    out = []
    for _ in range(N_BOOT):
        n, sx, sy, sxx, syy, sxy = g[rng.integers(0, len(g), len(g))].sum(axis=0)
        cxy, cxx, cyy = sxy - sx * sy / n, sxx - sx * sx / n, syy - sy * sy / n
        out.append(cxy / np.sqrt(cxx * cyy))
    return np.percentile(out, [2.5, 97.5])


def main():
    s_split_half = load('s_split_half_raw', S_SPLIT_HALF); s_diabetes_exclusion = load('s_diabetes_exclusion_raw', S_DIABETES_EXCLUSION)
    frame = shared.load_responses()
    with duckdb.connect(str(database_path()), read_only=True) as con:
        excl = s_diabetes_exclusion.diabetes_range_users(con)
    frame = frame.loc[~frame.uid.isin(excl)].reset_index(drop=True)
    rng = np.random.default_rng(SEED); rows = []
    for split in ('interleaved', 'consecutive'):
        for k in range(1, 8):
            h = s_split_half.halves(frame, k, split)
            # food means over all pairs of that food (>= 2 persons)
            fm = h.groupby('food').agg(fa=('a', 'mean'), fb=('b', 'mean'), np_=('uid', 'nunique')).reset_index()
            fm = fm.loc[fm.np_.ge(2), ['food', 'fa', 'fb']]
            for kind in ('within_person', 'raw_same_pairs', 'raw_all_pairs', 'personal_only'):
                v = variant(h, kind, fm)
                r = r_from(v.x.to_numpy(), v.y.to_numpy())
                lo, hi = boot(v, rng) if k in (1, 7) else (np.nan, np.nan)
                rows.append(dict(split='time-separated' if split == 'consecutive' else 'interleaved', k=k, variant=kind,
                                 n_pairs=int(len(v)), n_participants=int(v.uid.nunique()), r=r, ci_low=lo, ci_high=hi))
    T = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / 'same_food_r_variants.csv', index=False)
    meta = dict(n_boot=N_BOOT, seed=SEED)
    (OUT / 'summary.json').write_text(json.dumps(meta, indent=1))
    print(T.loc[T.k.isin([1, 7])].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
