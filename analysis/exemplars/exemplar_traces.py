#!/usr/bin/env python3
"""Rule-based real-trace exemplars (food, person, person x food, meal-to-meal); Fig. 6a draws the meal-to-meal one.

Selects, by deterministic rule, representative real glucose traces for four mini-panels and writes a candidate
list and contact sheet alongside the selected exemplars. The variance shares shown beside the traces are read by
the builder from analysis/meals/food_variance_log_scale.py.

Population and meals (identical to the raw-iAUC 'preventive' row of analysis/meals/food_variance_log_scale.py)
  corrected single-food substrate (meal_food_substrate.parquet via _food_analysis.load_responses: one log and one
  macro row per isolated 2/2 meal, all 24 post-meal cells observed), participants without a diabetes-range
  laboratory value (analysis/validation/diabetes_range_exclusion.py roster), named foods eaten by >= 50
  participants (analysis/meals/food_variance_log_scale.py reference()).
Traces
  5-minute observed cells g0..g29 of meal_cells_observed.parquet = [-30, +120) min, minus BASELINE_PRE (average of
  the three observed cells in [-15, 0)). No interpolation; unobserved pre-baseline cells stay missing. Baseline and
  positive 2-h iAUC are recomputed from the cells of every meal drawn and checked against the substrate values.
Rules (generic food labels only: no parentheses and no term of the brand-name list)
  food            reference foods with >= 200 meals; food average iAUC = average over its meals; targets = 10th,
                  50th, 90th percentile of food averages; candidate distance |avg - target| / SD(food averages);
                  ties by more meals, then name. Curve: average trace over all meals of the food.
  person          participants with >= 20 meals of >= 5 reference foods; person level = average over their meals of
                  (iAUC - food average iAUC); targets 10th/50th/90th percentile; distance |level - target| / SD;
                  ties by more meals. Curve, on the same food-adjusted footing as the level: the average trace of all
                  reference meals plus the person's average of (meal trace - that food's average trace).
  person_x_food   person-food cells with >= 4 meals; for every food pair (both generic, from DIFFERENT food groups of
                  reference_food_groups.csv, so that name variants such as two coffees do not pair)
                  with >= 3 people holding both cells, and every such person,
                  d = mean iAUC(food 1) - mean iAUC(food 2), t = d / sqrt(var1/n1 + var2/n2); per food pair the
                  person with the largest t and a different person with the most negative t; score = min(t+, -t-);
                  pairs ranked by score (ties by food names). Curves: each person's average trace per food.
  meal_to_meal    person-food cells with >= 12 meals; within-cell SD of iAUC; the cell nearest the median SD of
                  such cells (typical scatter); ties by more meals. Curves: its first 20 meals in time (thin) and
                  the average of all its meals (bold).
Checks: the analysis frame reproduces the raw preventive N, nP, nF, nPF of food_variance_log_scale.py; cell-level
  reproduction of baseline and iAUC for every meal drawn; every selected food label is generic; the person x food
  flip holds in both persons (t+ > 0 > t-).

Inputs
  OUTPUTS/meals/meal_food_substrate.parquet, OUTPUTS/meals/meal_cells_observed.parquet (analysis/meals/build_meals.py);
  OUTPUTS/meals/food_variance_log_scale/variance_components.csv and reference_food_groups.csv
  (analysis/meals/food_variance_log_scale.py); diabetes-range exclusion roster (analysis/validation/
  diabetes_range_exclusion.py); database via LEVELS_DB.
Outputs (OUTPUTS/exemplars/traces/, individual traces restricted to internal review; no identifiers drawn)
  fig6_trace_exemplars.json (selected exemplars), fig6_trace_candidates.json (full candidate lists and traces),
  fig6_contact_sheet.png (visual review sheet), fig6_trace_exemplars.metadata.json.
Usage
  python analysis/exemplars/exemplar_traces.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
sys.path.insert(0, str(ROOT / 'analysis/daily_glucose'))
from analysis.meals import _food_analysis as shared# noqa: E402

VERSION = 'trace-exemplars'
PUB = OUTPUTS
OUT = PUB / 'exemplars/traces'
DB = database_path()
CELLS = PUB / 'meals/meal_cells_observed.parquet'
FOOD_SUB = PUB / 'meals/meal_food_substrate.parquet'
V_FOOD_VARIANCE_LOG = PUB / 'meals/food_variance_log_scale'
S_FOOD_VARIANCE_LOG = ROOT / 'analysis/meals/food_variance_log_scale.py'
S_DIABETES_EXCLUSION = ROOT / 'analysis/validation/diabetes_range_exclusion.py'
N_DISPLAY = 30                  # g0..g29 = [-30, +120) min
BASE_CELLS = (3, 4, 5)          # [-15, 0)
POST = slice(6, 30)             # [0, 120)
MIN_FOOD_MEALS = 200
MIN_PERSON_MEALS, MIN_PERSON_FOODS = 20, 5
MIN_PXF_CELL, MIN_PXF_PEOPLE = 4, 3
GROUPS = V_FOOD_VARIANCE_LOG / 'restricted/reference_food_groups.csv'
MIN_M2M_CELL, M2M_SHOW = 12, 20
N_CAND = 6
BRAND_TERMS = ('coke', 'coca-cola', 'pepsi', 'fairlife', 'starbucks', 'chobani', 'fage', 'kind', 'rxbar', 'quest',
               'siggi', 'oikos', 'lara', 'kirkland', 'trader joe', 'ag1', 'athletic green', 'la croix', 'lacroix', 'premier',
               'orgain', 'optimum', 'dave', 'ezekiel', 'huel', 'soylent', 'celsius', 'red bull', 'monster', 'olipop',
               'poppi', 'liquid iv', 'lmnt', 'nuun', 'chipotle', 'mcdonald', 'sweetgreen', 'cava', 'subway', 'jersey mike',
               'bulletproof', 'perfect bar', 'iq bar', 'rx bar', 'grass fed beef snack sticks')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def generic(s):
    s = str(s).lower()
    return bool(s.strip()) and '(' not in s and ')' not in s and not any(t in s for t in BRAND_TERMS)


def guard(ok, what):
    if not ok:
        raise SystemExit(f'GUARD FAILED: {what}')


def frame_food_variance_log(guards):
    m_food_variance_log = json.loads((V_FOOD_VARIANCE_LOG / 'food_variance_log_scale.metadata.json').read_text())
    s_food_variance_log, s_diabetes_exclusion = load('food_variance_log_for_exemplar_traces', S_FOOD_VARIANCE_LOG), load('diabetes_exclusion_for_exemplar_traces', S_DIABETES_EXCLUSION)
    import duckdb
    with duckdb.connect(str(DB), read_only=True) as con:
        excl = s_diabetes_exclusion.diabetes_range_users(con)
    fr = shared.load_responses(identity='name')
    ref = s_food_variance_log.reference(fr.loc[~fr.uid.isin(excl)])
    V = pd.read_csv(V_FOOD_VARIANCE_LOG / 'variance_components.csv')
    raw = V.loc[(V['sample'] == 'preventive') & (V.scale == 'raw_iauc')].iloc[0]
    got = dict(N=len(ref), nP=ref.uid.nunique(), nF=ref.food.nunique(), nPF=len(ref.groupby(['uid', 'food'])))
    want = dict(N=int(raw.N), nP=int(raw.nP), nF=int(raw.nF), nPF=int(raw.nPF))
    guards['G1_frame_equals_food_variance_log_scale_raw_preventive'] = dict(observed=got, expected=want, match=got == want, excluded_users=len(excl))
    guard(got == want, 'frame does not reproduce the raw preventive counts of analysis/meals/food_variance_log_scale.py')
    return ref.reset_index(drop=True), m_food_variance_log


def load_cells(meal_ids):
    import pyarrow.dataset as ds
    d = ds.dataset(str(CELLS))
    cols = ['USER_ID', 'MEAL_EVENT_ID'] + [f'g{i}' for i in range(N_DISPLAY)]
    t = d.to_table(columns=cols, filter=ds.field('MEAL_EVENT_ID').isin(sorted(set(meal_ids)))).to_pandas()
    guard(not t.duplicated('MEAL_EVENT_ID').any(), 'duplicate meal cells')
    return t.set_index('MEAL_EVENT_ID')


def deltas(cells, sub, meal_ids, guards, key):
    """Delta traces for meal_ids; guard G2 recomputes baseline and iAUC from the cells."""
    G = cells.loc[list(meal_ids), [f'g{i}' for i in range(N_DISPLAY)]].to_numpy(float)
    S = sub.loc[list(meal_ids)]
    base = G[:, list(BASE_CELLS)].mean(axis=1)
    iauc = 5 * np.clip(G[:, POST] - base[:, None], 0, None).sum(axis=1)
    ok = (np.isfinite(G[:, list(BASE_CELLS)]).all() and np.isfinite(G[:, POST]).all()
          and np.allclose(base, S.BASELINE_PRE.to_numpy(float), atol=1e-6) and np.allclose(iauc, S.IAUC_2H.to_numpy(float), atol=1e-4))
    guards.setdefault('G2_cells_reproduce_baseline_iauc', {})[key] = dict(n_meals=len(G), match=bool(ok))
    guard(ok, f'cells do not reproduce baseline/iAUC for {key}')
    return G - base[:, None]


def avg_curve(D):
    return [None if not np.isfinite(v) else round(float(v), 4) for v in np.nanmean(D, axis=0)]


def one_curve(row):
    return [None if not np.isfinite(v) else round(float(v), 4) for v in row]


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    guards = {}
    ref, m_food_variance_log = frame_food_variance_log(guards)
    ref['generic'] = ref.food.map(generic)
    fav = ref.groupby('food').iauc.agg(['mean', 'size'])
    ref['centred'] = ref.iauc - ref.food.map(fav['mean'])
    sub = pd.read_parquet(FOOD_SUB, columns=['MEAL_EVENT_ID', 'BASELINE_PRE', 'IAUC_2H']).set_index('MEAL_EVENT_ID')
    cand, sel = {}, {}
    # ---- food
    F = fav.loc[fav['size'] >= MIN_FOOD_MEALS].copy()
    F['generic'] = F.index.map(generic)
    sd_f = float(F['mean'].std(ddof=1))
    tf = {q: float(np.percentile(F['mean'], q)) for q in (10, 50, 90)}
    food_c = {}
    for q, t in tf.items():
        E = F.loc[F.generic].assign(distance=lambda d: (d['mean'] - t).abs() / sd_f, name=lambda d: d.index)
        E = E.sort_values(['distance', 'size', 'name'], ascending=[True, False, True]).head(N_CAND)
        food_c[f'p{q}'] = [dict(rank=i + 1, food=f, n_meals=int(r['size']), mean_iauc=float(r['mean']), distance=float(r.distance))
                           for i, (f, r) in enumerate(E.iterrows())]
    cand['food'] = dict(targets={f'p{q}': t for q, t in tf.items()}, candidates=food_c, n_eligible_foods=int(len(F)))
    # ---- person
    pp = ref.groupby('uid').agg(n=('iauc', 'size'), nf=('food', 'nunique'), level=('centred', 'mean'))
    P = pp.loc[(pp.n >= MIN_PERSON_MEALS) & (pp.nf >= MIN_PERSON_FOODS)].copy()
    sd_p = float(P.level.std(ddof=1))
    tp = {q: float(np.percentile(P.level, q)) for q in (10, 50, 90)}
    per_c = {}
    for q, t in tp.items():
        E = P.assign(distance=(P.level - t).abs() / sd_p, uid_=P.index).sort_values(['distance', 'n', 'uid_'], ascending=[True, False, True]).head(N_CAND)
        per_c[f'p{q}'] = [dict(rank=i + 1, uid=u, n_meals=int(r.n), n_foods=int(r.nf), level=float(r.level), distance=float(r.distance))
                          for i, (u, r) in enumerate(E.iterrows())]
    cand['person'] = dict(targets={f'p{q}': t for q, t in tp.items()}, candidates=per_c, n_eligible_people=int(len(P)))
    # ---- person x food
    cs = ref.loc[ref.generic].groupby(['uid', 'food']).iauc.agg(['mean', 'var', 'size']).reset_index()
    cs = cs.loc[cs['size'] >= MIN_PXF_CELL].copy()
    grp = pd.read_csv(GROUPS).set_index('food').group
    guard(set(ref.food) <= set(grp.index), 'food-group map does not cover the reference foods')
    cs['group'] = cs.food.map(grp)
    pr = cs.merge(cs, on='uid', suffixes=('1', '2'))
    pr = pr.loc[(pr.food1 < pr.food2) & (pr.group1 != pr.group2)].copy()
    pr['t'] = (pr.mean1 - pr.mean2) / np.sqrt(pr.var1 / pr.size1 + pr.var2 / pr.size2)
    pr = pr.loc[np.isfinite(pr.t)]
    rows = []
    for (f1, f2), g in pr.groupby(['food1', 'food2'], sort=True):
        if len(g) < MIN_PXF_PEOPLE:
            continue
        g = g.sort_values(['t', 'uid'])
        neg, pos = g.iloc[0], g.iloc[-1]
        if neg.uid == pos.uid or not (pos.t > 0 > neg.t):
            continue
        rows.append(dict(food1=f1, food2=f2, score=float(min(pos.t, -neg.t)), n_people=int(len(g)), uid_a=pos.uid, t_a=float(pos.t),
                         uid_b=neg.uid, t_b=float(neg.t)))
    X = pd.DataFrame(rows).sort_values(['score', 'food1', 'food2'], ascending=[False, True, True]).head(N_CAND).reset_index(drop=True)
    cand['person_x_food'] = dict(candidates=[dict(rank=i + 1, **r) for i, r in enumerate(X.to_dict('records'))], n_food_pairs=int(len(rows)))
    # ---- meal-to-meal
    mc = ref.loc[ref.generic].groupby(['uid', 'food']).iauc.agg(['std', 'size']).reset_index()
    mc = mc.loc[mc['size'] >= MIN_M2M_CELL].copy()
    med = float(mc['std'].median())
    mc['distance'] = (mc['std'] - med).abs()
    M = mc.sort_values(['distance', 'size', 'uid', 'food'], ascending=[True, False, True, True]).head(N_CAND).reset_index(drop=True)
    cand['meal_to_meal'] = dict(median_within_cell_sd=med, n_eligible_cells=int(len(mc)),
                                candidates=[dict(rank=i + 1, uid=r.uid, food=r.food, n_meals=int(r['size']), sd_iauc=float(r['std']))
                                            for i, r in M.iterrows()])
    # ---- cells for every reference meal; G2 on all of them (food average curves need every meal of a food)
    cells = load_cells(ref.meal_id)
    guard(len(cells) == len(ref), 'cells missing for reference meals')
    D_all = deltas(cells, sub, ref.meal_id, guards, 'all_reference_meals')
    grand = np.nanmean(D_all, axis=0)
    fcurve = pd.DataFrame(D_all).groupby(ref.food.to_numpy()).mean()     # nan-skipping average trace per food
    D_adj = D_all - fcurve.loc[ref.food.to_numpy()].to_numpy()

    def curves_food(food):
        ids = ref.loc[ref.food == food, 'meal_id']
        return avg_curve(deltas(cells, sub, ids, guards, f'food:{food}')), len(ids)

    def curves_person(uid):
        m = (ref.uid == uid).to_numpy()
        return avg_curve(grand[None, :] + D_adj[m]), int(m.sum())

    sheet = {}
    sheet['food'] = {q: [dict(c, curve=curves_food(c['food'])[0]) for c in L] for q, L in food_c.items()}
    sheet['person'] = {q: [dict(c, curve=curves_person(c['uid'])[0]) for c in L] for q, L in per_c.items()}
    pxf = []
    for r in X.itertuples():
        e = dict(rank=len(pxf) + 1, foods=[r.food1, r.food2], score=r.score, people=[])
        for uid, t in ((r.uid_a, r.t_a), (r.uid_b, r.t_b)):
            per = dict(uid=uid, t=t, foods={})
            for f in (r.food1, r.food2):
                s = ref.loc[(ref.uid == uid) & (ref.food == f)]
                per['foods'][f] = dict(n_meals=int(len(s)), mean_iauc=float(s.iauc.mean()),
                                       curve=avg_curve(deltas(cells, sub, s.meal_id, guards, f'pxf:{uid[:6]}:{f}')))
            e['people'].append(per)
        pxf.append(e)
    sheet['person_x_food'] = pxf
    m2m = []
    for r in M.itertuples():
        s = ref.loc[(ref.uid == r.uid) & (ref.food == r.food)].sort_values(['ts', 'meal_id'])
        D = deltas(cells, sub, s.meal_id, guards, f'm2m:{r.uid[:6]}:{r.food}')
        m2m.append(dict(rank=len(m2m) + 1, uid=r.uid, food=r.food, n_meals=int(len(s)), sd_iauc=float(r.std),
                        meals=[one_curve(x) for x in D[:M2M_SHOW]], average=avg_curve(D), n_shown=int(min(M2M_SHOW, len(D)))))
    sheet['meal_to_meal'] = m2m
    # ---- selection = rank 1 of every mini-panel; G3, G4
    sel = dict(food=[dict(target=q, **sheet['food'][q][0]) for q in ('p90', 'p50', 'p10')],
               person=[dict(target=q, **sheet['person'][q][0]) for q in ('p90', 'p50', 'p10')],
               person_x_food=pxf[0], meal_to_meal=m2m[0])
    labels = [e['food'] for e in sel['food']] + pxf[0]['foods'] + [m2m[0]['food']]
    guards['G3_generic_labels'] = dict(labels=labels, match=all(generic(x) for x in labels))
    guard(guards['G3_generic_labels']['match'], 'non-generic label selected')
    a, b = pxf[0]['people']
    f1, f2 = pxf[0]['foods']
    flip = a['foods'][f1]['mean_iauc'] > a['foods'][f2]['mean_iauc'] and b['foods'][f2]['mean_iauc'] > b['foods'][f1]['mean_iauc']
    guards['G4_person_x_food_flip'] = dict(t_a=a['t'], t_b=b['t'], match=bool(flip and a['t'] > 0 > b['t']))
    guard(guards['G4_person_x_food_flip']['match'], 'person x food flip does not hold')
    minutes = [-27.5 + 5 * i for i in range(N_DISPLAY)]
    sel['all_meals_average'] = avg_curve(grand[None, :])
    (OUT / 'fig6_trace_exemplars.json').write_text(json.dumps(dict(version=VERSION, minutes=minutes, selected=sel, rules=__doc__.split('Rules')[1].split('Guards')[0].strip()), indent=1))
    (OUT / 'fig6_trace_candidates.json').write_text(json.dumps(dict(version=VERSION, minutes=minutes, candidates=cand, sheet=sheet), indent=1))
    contact_sheet(sheet, minutes, OUT / 'fig6_contact_sheet.png')
    outs = ['fig6_trace_exemplars.json', 'fig6_trace_candidates.json', 'fig6_contact_sheet.png']
    inputs = [FOOD_SUB, FOOD_SUB.with_suffix('.metadata.json'), CELLS, PUB / 'meals/meal_cells_observed.metadata.json',
              V_FOOD_VARIANCE_LOG / 'variance_components.csv', V_FOOD_VARIANCE_LOG / 'food_variance_log_scale.metadata.json', GROUPS, S_FOOD_VARIANCE_LOG, S_DIABETES_EXCLUSION,
              ROOT / 'analysis/meals/_food_analysis.py']
    meta = dict(analysis='fig6_trace_exemplars', version=VERSION, status='passed', guards=guards,
                design=dict(min_food_meals=MIN_FOOD_MEALS, min_person_meals=MIN_PERSON_MEALS, min_person_foods=MIN_PERSON_FOODS,
                            min_pxf_cell=MIN_PXF_CELL, min_pxf_people=MIN_PXF_PEOPLE, pxf_different_food_groups=True, min_m2m_cell=MIN_M2M_CELL, m2m_shown=M2M_SHOW, n_candidates=N_CAND,
                            window_min=[-30, 120], producer_food_variance_log_version=m_food_variance_log['version']),
                inputs=[str(p.relative_to(ROOT)) for p in inputs], outputs=outs,
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'fig6_trace_exemplars.metadata.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(dict(status='passed', n_meals=guards['G1_frame_equals_food_variance_log_scale_raw_preventive']['observed'],
                          runtime=meta['runtime_seconds']), indent=1))


def contact_sheet(sheet, minutes, path):
    """Top candidates per mini-panel, for review by eye (restricted: individual traces; no identifiers drawn)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [('food', q) for q in ('p90', 'p50', 'p10')] + [('person', q) for q in ('p90', 'p50', 'p10')] + [('person_x_food', None), ('meal_to_meal', None)]
    fig, axs = plt.subplots(len(rows), N_CAND, figsize=(3.0 * N_CAND, 2.1 * len(rows)), sharex=True, squeeze=False)
    x = np.array(minutes)
    for i, (kind, q) in enumerate(rows):
        L = sheet[kind][q] if q else sheet[kind]
        for j in range(N_CAND):
            ax = axs[i, j]
            ax.axhline(0, color='0.8', lw=0.6)
            if j >= len(L):
                ax.axis('off'); continue
            c = L[j]
            if kind == 'food':
                ax.plot(x, np.array(c['curve'], float), color='C0')
                ax.set_title(f"food {q} #{c['rank']}: {c['food'][:24]} (n={c['n_meals']})", fontsize=7)
            elif kind == 'person':
                ax.plot(x, np.array(c['curve'], float), color='C3')
                ax.set_title(f"person {q} #{c['rank']}: level {c['level']:+.0f}, n={c['n_meals']}", fontsize=7)
            elif kind == 'person_x_food':
                for k, per in enumerate(c['people']):
                    for f, col in zip(c['foods'], ('k', 'C2')):
                        ax.plot(x + (0 if k == 0 else 160), np.array(per['foods'][f]['curve'], float), color=col, lw=1)
                ax.set_title(f"pxf #{c['rank']}: {c['foods'][0][:14]} (k) vs {c['foods'][1][:14]} (g) score {c['score']:.1f}", fontsize=7)
            else:
                for m in c['meals']:
                    ax.plot(x, np.array(m, float), color='0.6', lw=0.6)
                ax.plot(x, np.array(c['average'], float), color='k', lw=1.6)
                ax.set_title(f"m2m #{c['rank']}: {c['food'][:20]} n={c['n_meals']} SD {c['sd_iauc']:.0f}", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle('Rule-based trace candidates (rank 1 = selected); RESTRICTED - individual traces', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
