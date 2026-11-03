#!/usr/bin/env python3
"""Crossed food-response variance decomposition on the log scale, preventive population.

Descriptive, cross-sectional variance decomposition; no projection.

Sample: analysis/meals/_food_analysis.load_responses(identity='name'), inf/NaN 2-h iAUC dropped,
normalized food names eaten by >=50 participants. Preventive population: participants with any
accepted diabetes-range laboratory value (HbA1c >=6.5% decimal half-up, or fasting glucose
>=126 mg/dL; analysis/validation/diabetes_range_exclusion.py diabetes_range_users()) are removed
first and the >=50-participant food rule is applied afterwards.

Model: response = person + food + person x food + occasion, via analysis/meals/food_variance.py
henderson3() (Monte Carlo EMS method of moments, 500 traces, negative components truncated at
zero for shares). Primary response: log(IAUC_2H + 200). Reported: shares of total variance (pct_P, pct_F, pct_PF, pct_E) and the
person-specific share of food-linked variation PF/(F+PF).

Intervals: seeded participant HALF-samples drawn without replacement
(half b = rng(SEED + b).permutation(n_participants) < n_participants // 2; henderson3 seed
m_food_variance.SEED + b). Intervals are 2.5/97.5 percentiles of half-sample
estimates. The participant bootstrap is not used: resampling with replacement duplicates participants
and inflates the apparent person x food component.

Food groups: reference foods map raw_name -> merged_food_id through a curated food-name harmonization table and
merged_food_id -> group through a curated food-group map (both restricted inputs, placed under OUTPUTS/meals/food_groups/). The
decomposition is refit within each group that has >=10 reference foods, on the same participant
half-samples. The mapping rate is reported.

Structural checks (fail closed): internal consistency between the raw-scale and preventive-sample
henderson3 fits in this run; diabetes_range_users() captures all canonical-panel diabetes-range
participants; every reference food maps to a food group (mapping rate reported; failure if below 95%).

Inputs: LEVELS_DB tables via analysis/meals/_food_analysis.load_responses; OUTPUTS/meals/
food_general_vs_personal_variance_components.csv (analysis/meals/food_variance.py);
OUTPUTS/validation/diabetes_range_exclusion_sensitivity/exclusion_sensitivity.csv
(analysis/validation/diabetes_range_exclusion.py); OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv
(analysis/canonical_panel/build_panel.py); OUTPUTS/meals/food_groups/name_map_merged.parquet and food_group_map.csv
(curated restricted inputs, not produced by this repository).

Outputs: OUTPUTS/meals/food_variance_log_scale/variance_components.csv, variance_intervals.csv,
food_group_components.csv, halfsamples.csv, summary.json, summary.md, and
restricted/reference_food_groups.csv (contains raw food names; participant-level, local audit only,
not for publication).

Usage: python analysis/meals/food_variance_log_scale.py [--half-samples N] [--workers N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
from analysis.meals import _food_analysis as shared# noqa: E402

VERSION = 'food-variance-log-scale'
PUB = OUTPUTS
OUT = PUB / 'meals/food_variance_log_scale'
DB = database_path()
COMPONENTS = PUB / 'meals/food_general_vs_personal_variance_components.csv'
S_DIABETES_EXCLUSION_TABLE = PUB / 'validation/diabetes_range_exclusion_sensitivity/exclusion_sensitivity.csv'
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
NAME_MAP = PUB / 'meals/food_groups/name_map_merged.parquet'        # curated food-name harmonization (restricted input)
GROUP_MAP = PUB / 'meals/food_groups/food_group_map.csv'            # curated food-group map (restricted input)
SEED = 20260926
N_HALF = 100
MIN_USERS = 50
MIN_GROUP_FOODS = 10
LOG_OFFSET = 200.0
COMPONENT_KEYS = ['pct_P', 'pct_F', 'pct_PF', 'pct_E']


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


m_food_variance = load('food_variance_module', 'analysis/meals/food_variance.py')


def reference(frame):
    """Named foods eaten by >=50 participants, after inf/NaN iAUC removal."""
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=['iauc'])
    cnt = frame.groupby('food').uid.nunique()
    ref = frame.loc[frame.food.isin(cnt.index[cnt.ge(MIN_USERS)])].copy()
    ref['log200'] = np.log(ref.iauc + LOG_OFFSET)
    return ref


def shares(h):
    f, pf = h['s2_F'], h['s2_PF']
    return dict(pct_P=float(h['pct_P']), pct_F=float(h['pct_F']), pct_PF=float(h['pct_PF']), pct_E=float(h['pct_E']),
                pf_share_of_food_linked=float(pf / (f + pf)) if f + pf > 0 else np.nan,
                s2_P=float(h['s2_P']), s2_F=float(f), s2_PF=float(pf), s2_E=float(h['s2_E']),
                n_boundary_components=int(h['n_boundary_components']), connected_components=int(h['connected_components']),
                N=int(h['N']), nP=int(h['nP']), nF=int(h['nF']), nPF=int(h['nPF']))


def fit(frame, col, seed=None):
    h = m_food_variance.henderson3(frame, col) if seed is None else m_food_variance.henderson3(frame, col, n_trace=m_food_variance.N_TRACE, seed=seed)
    return shares(h)


# ---------------------------------------------------------------- half-sample workers
# Independent subprocesses (the host sandbox forbids multiprocessing semaphores). Each worker
# fits half-samples b = slot, slot + n_workers, ...; results do not depend on the worker count.
def _worker(tmp, slot, n_workers, n_half, groups):
    frame = pd.read_parquet(Path(tmp) / 'frame.parquet')
    users = np.array(sorted(frame.uid.unique()))
    rows = []
    for b in range(slot, n_half, n_workers):
        rows += _half(b, groups, frame, users)
    (Path(tmp) / f'rows_{slot}.json').write_text(json.dumps(shared.json_safe(rows)))


def run_half_samples(frame, groups, n_half, n_workers):
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix='food_variance_log_') as tmp:
        frame[['uid', 'food', 'log200', 'group']].to_parquet(Path(tmp) / 'frame.parquet', index=False)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', MKL_NUM_THREADS='1')
        procs = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--_worker', tmp, str(k), str(n_workers),
                                   str(n_half), json.dumps(groups)], env=env) for k in range(n_workers)]
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f'Half-sample worker failed: exit codes {codes}')
        rows = [r for k in range(n_workers) for r in json.loads((Path(tmp) / f'rows_{k}.json').read_text())]
    order = {g: i for i, g in enumerate(['all', *groups])}
    return sorted(rows, key=lambda r: (r['half'], order[r['group']]))


def _half(b, groups, frame, users):
    pick = np.random.default_rng(SEED + b).permutation(len(users)) < len(users) // 2
    sub = frame.loc[frame.uid.isin(set(users[pick]))]
    rows = []
    for g in [None, *groups]:
        s = sub if g is None else sub.loc[sub.group.eq(g)]
        row = dict(half=b, group='all' if g is None else g, n_resp=int(len(s)), n_users=int(s.uid.nunique()),
                   n_foods=int(s.food.nunique()))
        try:
            row.update(fit(s, 'log200', seed=m_food_variance.SEED + b))
            row['error'] = ''
        except (ValueError, RuntimeError) as err:
            row['error'] = str(err)
        rows.append(row)
    return rows


def interval(values):
    v = pd.Series(values, dtype=float).dropna()
    if len(v) == 0:
        return dict(q025=np.nan, q975=np.nan, median=np.nan, sd=np.nan, n_ok=0)
    return dict(q025=float(v.quantile(.025)), q975=float(v.quantile(.975)), median=float(v.median()),
                sd=float(v.std()), n_ok=int(len(v)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--half-samples', type=int, default=N_HALF)
    ap.add_argument('--workers', type=int, default=min(10, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    t0 = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True); (OUT / 'restricted').mkdir(exist_ok=True)
    import duckdb
    s_diabetes_exclusion = load('diabetes_exclusion_module', 'analysis/validation/diabetes_range_exclusion.py')
    with duckdb.connect(str(DB), read_only=True) as con:
        excl = s_diabetes_exclusion.diabetes_range_users(con)
    panel = pd.read_csv(PANEL, usecols=['USER_ID', 'no_diabetes_range'], low_memory=False)
    pflag = set(panel.loc[panel.no_diabetes_range == False, 'USER_ID'])  # noqa: E712
    guards = dict(excluded_users=len(excl), panel_diabetes_range=len(pflag), panel_diabetes_range_captured=len(pflag & excl))

    fr = shared.load_responses(identity='name')
    full = reference(fr)
    prev = reference(fr.loc[~fr.uid.isin(excl)])
    guards['removed_participants_in_reference_input'] = int(fr.loc[fr.uid.isin(excl)].uid.nunique())

    full_raw = fit(full, 'iauc')
    prev_raw = fit(prev, 'iauc')
    print(f'point estimates (raw scale) computed ({time.monotonic() - t0:.0f}s)', flush=True)

    # Point estimates
    full_log = fit(full, 'log200')
    prev_log = fit(prev, 'log200')
    rows = [dict(sample='full_unrestricted', scale='raw_iauc', **full_raw), dict(sample='full_unrestricted', scale='log_iauc_plus_200', **full_log),
            dict(sample='preventive', scale='raw_iauc', **prev_raw), dict(sample='preventive', scale='log_iauc_plus_200', primary=True, **prev_log)]
    V = pd.DataFrame(rows); V['primary'] = V.primary.fillna(False).astype(bool)

    # Food groups
    nm = pd.read_parquet(NAME_MAP, columns=['raw_name', 'merged_food_id'])
    if not nm.raw_name.is_unique:
        raise SystemExit('GUARD FAILED: name map raw_name not unique')
    gm = pd.read_csv(GROUP_MAP)
    if not gm.food_id.is_unique:
        raise SystemExit('GUARD FAILED: food group map food_id not unique')
    foods = pd.DataFrame(dict(food=sorted(prev.food.unique())))
    foods = foods.merge(nm, left_on='food', right_on='raw_name', how='left').merge(gm, left_on='merged_food_id', right_on='food_id', how='left')
    mapped = foods.group.notna()
    mapping = dict(reference_foods=int(len(foods)), mapped_to_merged_food=int(foods.merged_food_id.notna().sum()),
                   mapped_to_group=int(mapped.sum()), mapping_rate=float(mapped.mean()),
                   reference_responses=int(len(prev)), responses_mapped=int(prev.food.isin(set(foods.food[mapped])).sum()),
                   distinct_merged_foods=int(foods.merged_food_id.nunique()),
                   foods_by_group={k: int(v) for k, v in foods.group.value_counts().sort_index().items()})
    guards['mapping_rate'] = mapping['mapping_rate']
    if mapping['mapping_rate'] < 0.95:
        raise SystemExit(f'GUARD FAILED: food-group mapping rate {mapping["mapping_rate"]:.3f} < 0.95')
    prev = prev.merge(foods[['food', 'group']], on='food', how='left', validate='many_to_one')
    fcount = foods.group.value_counts()
    groups = sorted(fcount.index[fcount.ge(MIN_GROUP_FOODS)])
    G = []
    for g in groups:
        s = prev.loc[prev.group.eq(g)]
        row = dict(group=g, n_reference_foods=int(fcount[g]))
        try:
            lg = fit(s, 'log200'); rw = fit(s, 'iauc')
            row.update(lg); row.update(raw_pct_PF=rw['pct_PF'], raw_pf_share_of_food_linked=rw['pf_share_of_food_linked'],
                                       raw_n_boundary_components=rw['n_boundary_components'], error='')
        except (ValueError, RuntimeError) as err:
            row['error'] = str(err)
        G.append(row)
    G = pd.DataFrame(G)
    print(f'point estimates and {len(groups)} groups done ({time.monotonic() - t0:.0f}s)', flush=True)

    # Half-samples (pooled + groups share participant halves)
    hs_frame = prev.assign(group=prev.group.fillna('__unmapped__'))
    H = pd.DataFrame(run_half_samples(hs_frame, groups, args.half_samples, args.workers))
    if sorted(H.half.unique()) != list(range(args.half_samples)) or len(H) != args.half_samples * (1 + len(groups)):
        raise SystemExit('GUARD FAILED: incomplete half-sample results')
    print(f'{args.half_samples} half-samples done ({time.monotonic() - t0:.0f}s)', flush=True)
    Hp = H.loc[H.group.eq('all')]
    I = []
    for k in ['pf_share_of_food_linked', *COMPONENT_KEYS]:
        I.append(dict(group='all', quantity=k, estimate=float(prev_log[k]), **interval(Hp[k])))
    for g in groups:
        hg = H.loc[H.group.eq(g)]; est = G.set_index('group').loc[g]
        for k in ['pf_share_of_food_linked', *COMPONENT_KEYS]:
            I.append(dict(group=g, quantity=k, estimate=float(est[k]) if k in est and pd.notna(est[k]) else np.nan, **interval(hg[k])))
    I = pd.DataFrame(I)
    I['n_half_samples'] = args.half_samples
    gi = I.loc[I.quantity.eq('pf_share_of_food_linked')].set_index('group')
    G = G.merge(gi[['q025', 'q975', 'median', 'n_ok']].rename(columns=lambda c: f'pf_share_{c}'), left_on='group', right_index=True, how='left')

    files = {'variance_components.csv': V, 'variance_intervals.csv': I, 'food_group_components.csv': G,
             'halfsamples.csv': H}
    for n, f in files.items():
        f.to_csv(OUT / n, index=False)
    foods[['food', 'merged_food_id', 'group']].to_csv(OUT / 'restricted/reference_food_groups.csv', index=False)
    pi = I.loc[I.group.eq('all')].set_index('quantity')
    summary = dict(
        primary=dict(sample=f"preventive ({guards['excluded_users']} diabetes-range participants excluded)", scale='log(IAUC_2H + 200)',
                     n_responses=prev_log['N'], n_participants=prev_log['nP'], n_foods=prev_log['nF'], n_cells=prev_log['nPF'],
                     **{k: prev_log[k] for k in ['pct_P', 'pct_F', 'pct_PF', 'pct_E', 'pf_share_of_food_linked', 'n_boundary_components']},
                     intervals={k: dict(q025=float(pi.loc[k, 'q025']), q975=float(pi.loc[k, 'q975']), n_ok=int(pi.loc[k, 'n_ok']))
                                for k in pi.index}),
        food_group_mapping=mapping, groups_fitted=groups, min_group_foods=MIN_GROUP_FOODS,
        half_sampling=dict(n_half_samples=args.half_samples, seed=SEED, henderson3_seed='m_food_variance.SEED + b',
                           interval_method='2.5/97.5 percentiles over participant half-samples drawn without replacement; no bootstrap'),
        guards=guards)
    (OUT / 'summary.json').write_text(json.dumps(shared.json_safe(summary), indent=1) + '\n')
    lines = ['# Food-response variance on the log scale, preventive population', '',
             f"Primary: log(IAUC_2H+200), named foods eaten by >=50 participants, {guards['excluded_users']} diabetes-range participants excluded.",
             f'{prev_log["N"]:,} responses; {prev_log["nP"]:,} participants; {prev_log["nF"]} foods.', '',
             V.round(4).to_string(index=False), '', I.round(4).to_string(index=False), '', G.round(4).to_string(index=False), '',
             'Descriptive shares; free-living repeats; portions, preparation and unlogged intake uncontrolled.']
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    meta = dict(analysis='food_variance_log_scale', version=VERSION, status='passed',
                design='descriptive crossed variance decomposition; cross-sectional; no projection', seed=SEED,
                henderson3='food_variance.py henderson3 unchanged (N_TRACE=500, SEED=20260615 for point estimates)',
                log_offset=LOG_OFFSET, min_users=MIN_USERS, min_group_foods=MIN_GROUP_FOODS, n_half_samples=args.half_samples,
                guards=guards, food_group_mapping=mapping,
                restricted_outputs='restricted/reference_food_groups.csv contains raw food names; local audit only, not for publication',
                runtime_seconds=round(time.monotonic() - t0, 1))
    (OUT / 'food_variance_log_scale.metadata.json').write_text(json.dumps(shared.json_safe(meta), indent=1) + '\n')
    print(json.dumps(shared.json_safe(dict(guards=guards, primary=summary['primary'])), indent=1))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--_worker':
        _worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), json.loads(sys.argv[6]))
    else:
        main()
