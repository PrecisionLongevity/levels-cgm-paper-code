"""Fasting insulin before March 2023: reporting pattern and sensitivity of the insulin-marker results to its handling.

The canonical rule (cgmpaper/lab_normalization.py, INSULIN_RULE = "exclude_period") treats every fasting-insulin
draw before 1 March 2023 as missing. This producer
  (1) describes the reporting pattern that motivates the rule (accepted daily values, rule off): per draw month, how
      many are exactly 1.9 uIU/mL and how many other values lie below 2; and the same-day profile and later draws of
      participants whose early value was 1.9 (aggregates only);
  (2) rebuilds the primary-cohort fasting insulin and HOMA-IR under three rules -- exclude_period (canonical),
      floor_code_only (only the early 1.9 values missing) and keep_all -- with the canonical builder's own nearest-draw
      function, and compares unadjusted R2 (average glucose vs HbA1c), percent per SD of average glucose and the
      CGM-only vs HbA1c-only contrast.
Months with fewer than MIN_MONTH_N draws are not written individually.

Inputs: database tables labs, users (LEVELS_DB); OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv
  (analysis/canonical_panel/build_panel.py); OUTPUTS/metabolic/unadjusted_insulin_markers/cgm_measures_unadjusted.csv
  (analysis/metabolic/unadjusted_insulin_markers.py); OUTPUTS/metabolic/hba1c_unadjusted_r2/{unadjusted_r2.csv,
  discordant_groups_unadjusted.csv} (analysis/metabolic/hba1c_unadjusted_r2.py).

Outputs: OUTPUTS/canonical_panel/insulin_fill_value/
  {insulin_fill_value_by_month.csv, insulin_handling_sensitivity.csv, summary.json, insulin_fill_value.metadata.json}

Usage: python analysis/canonical_panel/insulin_value_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS, database_path  # noqa: E402
import cgmpaper.lab_normalization as LN  # noqa: E402
import analysis.canonical_panel.build_panel as B  # noqa: E402

DB = database_path()
PUB = OUTPUTS
PANEL = PUB / 'canonical_panel/canonical_lab_cgm_panel.csv'
U_UNADJUSTED = PUB / 'metabolic/unadjusted_insulin_markers/cgm_measures_unadjusted.csv'
R_UNADJUSTED_R2 = PUB / 'metabolic/hba1c_unadjusted_r2'
OUT = PUB / 'canonical_panel/insulin_fill_value'
MIN_MONTH_N = 20
BELOW = 2.0
FILL_PERIOD = ('2022-07', '2022-12')   # months described in Methods
RULES = ('exclude_period', 'floor_code_only', 'keep_all')
N_BOOT, SEED = 2000, 20261019
TOL = 1e-9


def fail(msg):
    raise SystemExit(f'GUARD FAILED: {msg}')


def with_rule(rule, fn):
    """Set INSULIN_RULE on every loaded copy of lab_normalization (the builder imports it as shared.lab_normalization)."""
    mods = [m for name, m in list(sys.modules.items()) if name.endswith('lab_normalization') and hasattr(m, 'INSULIN_RULE')]
    old = [m.INSULIN_RULE for m in mods]
    for m in mods:
        m.INSULIN_RULE = rule
    try:
        return fn()
    finally:
        for m, o in zip(mods, old):
            m.INSULIN_RULE = o


def describe(raw):
    daily, _, _ = with_rule('keep_all', lambda: LN.normalize_labs(raw))
    _, _, audit_on = LN.normalize_labs(raw)
    w = daily.pivot_table(index=['USER_ID', 'DRAW_DATE'], columns='TYPE', values='value').reset_index()
    w['DRAW_DATE'] = pd.to_datetime(w.DRAW_DATE)
    d = w.dropna(subset=['fasting_insulin']).copy()
    d['month'] = d.DRAW_DATE.dt.strftime('%Y-%m')
    d['code'] = (d.fasting_insulin - LN.INSULIN_FLOOR_CODE).abs().lt(1e-4)
    d['other_below'] = d.fasting_insulin.lt(BELOW) & ~d.code
    d['early'] = d.DRAW_DATE.lt(LN.INSULIN_EXCLUDE_BEFORE)
    m = d.groupby('month').agg(n=('code', 'size'), n_code_1_9=('code', 'sum'), n_other_below_2=('other_below', 'sum')).reset_index()
    m['code_share_pct'] = 100 * m.n_code_1_9 / m.n
    m['before_cutoff'] = pd.to_datetime(m.month + '-01') < LN.INSULIN_EXCLUDE_BEFORE
    big = m.loc[m.n >= MIN_MONTH_N]
    per = big.loc[big.month.between(*FILL_PERIOD)]
    pre, post = d.loc[d.early], d.loc[~d.early]
    grp = np.select([d.early & d.code, d.early & ~d.code, ~d.early & d.fasting_insulin.lt(3)],
                    ['early_1_9', 'early_other', 'later_below_3'], 'later_3_plus')
    prof = d.assign(group=grp).groupby('group').agg(
        n=('fasting_insulin', 'size'), insulin=('fasting_insulin', 'median'), fasting_glucose=('fasting_glucose', 'median'),
        triglycerides=('triglycerides', 'median'), hdl=('hdl_cholesterol', 'median'))
    first_pre = pre.sort_values('DRAW_DATE').groupby('USER_ID').first()
    later = post.sort_values('DRAW_DATE').groupby('USER_ID').first()[['fasting_insulin']].rename(columns={'fasting_insulin': 'later'})
    fu = first_pre.join(later, how='inner')
    follow = {('early_1_9' if k else 'early_other'): dict(n=int(len(s)), later_median=float(s.later.median()))
              for k, s in fu.groupby('code')}
    summ = dict(
        floor_code=LN.INSULIN_FLOOR_CODE, cutoff=str(LN.INSULIN_EXCLUDE_BEFORE.date()), insulin_rule=LN.INSULIN_RULE,
        lab_definition_version=LN.LAB_DEFINITION_VERSION, fill_period=list(FILL_PERIOD), fill_period_months_reported=int(len(per)),
        fill_period_monthly_share_min_pct=float(per.code_share_pct.min()), fill_period_monthly_share_max_pct=float(per.code_share_pct.max()),
        fill_period_other_values_below_2=int(d.loc[d.month.between(*FILL_PERIOD), 'other_below'].sum()),
        before_cutoff_draws=int(len(pre)), before_cutoff_code_1_9=int(pre.code.sum()),
        from_cutoff_draws=int(len(post)), from_cutoff_fill_share_pct=float(100 * post.code.mean()),
        from_cutoff_other_values_below_2=int(post.other_below.sum()),
        rows_excluded_by_rule=int(audit_on['fasting_insulin_rows_before_2023_03_excluded']),
        same_day_profile={k: {c: (int(v) if c == 'n' else float(v)) for c, v in r.items()} for k, r in prof.iterrows()},
        later_draws_same_participants=follow, min_month_n=MIN_MONTH_N)
    if summ['fill_period_other_values_below_2'] > 1 or summ['before_cutoff_code_1_9'] == 0:
        fail(f'reporting pattern not as described {summ}')
    if summ['rows_excluded_by_rule'] != summ['before_cutoff_draws']:
        fail('exclude_period rule does not remove every early draw')
    return big, summ


def rebuild(con, anchors, rule):
    def go():
        B.prepare_laboratories(con)
        wide = B.lab_wide(con)
        return B.nearest_within_band(anchors, wide, 'DRAW_DATE', ['fasting_insulin', 'homa_ir'], B.LAB_BAND_DAYS)
    return with_rule(rule, go)


def r2(y, x):
    return float(np.corrcoef(y, x)[0, 1] ** 2)


def estimates(P, y, rng):
    d = P.loc[P[y].notna() & (P[y] > 0)]
    ly, g, a = np.log(d[y].to_numpy(float)), d.mean_glucose.to_numpy(float), LN.round_a1c(d.a1c.to_numpy(float))
    row = dict(marker=y, n=int(len(d)))
    for scope, msk in (('all_primary', np.ones(len(d), bool)), ('hba1c_normal_lt57', a < 5.7)):
        yy, gg, aa = ly[msk], g[msk], a[msk]
        idx = rng.integers(0, len(yy), (N_BOOT, len(yy)))
        bs = np.array([r2(yy[i], gg[i]) - r2(yy[i], aa[i]) for i in idx])
        row.update({f'{scope}_n': int(msk.sum()), f'{scope}_r2_cgm': r2(yy, gg), f'{scope}_r2_hba1c': r2(yy, aa),
                    f'{scope}_r2_diff': r2(yy, gg) - r2(yy, aa),
                    f'{scope}_r2_diff_ci_low': float(np.percentile(bs, 2.5)), f'{scope}_r2_diff_ci_high': float(np.percentile(bs, 97.5))})
    z = (g - g.mean()) / g.std(ddof=1)
    f = sm.OLS(ly, sm.add_constant(z)).fit(cov_type='HC3'); ci = f.conf_int()[1]
    row.update(pct_per_sd=100 * np.expm1(f.params[1]), pct_per_sd_ci_low=100 * np.expm1(ci[0]), pct_per_sd_ci_high=100 * np.expm1(ci[1]))
    grp = np.select([(a < 5.7) & (g >= 110), (a >= 5.7) & (g < 110)], ['cgm_only', 'a1c_only'], 'other')
    k = grp != 'other'
    f = sm.OLS(ly[k], sm.add_constant((grp[k] == 'cgm_only').astype(float))).fit(cov_type='HC3'); ci = f.conf_int()[1]
    row.update(cgm_only_vs_a1c_only_pct=100 * np.expm1(f.params[1]), cgm_only_vs_a1c_only_ci_low=100 * np.expm1(ci[0]),
               cgm_only_vs_a1c_only_ci_high=100 * np.expm1(ci[1]))
    return row


def main():
    with duckdb.connect(str(DB), read_only=True) as con:
        raw = con.execute("""
            SELECT l.USER_ID, l.TYPE, l.DRAW_DATE, l.VALUE, l.UNIT
            FROM labs l JOIN users u ON u.USER_ID = l.USER_ID
            WHERE u.IS_INTERNAL_USER = FALSE AND l.TYPE IN ('fasting_insulin', 'fasting_glucose', 'triglycerides', 'hdl_cholesterol')
        """).fetchdf()
        big, summ = describe(raw)
        panel = pd.read_csv(PANEL)
        P0 = panel.loc[panel.no_diabetes_range.astype(bool)].copy()
        anchors = P0[['USER_ID', 'anchor_date']].copy()
        anchors['anchor_date'] = pd.to_datetime(anchors.anchor_date)
        picks = {rule: rebuild(con, anchors, rule) for rule in RULES}
    # G1: canonical rule reproduces the panel exactly
    chk = P0[['USER_ID', 'fasting_insulin', 'homa_ir']].merge(picks['exclude_period'], on='USER_ID', suffixes=('', '_re'), validate='one_to_one')
    for c in ('fasting_insulin', 'homa_ir'):
        a, b = chk[c].to_numpy(float), chk[f'{c}_re'].to_numpy(float)
        if not (np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) < TOL):
            fail(f'exclude_period rebuild does not reproduce the canonical panel column {c}')
    rng = np.random.default_rng(SEED)
    rows = []
    for rule in RULES:
        P = P0.drop(columns=['fasting_insulin', 'homa_ir']).merge(picks[rule][['USER_ID', 'fasting_insulin', 'homa_ir']], on='USER_ID', validate='one_to_one')
        for y in ('fasting_insulin', 'homa_ir'):
            rows.append(dict(rule=rule, **estimates(P, y, rng)))
    S = pd.DataFrame(rows)
    # G2: canonical-rule estimates equal those of unadjusted_insulin_markers.py and hba1c_unadjusted_r2.py (point estimates)
    U = pd.read_csv(U_UNADJUSTED); R = pd.read_csv(R_UNADJUSTED_R2 / 'unadjusted_r2.csv'); Dg = pd.read_csv(R_UNADJUSTED_R2 / 'discordant_groups_unadjusted.csv')
    for r in S.loc[S.rule == 'exclude_period'].itertuples():
        u = U.loc[(U.family == 'cgm') & (U.marker == r.marker) & (U.adjustment == 'none') & (U.measure == 'mean_glucose')].iloc[0]
        q = R.loc[(R.outcome == r.marker) & (R.scope == 'all_primary')].iloc[0]
        q57 = R.loc[(R.outcome == r.marker) & (R.scope == 'hba1c_normal_lt57')].iloc[0]
        dg = Dg.loc[(Dg.outcome == r.marker) & (Dg.contrast == 'cgm_only vs a1c_only')].iloc[0]
        pairs = [(r.n, u.n), (r.n, q.n), (r.pct_per_sd, u.estimate), (r.all_primary_r2_cgm, q.r2_cgm), (r.all_primary_r2_hba1c, q.r2_hba1c),
                 (r.hba1c_normal_lt57_r2_cgm, q57.r2_cgm), (r.hba1c_normal_lt57_r2_hba1c, q57.r2_hba1c), (r.cgm_only_vs_a1c_only_pct, dg.pct_difference)]
        if any(abs(float(x) - float(yv)) > 1e-6 for x, yv in pairs):
            fail(f'{r.marker}: canonical-rule estimates differ from unadjusted_insulin_markers.py / hba1c_unadjusted_r2.py: {pairs}')
    summ['sensitivity_rules'] = list(RULES)
    summ['sensitivity'] = {f'{r.rule}:{r.marker}': {k: (int(v) if k.endswith('n') and k != 'marker' else v) for k, v in r._asdict().items()
                                                   if k not in ('Index', 'rule', 'marker')} for r in S.itertuples()}
    OUT.mkdir(parents=True, exist_ok=True)
    big[['month', 'n', 'n_code_1_9', 'n_other_below_2', 'code_share_pct', 'before_cutoff']].to_csv(OUT / 'insulin_fill_value_by_month.csv', index=False)
    S.to_csv(OUT / 'insulin_handling_sensitivity.csv', index=False)
    (OUT / 'summary.json').write_text(json.dumps(summ, indent=1, default=float) + '\n')
    meta = dict(script=str(Path(__file__).resolve().relative_to(ROOT)),
                guards=dict(G1_exclude_period_reproduces_panel=True, G2_canonical_estimates_equal_unadjusted_unadjusted_r2=True),
                n_boot=N_BOOT, seed=SEED)
    (OUT / 'insulin_fill_value.metadata.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(S.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
