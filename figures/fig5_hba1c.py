"""Fig. 5: HbA1c within its normal range compared with average CGM glucose.

  a  average CGM glucose by HbA1c value (95% CI; values with >=30 participants) and the least-squares fits within
     HbA1c <5.7% and 5.7-6.4%
  b  Pearson correlation of HbA1c with average glucose within HbA1c ranges (participant bootstrap 95% CI)
  c  plain R² of log HOMA-IR and log fasting insulin on average glucose vs on HbA1c, same participants, and the R²
     each adds beyond the other (fasting insulin, HbA1c <5.7%)
  d  share with fasting insulin >=10 µIU/mL in the four HbA1c x average-glucose groups (Wilson 95% CI), and the size
     of each group in the primary cohort
  e  fasting insulin and hsCRP in the CGM-elevated-only vs the HbA1c-elevated-only group (unadjusted, HC3 95% CI)

Inputs (analysis outputs)
  metabolic/insulin_markers/hba1c_mean_cgm_table.csv   a, points
  metabolic/hba1c_within_range_slopes/within_range_slopes.csv      a, fitted lines
  metabolic/hba1c_range_resolution/within_range_correlations.csv  b
  metabolic/hba1c_unadjusted_r2/unadjusted_r2.csv                  c
  clinical/hba1c_resolution.json                                   c, R² added beyond the other measure
  clinical/markers/discordance_high_share.csv                      d
  clinical/discordance_groups/group_counts.csv                     d, group sizes
  clinical/markers/discordance_markers.csv                         e
Source data
  source_data/fig5a_hba1c_glucose.csv, fig5b_correlations.csv, fig5c_variance_explained.csv, fig5d_groups.csv,
  fig5e_group_contrast.csv
Outputs
  fig5_hba1c.svg / .png

Usage
  python figures/fig5_hba1c.py source-data
  python figures/fig5_hba1c.py draw
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data  # noqa: E402
from cgmpaper import svgfig as sd  # noqa: E402
from cgmpaper.paths import FIGURES_OUT, OUTPUTS  # noqa: E402
from cgmpaper.svgfig import F, S  # noqa: E402

NAME = 'fig5_hba1c'
TITLE = 'Fig. 5 | HbA1c within its normal range compared with average CGM glucose'
MP, CL = OUTPUTS / 'metabolic', OUTPUTS / 'clinical'
CUT_A1C, CUT_CGM = 5.7, 110.0          # HbA1c elevated: >=5.7%; CGM elevated: average glucose >=110 mg/dL
FI_THRESHOLD = 10.0                   # fasting insulin threshold (µIU/mL) in panel d
MIN_BIN_N = 30                        # HbA1c values with fewer participants are not drawn in panel a
WINDOW_DAYS = (90, 90)                # dense CGM days before and after the HbA1c draw
GROUPS = ['both_normal', 'cgm_only', 'a1c_only', 'both_elevated']
GROUP_LABEL = {'both_normal': 'both normal', 'cgm_only': 'CGM elevated only', 'a1c_only': 'HbA1c elevated only', 'both_elevated': 'both high'}
GROUP_LINES = {'both_normal': ['Both', 'normal'], 'cgm_only': ['CGM', 'elevated only'], 'a1c_only': ['HbA1c', 'elevated only'],
               'both_elevated': ['Both', 'elevated']}
GROUP_COL = {'both_normal': sd.BLUE_RAMP[2], 'cgm_only': sd.BLUE_RAMP[5], 'a1c_only': sd.BLUE_RAMP[3], 'both_elevated': sd.BLUE_RAMP[6]}
TILE_ORDER = [('both_normal', 'cgm_only'), ('a1c_only', 'both_elevated')]   # rows: HbA1c <5.7%, >=5.7%; columns: <110, >=110
C_CGM, C_A1C, C_FIT = sd.CATEGORICAL[0], sd.CATEGORICAL[2], sd.INK
MARKERS = [('homa_ir', 'HOMA-IR'), ('fasting_insulin', 'Fasting insulin')]
SCOPES = [('all_primary', 'Whole cohort'), ('hba1c_normal_lt57', 'HbA1c <5.7%')]
CONTRAST_MARKERS = [('fasting_insulin', 'Fasting insulin'), ('hscrp', 'hsCRP')]
RANGE_LABEL = {'all': 'All', 'lt_5_7': 'Normal', '5_7_to_6_4': 'Prediabetes range'}
# The layout starts below an empty band; the viewBox begins at CROP_TOP so the band is not shown.
Y_TOP = 71.5
CROP_TOP = 275 / (sd.PRINT_MM / 25.4 * 96 / sd.W * (2400 / sd.W))


# ============================================================================= source data
def make_source_data():
    t = pd.read_csv(MP / 'insulin_markers/hba1c_mean_cgm_table.csv')
    t = t.loc[t.n >= MIN_BIN_N].reset_index(drop=True)
    sl = pd.read_csv(MP / 'hba1c_within_range_slopes/within_range_slopes.csv')
    a = pd.concat([
        pd.DataFrame(dict(element='HbA1c value', hba1c_pct=t.hba1c, n_participants=t.n, average_glucose_mg_dl=t.mean_cgm,
                          ci_low=t.ci_low, ci_high=t.ci_high)),
        pd.DataFrame(dict(element='fit', hba1c_range=sl.range, n_participants=sl.n, hba1c_min=sl.hba1c_min, hba1c_max=sl.hba1c_max,
                          intercept_mg_dl=sl.intercept, slope_per_point=sl.slope_per_point, slope_per_0_1=sl.slope_per_0_1,
                          slope_per_0_1_ci_low=sl.slope_per_0_1_ci_low, slope_per_0_1_ci_high=sl.slope_per_0_1_ci_high))],
        ignore_index=True)
    a = a[['element', 'hba1c_pct', 'hba1c_range', 'n_participants', 'average_glucose_mg_dl', 'ci_low', 'ci_high', 'hba1c_min', 'hba1c_max',
           'intercept_mg_dl', 'slope_per_point', 'slope_per_0_1', 'slope_per_0_1_ci_low', 'slope_per_0_1_ci_high']]
    r = pd.read_csv(MP / 'hba1c_range_resolution/within_range_correlations.csv')
    r = r.loc[r.window == 'symmetric_90d']
    b = r[['range', 'hba1c_min', 'hba1c_max', 'n', 'pearson_r', 'ci_low', 'ci_high', 'hba1c_sd']].rename(
        columns={'range': 'hba1c_range', 'n': 'n_participants'})
    u = pd.read_csv(MP / 'hba1c_unadjusted_r2/unadjusted_r2.csv')
    rows = []
    for sc, _ in SCOPES:
        for mk, _ in MARKERS:
            x = sd.pick(u, outcome=mk, scope=sc)
            for q, col in (('R2 average glucose', 'r2_cgm'), ('R2 HbA1c', 'r2_hba1c'),
                           ('R2 difference, average glucose minus HbA1c', 'r2_diff_cgm_minus_hba1c')):
                rows.append(dict(scope=sc, marker=mk, n_participants=int(x.n), quantity=q, estimate=x[col],
                                 ci_low=x[f'{col}_ci_low'], ci_high=x[f'{col}_ci_high']))
    res = json.loads((CL / 'hba1c_resolution.json').read_text())
    nn = res['nested_normal']
    for q, k in (('R2 added by average glucose beyond HbA1c', 'avg_beyond_a1c'), ('R2 added by HbA1c beyond average glucose', 'a1c_beyond_avg')):
        rows.append(dict(scope='hba1c_normal_lt57', marker='fasting_insulin', n_participants=int(res['n_normal']), quantity=q,
                         estimate=nn[k][0], ci_low=nn[k][1], ci_high=nn[k][2]))
    c = pd.DataFrame(rows)
    hs = pd.read_csv(CL / 'markers/discordance_high_share.csv')
    hs = hs.loc[hs.marker == 'fasting_insulin'].set_index('group')
    gc = pd.read_csv(CL / 'discordance_groups/group_counts.csv').set_index('group')
    d = pd.DataFrame([dict(group=g, n_with_fasting_insulin=int(hs.loc[GROUP_LABEL[g], 'n']),
                           n_fasting_insulin_ge_10=int(hs.loc[GROUP_LABEL[g], 'k']), pct_fasting_insulin_ge_10=hs.loc[GROUP_LABEL[g], 'pct'],
                           wilson_ci_low=hs.loc[GROUP_LABEL[g], 'lo'], wilson_ci_high=hs.loc[GROUP_LABEL[g], 'hi'],
                           n_primary_cohort=int(gc.loc[g, 'n']), pct_of_primary_cohort=gc.loc[g, 'pct_of_primary']) for g in GROUPS])
    dm = pd.read_csv(CL / 'markers/discordance_markers.csv').set_index('marker')
    e = pd.DataFrame([dict(marker=mk, n_cgm_elevated_only=int(dm.loc[mk, 'n_CGM elevated only']),
                           n_hba1c_elevated_only=int(dm.loc[mk, 'n_HbA1c elevated only']), pct_difference=dm.loc[mk, 'cgm_vs_a1c_only_pct'],
                           ci_low=dm.loc[mk, 'lo'], ci_high=dm.loc[mk, 'hi'], p_value=dm.loc[mk, 'p']) for mk, _ in CONTRAST_MARKERS])
    return [source_data.write('fig5a_hba1c_glucose', a), source_data.write('fig5b_correlations', b),
            source_data.write('fig5c_variance_explained', c), source_data.write('fig5d_groups', d),
            source_data.write('fig5e_group_contrast', e)]


# ============================================================================= panels
def slope_label(r):
    return (f'{r.slope_per_0_1:.2f} mg/dL per 0.1 point', f'(95% CI {r.slope_per_0_1_ci_low:.2f}–{r.slope_per_0_1_ci_high:.2f})')


def panel_a(p, a):
    T = a[a.element == 'HbA1c value']
    s = a[a.element == 'fit'].set_index('hba1c_range')
    xlo, xhi = float(T.hba1c_pct.min()), float(T.hba1c_pct.max())
    segs = {}
    for k, r in s.iterrows():
        a0, a1 = max(float(r.hba1c_min), xlo), min(float(r.hba1c_max), xhi)
        segs[k] = (a0, a1, r.intercept_mg_dl + r.slope_per_point * a0, r.intercept_mg_dl + r.slope_per_point * a1)
    lo = min(float(T.ci_low.min()), min(v[2] for v in segs.values()))
    hi = max(float(T.ci_high.max()), max(v[3] for v in segs.values()))
    x = sd.Linear(xlo - 0.12, xhi + 0.12, p.px0, p.px1)
    y = sd.Linear(np.floor(lo / 5) * 5 - 3, np.ceil(hi / 5) * 5 + 3, p.py1, p.py0)
    yt = list(range(int(np.floor(lo / 5) * 5), int(np.ceil(hi / 5) * 5) + 1, 5))
    xt = [round(v, 1) for v in np.arange(np.ceil(xlo * 5) / 5, xhi + 1e-9, 0.2)]
    out = [sd.axis_left(p, y, yt, 'Average CGM glucose (mg/dL)', fmt_fn=lambda v: f'{v:.0f}'),
           sd.axis_bottom(p, x, xt, 'HbA1c (%)', fmt_fn=lambda v: f'{v:.1f}')]
    out.append(sd.line(p.px0, y(CUT_CGM), p.px1, y(CUT_CGM), stroke=sd.RULE, stroke_width=0.9, stroke_dasharray='3 2'))
    out.append(sd.text(p.px1 - 2, y(CUT_CGM) + 11, f'{CUT_CGM:g} mg/dL (CGM elevated)', size=S, fill=sd.INK_MUTED, anchor='end'))
    out.append(sd.line(x(CUT_A1C), p.py0 + 12, x(CUT_A1C), p.py1, stroke=sd.RULE, stroke_width=1.0))
    out.append(sd.text(x(CUT_A1C), p.py0 + 8, f'HbA1c {CUT_A1C:.1f}%', size=S, fill=sd.INK_MUTED, anchor='middle'))
    for r in T.itertuples():
        out.append(sd.line(x(r.hba1c_pct), y(r.ci_low), x(r.hba1c_pct), y(r.ci_high), stroke=C_CGM, stroke_width=1.2))
        out.append(sd.circle(x(r.hba1c_pct), y(r.average_glucose_mg_dl), 2.8, fill=C_CGM, stroke=sd.SURFACE, **{'stroke-width': 0.8}))
    for k, (a0, a1, g0, g1) in segs.items():
        out.append(sd.line(x(a0), y(g0), x(a1), y(g1), stroke=C_FIT, stroke_width=1.8, **{'stroke-linecap': 'round'}))
    rl, rh = s.loc['lt_5_7'], s.loc['5_7_to_6_4']
    l1, l2 = slope_label(rl)
    am = (segs['lt_5_7'][0] + segs['lt_5_7'][1]) / 2 + 0.1
    yl = y(rl.intercept_mg_dl + rl.slope_per_point * am) + 30
    for i, (t, w) in enumerate(((f'HbA1c <{CUT_A1C:.1f}%: {l1}', '600'), (l2, None))):
        out.append(sd.text(x(am), yl + 10 * i, t, size=7.8, fill=C_FIT, anchor='middle', weight=w))
    h1, h2 = slope_label(rh)
    ah = segs['5_7_to_6_4'][0] + 0.45
    yh = y(rh.intercept_mg_dl + rh.slope_per_point * ah) - 34
    lab = (f'HbA1c {CUT_A1C:.1f}–{rh.hba1c_max:.1f}%: {h1}', h2)
    wv = max(sd.tw(lab[0], 7.8, '600'), sd.tw(lab[1], 7.8))
    xe = x(ah) + 10
    out.append(sd.rect(xe - wv - 3, yh - 9, wv + 6, 22, fill=sd.SURFACE, opacity=0.92))
    for i, (t, w) in enumerate(((lab[0], '600'), (lab[1], None))):
        out.append(sd.text(xe, yh + 10 * i, t, size=7.8, fill=C_FIT, anchor='end', weight=w))
    out.append(sd.text(p.px0 + 4, p.py0 + 8, 'Points: average CGM glucose per HbA1c value (95% CI)', size=S, fill=sd.INK_SECONDARY))
    out.append(sd.text(p.px0 + 4, p.py0 + 18, 'Lines: least-squares fit within each HbA1c range, all participants in the range', size=S,
                       fill=sd.INK_SECONDARY))
    return sd.group(out)


def panel_b(p, b):
    sym = b.set_index('hba1c_range')
    order = ['all', 'lt_5_7', '5_7_to_6_4']
    rng = {'all': f"{sym.loc['all', 'hba1c_min']:.1f}–{sym.loc['all', 'hba1c_max']:.1f}%", 'lt_5_7': f'<{CUT_A1C:.1f}%',
           '5_7_to_6_4': f"{CUT_A1C:.1f}–{sym.loc['5_7_to_6_4', 'hba1c_max']:.1f}%"}
    top = float(np.ceil(sym.ci_high.max() * 10) / 10 + 0.05)
    y = sd.Linear(0, top, p.py1, p.py0)
    xb = sd.Band(order, p.px0, p.px1, pad=0.34)
    out = [sd.axis_left(p, y, [t for t in (0, 0.1, 0.2, 0.3, 0.4, 0.5) if t <= top], 'Pearson r with average glucose',
                        fmt_fn=lambda v: f'{v:.1f}' if v else '0'),
           sd.line(p.px0, p.py1, p.px1, p.py1, stroke=sd.RULE)]
    cols = {'all': sd.BLUE_RAMP[3], 'lt_5_7': sd.BLUE_RAMP[5], '5_7_to_6_4': sd.BLUE_RAMP[4]}
    for k in order:
        r = sym.loc[k]
        out.append(sd.rect(xb(k), y(r.pearson_r), xb.bandwidth, p.py1 - y(r.pearson_r), fill=cols[k], rx=1.5))
        cx = xb.center(k)
        out.append(sd.line(cx, y(r.ci_low), cx, y(r.ci_high), stroke=sd.INK, stroke_width=1.0))
        for yy in (r.ci_low, r.ci_high):
            out.append(sd.line(cx - 4, y(yy), cx + 4, y(yy), stroke=sd.INK, stroke_width=1.0))
        out.append(sd.text(cx, y(r.ci_high) - 5, f'{r.pearson_r:.2f}', size=8.4, fill=sd.INK, anchor='middle', weight='600'))
        yy = p.py1 + 12
        for part in (RANGE_LABEL[k].split(' ') if k == '5_7_to_6_4' else [RANGE_LABEL[k]]):
            out.append(sd.text(cx, yy, part, size=7.8, fill=sd.INK_SECONDARY, anchor='middle'))
            yy += 10
        yy = p.py1 + 32
        out.append(sd.text(cx, yy, rng[k], size=S, fill=sd.INK_MUTED, anchor='middle'))
        out.append(sd.text(cx, yy + 10, f'n={int(r.n_participants):,}', size=S, fill=sd.INK_MUTED, anchor='middle'))
        out.append(sd.text(cx, yy + 20, f'SD {r.hba1c_sd:.2f}', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.text(p.px0 + 2, p.py0 - 6, '95% CI: participant bootstrap', size=S, fill=sd.INK_MUTED))
    lo, hi = sym.loc['lt_5_7'], sym.loc['5_7_to_6_4']
    note = (f'A narrower range lowers r by itself, but {CUT_A1C:.1f}–{hi.hba1c_max:.1f}% is narrower still (SD {hi.hba1c_sd:.2f} vs '
            f'{lo.hba1c_sd:.2f}) and keeps r = {hi.pearson_r:.2f}.')
    yy = p.py1 + 70
    for part in sd.wrap(note, p.w - 10, 7.6):
        out.append(sd.text(p.x0 + 4, yy, part, size=7.6, fill=sd.INK_SECONDARY))
        yy += 9.6
    return sd.group(out), yy


def panel_c(x0, y0, w, c):
    q = lambda sc, mk, name: sd.pick(c, scope=sc, marker=mk, quantity=name)  # noqa: E731
    rows = [(sc, slab, mk, mlab) for sc, slab in SCOPES for mk, mlab in MARKERS]
    out = []
    lab_r, ax0, ax1 = x0 + 104, x0 + 114, x0 + 232
    top = max(100 * q(sc, mk, 'R2 average glucose').ci_high for sc, _, mk, _ in rows)
    lim = (0.0, float(math.ceil(top / 5) * 5))
    x = sd.Linear(lim[0], lim[1], ax0, ax1)
    kx = x0 + 14
    for lab, col in (('Average glucose', C_CGM), ('HbA1c', C_A1C)):
        out.append(sd.dot(kx + 3, y0 - 2.6, col, True))
        out.append(sd.text(kx + 10, y0, lab, size=S, fill=sd.INK_SECONDARY))
        kx += 10 + sd.tw(lab, S) + 14
    out.append(sd.text(634, y0 + 14, 'Average glucose − HbA1c,', size=S, fill=sd.INK_SECONDARY, anchor='end', weight='600'))
    out.append(sd.text(634, y0 + 24, 'points (95% CI)', size=S, fill=sd.INK_SECONDARY, anchor='end', weight='600'))
    y = y0 + 22
    ytop = y + 4
    all_above, prev = True, None
    for sc, slab, mk, mlab in rows:
        cg, a1, df = q(sc, mk, 'R2 average glucose'), q(sc, mk, 'R2 HbA1c'), q(sc, mk, 'R2 difference, average glucose minus HbA1c')
        if sc != prev:
            y += 22 if prev else 0
            out.append(sd.text(x0 + 4, y, slab, size=S, fill=sd.INK_MUTED, weight='600'))
            prev = sc
            y += 2
            first = True
        y += 14 if first else 27
        first = False
        out.append(sd.text(lab_r, y, mlab, size=F['body'], fill=sd.INK, anchor='end', weight='600'))
        out.append(sd.text(lab_r, y + 10, f'n = {int(cg.n_participants):,}', size=S, fill=sd.INK_MUTED, anchor='end'))
        for dy, (r, col) in ((-1, (cg, C_CGM)), (7, (a1, C_A1C))):
            out.append(sd.line(x(100 * r.ci_low), y + dy, x(100 * r.ci_high), y + dy, stroke=col, stroke_width=1.4))
            out.append(sd.dot(x(100 * r.estimate), y + dy, col, True, r=2.8))
        d, dl, dh = 100 * df.estimate, 100 * df.ci_low, 100 * df.ci_high
        ex = sd.excl0(dl, dh)
        all_above = all_above and ex and d > 0
        out.append(sd.text(634, y, f'{100 * cg.estimate:.1f}% vs {100 * a1.estimate:.1f}%', size=S, fill=sd.INK_SECONDARY, anchor='end'))
        out.append(sd.text(634, y + 10, f'{sd.sg(d)} ({sd.sg(dl)} to {sd.sg(dh)})' + ('' if ex else ', not detected'), size=S,
                           fill=sd.INK if ex else sd.INK_MUTED, anchor='end', weight='700' if ex else None))
    ybot = y + 16
    under = []
    for t in range(0, int(lim[1]) + 1, 5):
        under.append(sd.line(x(t), ytop, x(t), ybot, stroke=sd.INK_MUTED if t == 0 else sd.GRID, stroke_width=0.8))
        out.append(sd.line(x(t), ybot, x(t), ybot + 3, stroke=sd.RULE))
        out.append(sd.text(x(t), ybot + 11, f'{t}', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.line(x(lim[0]), ybot, x(lim[1]), ybot, stroke=sd.RULE))
    out.append(sd.text((ax0 + ax1) / 2, ybot + 22, 'R², % (95% CI)', size=S, fill=sd.INK_SECONDARY, anchor='middle'))
    yend = ybot + 26
    if all_above:
        els, yend = sd.wrap_lines(x0 + 14, ybot + 42, 'Average glucose explained more than HbA1c in every row, whole cohort and within '
                                  'the normal HbA1c range (95% CI of the difference above 0).', w - 18, size=F['body'], fill=sd.INK, weight='600')
        out += els
    # R² added beyond the other measure (fasting insulin, HbA1c <5.7%)
    avg, a1c = (q('hba1c_normal_lt57', 'fasting_insulin', 'R2 added by average glucose beyond HbA1c'),
                q('hba1c_normal_lt57', 'fasting_insulin', 'R2 added by HbA1c beyond average glucose'))
    lines = [(f'R² added beyond the other measure, fasting insulin, HbA1c <5.7% (n = {int(avg.n_participants):,}):', sd.INK_SECONDARY, '600'),
             (f'average glucose beyond HbA1c {sd.sg(100 * avg.estimate)} points ({100 * avg.ci_low:.1f}–{100 * avg.ci_high:.1f})', sd.INK, None),
             (f'HbA1c beyond average glucose {sd.sg(100 * a1c.estimate, 2)} points ({100 * a1c.ci_low:.2f}–{100 * a1c.ci_high:.2f})', sd.INK, None)]
    y = yend + 8
    for t, col, wt in lines:
        out.append(sd.text(298, y, t, size=S, fill=col, weight=wt))
        y += 9.6
    return under + out, y


def panel_d(p, d):
    g = d.set_index('group')
    top = float(np.ceil(g.wilson_ci_high.max() / 10) * 10)
    y = sd.Linear(0, top, p.py1, p.py0)
    xb = sd.Band(GROUPS, p.px0, p.px1, pad=0.36)
    out = [sd.axis_left(p, y, list(range(0, int(top) + 1, 10)), f'% with fasting insulin ≥{FI_THRESHOLD:g}', fmt_fn=lambda v: f'{v:.0f}'),
           sd.line(p.px0, p.py1, p.px1, p.py1, stroke=sd.RULE)]
    for k in GROUPS:
        r = g.loc[k]
        out.append(sd.rect(xb(k), y(r.pct_fasting_insulin_ge_10), xb.bandwidth, p.py1 - y(r.pct_fasting_insulin_ge_10), fill=GROUP_COL[k], rx=1.5))
        cx = xb.center(k)
        out += sd.vbar_ci(cx, y, r.wilson_ci_low, r.wilson_ci_high)
        out.append(sd.text(cx, y(r.wilson_ci_high) - 5, f'{r.pct_fasting_insulin_ge_10:.1f}%', size=F['body'], fill=sd.INK, anchor='middle', weight='600'))
        for i, part in enumerate(GROUP_LINES[k]):
            out.append(sd.text(cx, p.py1 + 11 + 10 * i, part, size=S, fill=sd.INK_SECONDARY, anchor='middle'))
        out.append(sd.text(cx, p.py1 + 33, f'{int(r.n_fasting_insulin_ge_10):,}/{int(r.n_with_fasting_insulin):,}', size=S,
                           fill=sd.INK_MUTED, anchor='middle'))
    return out


def group_tile(x0, y0, d):
    g = d.set_index('group')
    out = [sd.text(x0, y0, f'All participants by group (primary cohort, n = {int(g.n_primary_cohort.sum()):,})', size=S, fill=sd.INK_SECONDARY)]
    rl, cw, ch = 58, 100, 24
    cx0, yh = x0 + rl, y0 + 12
    for j, t in enumerate((f'Average glucose <{CUT_CGM:g}', f'≥{CUT_CGM:g} mg/dL')):
        out.append(sd.text(cx0 + cw * (j + 0.5), yh, t, size=S, fill=sd.INK_MUTED, anchor='middle'))
    for i, (rowlab, gs) in enumerate(zip((f'HbA1c <{CUT_A1C:.1f}%', f'≥{CUT_A1C:.1f}%'), TILE_ORDER)):
        yy = yh + 4 + ch * i
        out.append(sd.text(cx0 - 6, yy + ch / 2 + 2.5, rowlab, size=S, fill=sd.INK_MUTED, anchor='end'))
        for j, k in enumerate(gs):
            r = g.loc[k]
            xx = cx0 + cw * j
            out.append(sd.rect(xx, yy, cw - 2, ch - 2, fill=sd.WASH, stroke=sd.RULE, stroke_width=0.8, rx=2))
            out.append(sd.rect(xx + 4, yy + 4.5, 6, 6, fill=GROUP_COL[k], rx=1))
            out.append(sd.text(xx + 13, yy + 10, ' '.join(GROUP_LINES[k]), size=S, fill=sd.INK, weight='600'))
            out.append(sd.text(xx + 13, yy + 19.2, f'{int(r.n_primary_cohort):,} ({r.pct_of_primary_cohort:.1f}%)', size=S, fill=sd.INK_SECONDARY))
    return out, yh + 4 + 2 * ch


def panel_e(x0, y0, w, e, row_step=24, axis_label='% higher, unadjusted (95% CI)'):
    out = []
    y = y0
    lab_r, ax0, ax1 = x0 + 92, x0 + 102, x0 + w - 50
    vals = [v for r in e.itertuples() for v in (r.ci_low, r.ci_high)]
    lim = (float(math.floor(min(0, min(vals)) / 10) * 10), float(math.ceil(max(vals) / 10) * 10))
    step = 20 if lim[1] - lim[0] > 60 else 10
    lim = (float(math.floor(lim[0] / step) * step), float(math.ceil(lim[1] / step) * step))
    x = sd.Linear(lim[0], lim[1], ax0, ax1)
    ytop = y + 2
    y += 14
    for mk, mlab in CONTRAST_MARKERS:
        r = sd.pick(e, marker=mk)
        out.append(sd.text(lab_r, y, mlab, size=F['body'], fill=sd.INK, anchor='end', weight='600'))
        out.append(sd.text(lab_r, y + 10, f'{int(r.n_cgm_elevated_only):,} vs {int(r.n_hba1c_elevated_only):,}', size=S, fill=sd.INK_MUTED, anchor='end'))
        out += sd.ci_mark(x, lim, y + 2, r.pct_difference, r.ci_low, r.ci_high, sd.INK)
        ex = sd.excl0(r.ci_low, r.ci_high)
        lab = sd.pct(r.pct_difference) + ('' if ex else f', P = {r.p_value:.2f}')
        out.append(sd.text(x(r.ci_high) + 5, y + 5, lab, size=S, fill=sd.INK if ex else sd.INK_MUTED, weight='700' if ex else None))
        y += row_step
    ybot = y - 10
    under = []
    for t in range(int(lim[0]), int(lim[1]) + 1, step):
        under.append(sd.line(x(t), ytop, x(t), ybot, stroke=sd.INK_MUTED if t == 0 else sd.GRID, stroke_width=0.9 if t == 0 else 0.8))
        out.append(sd.line(x(t), ybot, x(t), ybot + 3, stroke=sd.RULE))
        out.append(sd.text(x(t), ybot + 11, sd.sg(t, 0) if t else '0', size=S, fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.line(x(lim[0]), ybot, x(lim[1]), ybot, stroke=sd.RULE))
    out.append(sd.text((ax0 + ax1) / 2, ybot + 22, axis_label, size=S, fill=sd.INK_SECONDARY, anchor='middle'))
    return under + out, ybot + 26


# ============================================================================= figure
def draw(out_dir):
    a, b, c = source_data.read('fig5a_hba1c_glucose'), source_data.read('fig5b_correlations'), source_data.read('fig5c_variance_explained')
    d, e = source_data.read('fig5d_groups'), source_data.read('fig5e_group_contrast')
    fit_hi = a[a.element == 'fit'].set_index('hba1c_range').loc['5_7_to_6_4', 'hba1c_max']
    ya = Y_TOP + 4
    pa = sd.Panel(6, ya + 14, 628, 206, ml=46, mr=10, mt=4, mb=34)
    g_a = panel_a(pa, a)
    yb = ya + 14 + 206 + 12
    pb = sd.Panel(6, yb + 20, 262, 176, ml=40, mr=4, mt=12, mb=0)
    g_b, yb_end = panel_b(pb, b)
    g_c, yc_end = panel_c(284, yb + 30, 350, c)
    yd = max(yb_end, yc_end) + 8
    pdd = sd.Panel(6, yd + 26, 340, 150, ml=44, mr=6, mt=10, mb=40)
    g_d = panel_d(pdd, d)
    g_tile, yt_end = group_tile(372, yd + 22, d)
    ye = yt_end + 12
    g_e, ye_end = panel_e(372, ye + 16, 262, e)
    g_de = g_tile + [sd.line(366, ye - 4, 634, ye - 4, stroke=sd.GRID), sd.plabel(366, ye, 'e', 'CGM elevated only vs HbA1c elevated only')] + g_e
    g = d.set_index('group')
    k, n = int(g.loc['cgm_only', 'n_primary_cohort']), int(g.loc['both_normal', 'n_primary_cohort'] + g.loc['cgm_only', 'n_primary_cohort'])
    y_share = max(pdd.y0 + pdd.h, ye_end) + 8
    share_txt = f'{100 * k / n:.1f}% of HbA1c-normal participants ({k:,} of {n:,}) had average glucose ≥{CUT_CGM:g} mg/dL.'
    body = [sd.plabel(6, ya, 'a', f'Average glucose rises less steeply with HbA1c below {CUT_A1C:.1f}% than within {CUT_A1C:.1f}–{fit_hi:.1f}%'), g_a,
            sd.line(6, yb - 6, 634, yb - 6, stroke=sd.GRID),
            sd.plabel(6, yb, 'b', 'Correlation within HbA1c ranges'), g_b,
            sd.plabel(284, yb, 'c', 'Insulin resistance explained: average glucose vs HbA1c'),
            sd.text(298, yb + 20, 'Plain R², one measure at a time, no covariates; same participants', size=S, fill=sd.INK_MUTED),
            *g_c,
            sd.line(276, yb, 276, max(yb_end, yc_end) - 4, stroke=sd.GRID),
            sd.line(6, yd - 4, 634, yd - 4, stroke=sd.GRID),
            sd.plabel(6, yd, 'd', 'HbA1c and CGM groups'),
            sd.text(20, yd + 20, f'Participants with fasting insulin; % with fasting insulin ≥{FI_THRESHOLD:g} µIU/mL, 95% CI: Wilson', size=S,
                    fill=sd.INK_MUTED),
            *g_d, *g_de,
            sd.line(360, yd + 18, 360, ye_end - 4, stroke=sd.GRID),
            sd.text(6, y_share, share_txt, size=F['body'], fill=sd.INK, weight='600')]
    footer = (f'HbA1c rounded half-up to 0.1; average glucose over dense CGM days {WINDOW_DAYS[0]} days before to {WINDOW_DAYS[1]} days after '
              f'the draw. CGM elevated: average glucose ≥{CUT_CGM:g} mg/dL; HbA1c elevated: ≥{CUT_A1C:.1f}%. Filled points: 95% CI excludes 0; '
              'open: 95% CI includes 0 (not detected).')
    h = y_share + 10 + 6 + 9.4 * len(sd.wrap(footer, sd.W - 12, S))
    svg = sd.render_svg(h, [], body, TITLE, footer=footer)
    svg = (svg.replace(f'viewBox="0 0 {sd.fmt(sd.W)} {sd.fmt(h)}"', f'viewBox="0 {CROP_TOP:.6f} {sd.fmt(sd.W)} {h - CROP_TOP:.6f}"')
              .replace(f'height="{sd.fmt(sd.PRINT_MM * h / sd.W)}mm"', f'height="{sd.PRINT_MM * (h - CROP_TOP) / sd.W:.6f}mm"'))
    return sd.write(svg, out_dir, NAME)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, FIGURES_OUT)
