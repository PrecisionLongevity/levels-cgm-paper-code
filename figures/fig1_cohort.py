"""Fig. 1: the cohort and its linked data streams.

  a  schematic of one participant's observation streams and the ±90-day laboratory pairing window
  b  overlap of CGM, logged behavior and laboratory results (area-proportional Euler diagram), and the flow from consent
     to the primary cohort
  c  age decade and recorded sex of consenting participants
  d  ADA range of each participant's highest accepted HbA1c or fasting glucose, overall, by age decade and by sex
  e  distributions of laboratory values (primary cohort) and logged behavior across participants

Inputs (analysis outputs)
  cohort/cohort_summary.json                                           a, c
  cohort/stream_overlap_regions/summary.json, cohort_funnel.csv        b
  cohort/glycemic_composition/glycemic_composition.csv                 d
  cohort/multimodal_distributions/, cohort/lab_distributions_extended/ e (distribution_summary.csv, distribution_histograms.csv)
Source data
  source_data/fig1ab_streams.csv, fig1b_funnel.csv, fig1c_age_sex.csv, fig1d_glycemic_range.csv,
  fig1e_distributions.csv, fig1e_histograms.csv
  In fig1e_histograms.csv, bins holding 1-10 participants are merged with a neighbouring bin (n_bins_merged > 1). The
  figure is drawn from the unmerged bins in the analysis outputs when they are available; drawn from source_data alone,
  merged bins are shown at their average height per original bin. Likewise the number of participants in their teens
  (1-10) is not released; without the analysis outputs the panel c note reads 'fewer than 11'.
Outputs
  fig1_cohort.svg / .png

Usage
  python figures/fig1_cohort.py source-data
  python figures/fig1_cohort.py draw
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cgmpaper import cli, source_data  # noqa: E402
from cgmpaper import svgfig as sd  # noqa: E402
from cgmpaper.paths import FIGURES_OUT, OUTPUTS  # noqa: E402
from cgmpaper.svgfig import F, tw, wrap  # noqa: E402

NAME = 'fig1_cohort'
TITLE = 'Fig. 1 | Cohort and linked data streams'
CO = OUTPUTS / 'cohort'
HIST_DIRS = {'multimodal_distributions': ('a1c', 'fasting_insulin', 'triglycerides', 'daily_steps', 'sleep_hours', 'exercise_days_per_week'),
             'lab_distributions_extended': ('ldl_cholesterol', 'apo_b', 'creatinine')}
C_CGM, C_BEH, C_LAB = sd.CATEGORICAL[0], sd.CATEGORICAL[2], sd.CATEGORICAL[3]
RANGES = [('normal', 'Normal', sd.CATEGORICAL[0]), ('prediabetes range', 'Prediabetes range', sd.CATEGORICAL[1]),
          ('diabetes range', 'Diabetes range', sd.CATEGORICAL[3])]
DECADES = [(20, '20s'), (30, '30s'), (40, '40s'), (50, '50s'), (60, '60s'), (70, '70s'), (80, '80s')]
AGE_LABELS = ['20s', '30s', '40s', '50s', '60s', '70s', '80s+']
SEX_SEGMENTS = [('female', 'Female', sd.CATEGORICAL[0]), ('male', 'Male', sd.CATEGORICAL[1]), ('not_recorded', 'No recorded sex', sd.RULE)]
STAGE_LABELS = {'consented': ['Consenting participants'],
                'with_cgm': ['With CGM readings'],
                'lab_cgm_sample': ['Lab–CGM sample', 'HbA1c with ≥14 dense CGM days within ±90 d'],
                'primary_no_diabetes_range_labs': ['Primary cohort', 'no observed diabetes-range laboratory value']}
# (measure, label, ticks, tick format, decimals of the mean, colour)
DIST_LAB = [('a1c', 'HbA1c (%)', [4, 5, 6], '{:.0f}', 2, sd.CATEGORICAL[3]),
            ('fasting_insulin', 'Fasting insulin (µIU/mL)', [0, 10, 20, 30, 40], '{:.0f}', 1, sd.CATEGORICAL[3]),
            ('triglycerides', 'Triglycerides (mg/dL)', [0, 100, 200, 300, 400], '{:.0f}', 0, sd.CATEGORICAL[3]),
            ('ldl_cholesterol', 'LDL cholesterol (mg/dL)', [0, 50, 100, 150, 200, 250], '{:.0f}', 0, sd.CATEGORICAL[3]),
            ('apo_b', 'ApoB (mg/dL)', [50, 100, 150, 200], '{:.0f}', 0, sd.CATEGORICAL[3]),
            ('creatinine', 'Creatinine (mg/dL)', [0.5, 1.0, 1.5], '{:.1f}', 2, sd.CATEGORICAL[3])]
DIST_BEH = [('daily_steps', 'Average daily steps', [0, 5000, 10000, 15000, 20000, 25000], 'k', 0, sd.CATEGORICAL[2]),
            ('sleep_hours', 'Sleep per night (h)', [4, 6, 8, 10], '{:.0f}', 1, sd.CATEGORICAL[2]),
            ('exercise_days_per_week', 'Exercise days per week', [0, 1, 2, 3, 4, 5, 6, 7], '{:.0f}', 1, sd.CATEGORICAL[2])]


# ============================================================================= source data
def make_source_data():
    cs = json.loads((CO / 'cohort_summary.json').read_text())
    so = json.loads((CO / 'stream_overlap_regions/summary.json').read_text())
    q = [('consenting participants', so['total']), ('median consented interval (days)', cs['consent_median'])]
    q += [(f'participants with {k}', v) for k, v in so['set_marginals'].items()]
    q += [(f'participants with {k.replace("_any", "")} records', so['stream_coverage'][k]) for k in ('food_any', 'exercise_any', 'steps_any', 'sleep_any', 'weight_any')]
    q += [(f'pairwise overlap {k}', v) for k, v in so['pairwise_intersections'].items() if k != 'all_three']
    q += [(f'region {k}', v) for k, v in so['regions'].items()]
    ab = pd.DataFrame(q, columns=['quantity', 'value'])
    fn = pd.read_csv(CO / 'stream_overlap_regions/cohort_funnel.csv').rename(columns={'n': 'n_participants'})
    c = pd.DataFrame([dict(age_decade=lab, female=cs['ages'][lab]['female'], male=cs['ages'][lab]['male'],
                           all_participants=sum(cs['ages'][lab].values())) for lab in AGE_LABELS])
    c = pd.concat([c, pd.DataFrame([dict(age_decade='not plotted: no plausible age decade', all_participants=cs['age_missing']),
                                    dict(age_decade='not plotted: in their teens', all_participants=None,
                                         note='1-10 participants; not released')])], ignore_index=True)
    c[['female', 'male', 'all_participants']] = c[['female', 'male', 'all_participants']].astype('Int64')
    comp = pd.read_csv(CO / 'glycemic_composition/glycemic_composition.csv')
    comp = comp.loc[comp.classification == 'ada_hba1c_or_fpg']   # primary rule: HbA1c or fasting glucose
    groups = ['all'] + [f'age_bucket={a}' for a, _ in DECADES] + ['sex=female', 'sex=male']
    d = comp.loc[comp.group.isin(groups), ['group', 'range', 'pct', 'denominator']].rename(
        columns={'pct': 'pct_of_group', 'denominator': 'n_group'})
    d['group'] = d.group.map(lambda g: {'all': 'all with labs', 'sex=female': 'female', 'sex=male': 'male'}.get(g, g.replace('age_bucket=', 'age ') + 's'))
    summ, hist = [], []
    for sub, keys in HIST_DIRS.items():
        s = pd.read_csv(CO / sub / 'distribution_summary.csv').set_index('measure')
        h = pd.read_csv(CO / sub / 'distribution_histograms.csv')
        for k in keys:
            r = s.loc[k]
            summ.append(dict(measure=k, population=r.population, n_participants=int(r.n), mean=r['mean'], q25=r.q25, q75=r.q75,
                             display_lo=r.display_lo, display_hi=r.display_hi))
            m = source_data.merge_small_bins(h.loc[h.measure == k])
            m.insert(0, 'measure', k)
            hist.append(m.rename(columns={'count': 'n_participants'}))
    return [source_data.write('fig1ab_streams', ab), source_data.write('fig1b_funnel', fn), source_data.write('fig1c_age_sex', c),
            source_data.write('fig1d_glycemic_range', d), source_data.write('fig1e_distributions', pd.DataFrame(summ)),
            source_data.write('fig1e_histograms', pd.concat(hist, ignore_index=True))]


def restricted_inputs():
    """Unmerged histogram bins and the teens count from the analysis outputs, when available."""
    try:
        h = pd.concat([pd.read_csv(CO / sub / 'distribution_histograms.csv') for sub in HIST_DIRS], ignore_index=True)
        teens = int(json.loads((CO / 'cohort_summary.json').read_text())['age_off_range'])
    except FileNotFoundError:
        print('analysis outputs not found: panel c and e drawn from source_data (bins of 1-10 participants merged)')
        return None, None
    return h.rename(columns={'count': 'n_participants'}).assign(n_bins_merged=1), teens


# ============================================================================= panel a
def phone(x0, y0, w, h):
    """Stylised app frame."""
    sx0, sy0, sw, sh = x0 + 5, y0 + 11, w - 10, h - 20
    out = [sd.rect(x0, y0, w, h, fill=sd.WASH, rx=9, stroke=sd.INK_SECONDARY, stroke_width=1.1),
           sd.rect(sx0, sy0, sw, sh, fill=sd.SURFACE, rx=3.5, stroke=sd.RULE),
           sd.rect(x0 + w / 2 - 6, y0 + 5, 12, 2.2, fill=sd.RULE, rx=1.1)]
    cx0, cy = sx0 + 4, sy0 + sh * 0.30
    cw, amp = sw - 8, sh * 0.14
    pts = [(0.00, 0.2), (0.12, -0.1), (0.24, 0.55), (0.36, 1.0), (0.50, 0.25), (0.62, -0.25), (0.74, 0.45), (0.86, 0.05), (1.00, -0.2)]
    steps = [f'M {sd.fmt(cx0)} {sd.fmt(cy - pts[0][1] * amp)}']
    for i in range(1, len(pts)):
        t0, v0 = pts[i - 1]
        t1, v1 = pts[i]
        mx = cx0 + (t0 + t1) / 2 * cw
        steps.append(f'C {sd.fmt(mx)} {sd.fmt(cy - v0 * amp)} {sd.fmt(mx)} {sd.fmt(cy - v1 * amp)} {sd.fmt(cx0 + t1 * cw)} {sd.fmt(cy - v1 * amp)}')
    out.append(sd.path(' '.join(steps), stroke=sd.CATEGORICAL[0], stroke_width=1.6))
    out.append(sd.circle(cx0 + 0.36 * cw, cy - amp, 2.4, fill=sd.CATEGORICAL[2], stroke=sd.SURFACE, stroke_width=0.8))
    for i in range(3):
        ry = sy0 + sh * 0.58 + i * (sh * 0.13)
        out.append(sd.circle(sx0 + 7, ry, 2.6, fill=sd.CATEGORICAL[2], fill_opacity=0.85))
        out.append(sd.rect(sx0 + 13, ry - 2.2, sw * 0.52, 2.4, fill=sd.RULE, rx=1.2))
    return sd.group(out)


def panel_schematic(x0, y0, w, consent_median):
    """One participant's streams on a shared timeline (illustrative, not data)."""
    fs = F['body']
    ph_w, ph_h = 42, 74
    ph_x, ph_y = x0 + 10, y0 + 4
    out = [phone(ph_x, ph_y, ph_w, ph_h),
           sd.text(ph_x + ph_w / 2, ph_y + ph_h + 12, 'Levels app', size=fs, fill=sd.INK_SECONDARY, anchor='middle', weight='600')]
    t0, t1 = x0 + 196, x0 + w - 6
    tl = t1 - t0
    x = lambda t: t0 + t * tl  # noqa: E731
    lanes = [('CGM sensor wear', sd.CATEGORICAL[0], 'segments', [(0.02, 0.15), (0.28, 0.41), (0.54, 0.62), (0.79, 0.93)]),
             ('Food logs', sd.CATEGORICAL[2], 'ticks', [0.03, 0.05, 0.08, 0.11, 0.14, 0.29, 0.31, 0.34, 0.38, 0.40, 0.55, 0.57, 0.60, 0.80, 0.83, 0.86, 0.90]),
             ('Activity & steps (HealthKit)', sd.CATEGORICAL[1], 'ticks',
              [0.02, 0.07, 0.12, 0.17, 0.22, 0.27, 0.32, 0.37, 0.42, 0.47, 0.52, 0.57, 0.62, 0.67, 0.72, 0.77, 0.82, 0.87, 0.92]),
             ('Sleep (HealthKit)', sd.BLUE_RAMP[2], 'segments', [(0.02, 0.18), (0.28, 0.44), (0.54, 0.66), (0.78, 0.94)]),
             ('Weight', sd.CATEGORICAL[4], 'diamonds', [0.06, 0.33, 0.58, 0.85]),
             ('Laboratory results', sd.CATEGORICAL[3], 'triangles', [0.18, 0.58, 0.88])]
    lane_y0, dy = y0 + 12, 12.0
    win_lo, win_hi = 0.58 - 0.085, 0.58 + 0.085
    out.append(sd.rect(x(win_lo), lane_y0 - 3, x(win_hi) - x(win_lo), dy * len(lanes) + 2, fill=sd.BLUE_RAMP[0], rx=3))
    out.append(sd.text((x(win_lo) + x(win_hi)) / 2, lane_y0 - 6, '±90 d laboratory pairing window', size=fs, fill=sd.INK_MUTED, anchor='middle'))
    for i, (label, color, kind, items) in enumerate(lanes):
        ly = lane_y0 + i * dy + dy / 2
        out.append(sd.text(t0 - 8, ly + 2.8, label, size=fs, fill=sd.INK_SECONDARY, anchor='end'))
        out.append(sd.line(t0, ly, t1, ly, stroke=sd.GRID))
        if kind == 'segments':
            for a, b in items:
                out.append(sd.rect(x(a), ly - 3.4, x(b) - x(a), 6.8, fill=color, rx=3.4))
        elif kind == 'ticks':
            for t in items:
                out.append(sd.line(x(t), ly - 3.6, x(t), ly + 3.6, stroke=color, stroke_width=1.6))
        elif kind == 'diamonds':
            for t in items:
                cx = x(t)
                out.append(sd.path(f'M {sd.fmt(cx)} {sd.fmt(ly - 4.2)} L {sd.fmt(cx + 3.9)} {sd.fmt(ly)} L {sd.fmt(cx)} {sd.fmt(ly + 4.2)} '
                                   f'L {sd.fmt(cx - 3.9)} {sd.fmt(ly)} Z', fill=color, stroke=None))
        else:
            for t in items:
                cx = x(t)
                out.append(sd.path(f'M {sd.fmt(cx)} {sd.fmt(ly - 4.4)} L {sd.fmt(cx + 4.2)} {sd.fmt(ly + 3.6)} L {sd.fmt(cx - 4.2)} {sd.fmt(ly + 3.6)} Z',
                                   fill=color, stroke=None))
    ax_y = lane_y0 + len(lanes) * dy + 5
    out += [sd.arrow_h(t0, ax_y, tl),
            sd.text(t1, ax_y + 12, f'participant-local time; median consented interval {consent_median:.0f} days; streams start, stop and '
                    "overlap at the participant's discretion", size=fs, fill=sd.INK_MUTED, anchor='end')]
    return sd.group(out), ax_y + 12 - y0


# ============================================================================= panel b: Euler diagram and funnel
REGION_MASK = {'cgm_only': (1, 0, 0), 'behavior_only': (0, 1, 0), 'labs_only': (0, 0, 1), 'cgm_behavior': (1, 1, 0),
               'cgm_labs': (1, 0, 1), 'behavior_labs': (0, 1, 1), 'all_three': (1, 1, 1)}


def lens_area(d, r1, r2):
    a1 = np.arccos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = np.arccos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    return r1 * r1 * (a1 - np.sin(2 * a1) / 2) + r2 * r2 * (a2 - np.sin(2 * a2) / 2)


def solve_distance(target, r1, r2):
    """Centre distance at which two circles overlap by the target area (bisection)."""
    lo, hi = abs(r1 - r2) + 1e-9, r1 + r2 - 1e-9
    for _ in range(100):
        mid = (lo + hi) / 2
        if lens_area(mid, r1, r2) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def region_masks(centers, radii, step, pad=10):
    lo = min(c[0] - r for c, r in zip(centers, radii)) - pad
    hi = max(c[0] + r for c, r in zip(centers, radii)) + pad
    lo_y = min(c[1] - r for c, r in zip(centers, radii)) - pad
    hi_y = max(c[1] + r for c, r in zip(centers, radii)) + pad
    X, Y = np.meshgrid(np.arange(lo, hi, step), np.arange(lo_y, hi_y, step))
    ins = [(X - c[0]) ** 2 + (Y - c[1]) ** 2 <= r * r for c, r in zip(centers, radii)]
    masks = {k: np.logical_and.reduce([ins[i] if f else ~ins[i] for i, f in enumerate(v)]) for k, v in REGION_MASK.items()}
    return masks, X, Y


def euler_layout(m, pw, regions, r_behavior):
    """Circle areas proportional to the three set sizes; centres fitted to the seven region areas (Nelder-Mead from
    deterministic starts, the first from exact pairwise overlaps)."""
    from scipy.optimize import minimize
    k = np.pi * r_behavior ** 2 / m['behavior']   # canvas units² per participant
    rB, rC, rL = r_behavior, np.sqrt(m['cgm'] * k / np.pi), np.sqrt(m['labs'] * k / np.pi)
    dCB, dCL, dBL = (solve_distance(pw['cgm_behavior'] * k, rC, rB), solve_distance(pw['cgm_labs'] * k, rC, rL),
                     solve_distance(pw['behavior_labs'] * k, rB, rL))
    lx = (dBL ** 2 - dCL ** 2 + dCB ** 2) / (2 * dCB)
    ly = np.sqrt(max(dBL ** 2 - lx ** 2, 0.0))

    def loss(p):
        masks, _, _ = region_masks([(p[0], 0.0), (0.0, 0.0), (p[1], p[2])], [rC, rB, rL], 1.0)
        return sum((float(masks[r].sum()) / k - regions[r]) ** 2 / max(regions[r], 300) for r in REGION_MASK)
    best = None
    for x0 in ([dCB, lx, ly], [dCB, 40.0, 20.0], [dCB, 45.0, 0.0], [dCB, 30.0, -30.0]):
        res = minimize(loss, x0, method='Nelder-Mead', options=dict(xatol=0.05, fatol=0.01, maxiter=400))
        if best is None or res.fun < best.fun:
            best = res
    p = best.x
    return dict(radii=dict(cgm=rC, behavior=rB, labs=rL), centers=dict(cgm=(p[0], 0.0), behavior=(0.0, 0.0), labs=(p[1], p[2])))


def anchor_points(lay, step=0.25):
    """Most interior point of each region (distance transform); None if the region is too small to draw."""
    from scipy.ndimage import distance_transform_edt
    cs, rs = lay['centers'], lay['radii']
    masks, X, Y = region_masks([cs['cgm'], cs['behavior'], cs['labs']], [rs['cgm'], rs['behavior'], rs['labs']], step)
    pts = {}
    for r, mk in masks.items():
        if mk.sum() * step * step < 2.0:
            pts[r] = None
            continue
        dist = distance_transform_edt(mk)
        i = np.unravel_index(np.argmax(dist), dist.shape)
        pts[r] = (float(X[i]), float(Y[i]), float(dist[i] * step))
    return pts


def panel_euler(x0, y0, w, h, q):
    m = {k: int(q[f'participants with {k}']) for k in ('cgm', 'behavior', 'labs')}
    pw = {k: int(q[f'pairwise overlap {k}']) for k in ('cgm_behavior', 'cgm_labs', 'behavior_labs')}
    R = {k: int(q[f'region {k}']) for k in list(REGION_MASK) + ['none']}
    lay = euler_layout(m, pw, R, r_behavior=66.0)
    cs, rs = lay['centers'], lay['radii']
    ox, oy = x0 + 118 + rs['behavior'], y0 + 58 + rs['behavior']
    P = lambda c: (ox + c[0], oy + c[1])  # noqa: E731
    out = [sd.rect(x0 + 2, y0 + 2, w - 4, h - 4, fill=sd.WASH, rx=6, stroke=sd.RULE, stroke_width=0.9)]
    out.append(sd.text(x0 + 10, y0 + 15, f"All {int(q['consenting participants']):,} consenting participants", size=F['body'], fill=sd.INK, weight='600'))
    out.append(sd.legend_row(x0 + 10, y0 + 29, [(f"CGM {m['cgm']:,}", C_CGM), (f"Logged behavior {m['behavior']:,}", C_BEH),
                                                (f"Laboratory results {m['labs']:,}", C_LAB)], size=F['body']))
    cov = {k: int(q[f'participants with {k} records']) for k in ('food', 'exercise', 'steps', 'sleep', 'weight')}
    sub = (f"Logged behavior = any food ({cov['food']:,}), exercise ({cov['exercise']:,}), step ({cov['steps']:,}) "
           f"or sleep ({cov['sleep']:,}) record; weight ({cov['weight']:,}) is not in the set.")
    for j, t in enumerate(wrap(sub, w - 22, F['small'])):
        out.append(sd.text(x0 + 10, y0 + 42 + 9.6 * j, t, size=F['small'], fill=sd.INK_SECONDARY))
    for key, col, op in (('behavior', C_BEH, 0.16), ('cgm', C_CGM, 0.16), ('labs', C_LAB, 0.22)):
        cx, cy = P(cs[key])
        out.append(sd.circle(cx, cy, rs[key], fill=col, fill_opacity=op, stroke=col, stroke_width=1.4))
    pts = anchor_points(lay)
    cx_c, cy_c = P(cs['cgm'])
    out.append(sd.text(cx_c + 22, cy_c - 40, f"{R['cgm_behavior']:,}", size=10.0, fill=sd.INK, anchor='middle', weight='700'))
    out.append(sd.text(cx_c + 22, cy_c - 29, 'CGM + behavior', size=F['small'], fill=sd.INK_SECONDARY, anchor='middle'))
    out.append(sd.text(cx_c + 22, cy_c - 19, 'no labs', size=F['small'], fill=sd.INK_SECONDARY, anchor='middle'))
    tx, ty, _ = pts['all_three']
    out.append(sd.text(ox + tx, oy + ty - 1, f"{R['all_three']:,}", size=9.4, fill=sd.INK, anchor='middle', weight='700'))
    out.append(sd.text(ox + tx, oy + ty + 9, 'all three', size=F['small'], fill=sd.INK, anchor='middle'))
    right_x = x0 + w - 8
    leaders = [('cgm_only', 'CGM only', 'right', -40), ('behavior_only', 'behavior only', 'left', -30),
               ('behavior_labs', 'behavior + labs, no CGM', 'left', 42), ('labs_only', 'labs only', 'left', 70)]
    for reg, lab, side, dyl in leaders:
        px, py = ox + pts[reg][0], oy + pts[reg][1]
        if side == 'left':
            lx_, anchor = x0 + 10, 'start'
            tx_end = lx_ + max(tw(f'{R[reg]:,}', F['body'], '700'), tw(lab, F['small'])) + 3
        else:
            lx_, anchor = right_x, 'end'
            tx_end = lx_ - max(tw(f'{R[reg]:,}', F['body'], '700'), tw(lab, F['small'])) - 3
        ly_ = oy + dyl
        out.append(sd.line(tx_end, ly_ - 3, px, py, stroke=sd.INK_MUTED, stroke_width=0.7))
        out.append(sd.circle(px, py, 1.4, fill=sd.INK))
        out.append(sd.text(lx_, ly_ - 5, f'{R[reg]:,}', size=F['body'], fill=sd.INK, anchor=anchor, weight='700'))
        out.append(sd.text(lx_, ly_ + 4.5, lab, size=F['small'], fill=sd.INK_SECONDARY, anchor=anchor))
    yy = y0 + h - 30
    out.append(sd.text(right_x, yy, f"{R['none']:,} with none of the three", size=F['small'], fill=sd.INK_SECONDARY, anchor='end'))
    out.append(sd.text(right_x, yy + 10, f"{R['cgm_labs']:,} CGM + labs, no behavior (too few to draw)", size=F['small'],
                       fill=sd.INK_SECONDARY, anchor='end'))
    return sd.group(out)


def panel_funnel(x0, y0, w, fn):
    total = int(fn.n_participants.iloc[0])
    x = sd.Linear(0, total, x0, x0 + w - 44)
    out, y = [], y0
    for i, (stage, n) in enumerate(zip(fn.stage, fn.n_participants)):
        lines = STAGE_LABELS[stage]
        for j, s in enumerate(lines):
            out.append(sd.text(x0, y + 8 + 9.6 * j, s, size=F['body'], fill=sd.INK if j == 0 else sd.INK_SECONDARY, weight='600' if j == 0 else None))
        yb = y + 12 + 9.6 * (len(lines) - 1)
        col = sd.CATEGORICAL[0] if stage != 'consented' else sd.BLUE_RAMP[2]
        out.append(sd.rect(x0, yb, x(n) - x0, 11, fill=col if stage != 'primary_no_diabetes_range_labs' else sd.INK, rx=1.5))
        out.append(sd.text(x(n) + 5, yb + 8.5, f'{n:,}', size=F['body'], fill=sd.INK, weight='700'))
        if i > 0:
            out.append(sd.text(x(n) + 5 + tw(f'{n:,}', F['body'], '700') + 4, yb + 8.5, f'({100 * n / total:.0f}%)', size=F['small'], fill=sd.INK_MUTED))
        y = yb + 11 + 8
        if i < len(fn) - 1:
            out.append(sd.path(f'M {sd.fmt(x0 + 6)} {sd.fmt(y - 6)} L {sd.fmt(x0 + 6)} {sd.fmt(y - 1)}', stroke=sd.INK_MUTED, stroke_width=1.0))
    return sd.group(out), y


# ============================================================================= panels c and d
def panel_age_sex(x0, y0, w, h, c, total, teens):
    p = sd.Panel(x0, y0, w, h, ml=40, mr=6, mt=26, mb=40)
    ages = c.set_index('age_decade')
    counts = {lab: dict(female=int(ages.loc[lab, 'female']), male=int(ages.loc[lab, 'male']),
                        not_recorded=int(ages.loc[lab, 'all_participants'] - ages.loc[lab, 'female'] - ages.loc[lab, 'male'])) for lab in AGE_LABELS}
    pct = {lab: {s: counts[lab][s] / total * 100 for s, _, _ in SEX_SEGMENTS} for lab in AGE_LABELS}
    top = max(sum(pct[lab].values()) for lab in AGE_LABELS) * 1.18
    x = sd.Band(AGE_LABELS, p.px0, p.px1, pad=0.26)
    y = sd.Linear(0, top, p.py1, p.py0)
    out = [sd.axis_left(p, y, [v for v in sd.nice_ticks(0, top, 4) if v <= top], '% of consenting participants', fmt_fn=lambda v: f'{v:.0f}'),
           sd.axis_bottom(p, x, AGE_LABELS, 'Age decade', fmt_fn=lambda v: v)]
    for lab in AGE_LABELS:
        cur = 0.0
        for s, _, col in SEX_SEGMENTS:
            v = pct[lab][s]
            y1, y0v = y(cur), y(cur + v)
            hh = max(0.0, y1 - y0v - 1.2)
            if hh > 0.3:
                out.append(sd.rect(x(lab), y0v, x.bandwidth, hh, fill=col, rx=1.2))
            if hh >= 11 and s != 'not_recorded':
                out.append(sd.text(x.center(lab), (y0v + y1) / 2 + 2.8, f'{counts[lab][s]:,}', size=F['small'],
                                   fill=sd.SURFACE if col == sd.CATEGORICAL[0] else sd.INK, anchor='middle', weight='600'))
            cur += v
        out.append(sd.text(x.center(lab), y(cur) - 4, f'{sum(counts[lab].values()):,}', size=F['small'], fill=sd.INK, anchor='middle', weight='700'))
    out.append(sd.legend_row(p.px0, y0 + 20, [(n, c_) for _, n, c_ in SEX_SEGMENTS], size=F['small']))
    teens_txt = f'{teens:,}' if teens is not None else 'fewer than 11'
    out.append(sd.text(p.px0, p.py1 + 38, f"Not plotted: {int(ages.loc['not plotted: no plausible age decade', 'all_participants']):,} "
                                          f'without a plausible decade; {teens_txt} in the 10s.', size=F['small'], fill=sd.INK_MUTED))
    return sd.group(out)


def panel_composition(x0, y0, w, d, bh=9.2, gap=1.6):
    groups = [('all with labs', 'All with labs')] + [(f'age {a}s', lab) for a, lab in DECADES] + [('female', 'Female'), ('male', 'Male')]
    lab_w = 58
    bx0, bx1 = x0 + lab_w, x0 + w - 62
    x = sd.Linear(0, 100, bx0, bx1)
    out = [sd.legend_row(x0, y0 + 8, [(lab, c) for _, lab, c in RANGES], size=F['small'])]
    y = y0 + 24
    out.append(sd.text(bx1 + 30, y - 3, 'diab.', size=F['small'], fill=sd.INK_MUTED, anchor='end'))
    out.append(sd.text(x0 + w, y - 3, 'n', size=F['small'], fill=sd.INK_MUTED, anchor='end'))
    y += 2
    for gi, (g, lab) in enumerate(groups):
        if gi in (1, 8):
            out.append(sd.text(x0, y + 7, 'By age decade' if gi == 1 else 'By sex', size=F['small'], fill=sd.INK_MUTED, weight='600'))
            y += 10
        r = d.loc[d.group == g].set_index('range')
        out.append(sd.text(bx0 - 5, y + bh / 2 + 2.8, lab, size=F['small'], fill=sd.INK, anchor='end', weight='700' if g == 'all with labs' else None))
        cur = 0.0
        for k, _, col in RANGES:
            pc = float(r.loc[k, 'pct_of_group'])
            ww = x(cur + pc) - x(cur)
            out.append(sd.rect(x(cur), y, max(0.0, ww - 0.8), bh, fill=col))
            if ww >= 26 and k != 'diabetes range':
                out.append(sd.text(x(cur) + ww / 2, y + bh / 2 + 2.8, f'{pc:.1f}%', size=F['small'],
                                   fill=sd.SURFACE if col == sd.CATEGORICAL[0] else sd.INK, anchor='middle', weight='600'))
            cur += pc
        out.append(sd.text(bx1 + 30, y + bh / 2 + 2.8, f"{r.loc['diabetes range', 'pct_of_group']:.1f}%", size=F['small'],
                           fill=sd.CATEGORICAL[3], anchor='end', weight='600'))
        out.append(sd.text(x0 + w, y + bh / 2 + 2.8, f"{int(r.n_group.iloc[0]):,}", size=F['small'], fill=sd.INK_MUTED, anchor='end'))
        y += bh + (5 if g == 'all with labs' else gap)
    ax = sd.Panel(x0, y0, w, y - y0 + 2, ml=lab_w, mr=x0 + w - bx1, mt=0, mb=0)
    out.append(sd.axis_bottom(ax, x, [0, 25, 50, 75, 100], '% with an HbA1c or fasting glucose', fmt_fn=lambda v: f'{v:.0f}'))
    return sd.group(out), y + 32


# ============================================================================= panel e: distributions
def fmt_tick(v, f):
    return (f'{v / 1000:.0f}k' if v else '0') if f == 'k' else f.format(v)


def hist_cell(x0, y0, w, h, key, label, ticks, tfmt, dg, col, summ, hist):
    st = summ.loc[key]
    hb = hist.loc[hist.measure == key]
    lo, hi = float(st['display_lo']), float(st['display_hi'])
    q = sd.Panel(x0, y0, w, h, ml=4, mr=4, mt=24, mb=14)
    x = sd.Linear(lo, hi, q.px0, q.px1)
    share = hb['n_participants'].to_numpy(float) / float(st['n_participants']) / hb['n_bins_merged'].to_numpy(float)
    y = sd.Linear(0, share.max() * 1.10, q.py1, q.py0)
    out = [sd.rect(x(st['q25']), q.py0, x(st['q75']) - x(st['q25']), q.ph, fill=sd.BLUE_RAMP[0])]
    for v, a, b in zip(share, hb.bin_lo, hb.bin_hi):
        if v > 0:
            out.append(sd.rect(x(a), y(v), max(0.2, x(b) - x(a) - 0.3), q.py1 - y(v), fill=col))
    out.append(sd.line(x(st['mean']), q.py0, x(st['mean']), q.py1, stroke=sd.INK, stroke_width=1.1))
    out.append(sd.line(q.px0, q.py1, q.px1, q.py1, stroke=sd.RULE))
    for v in ticks:
        out.append(sd.line(x(v), q.py1, x(v), q.py1 + 2.5, stroke=sd.RULE))
        out.append(sd.text(x(v), q.py1 + 10.5, fmt_tick(v, tfmt), size=F['small'], fill=sd.INK_MUTED, anchor='middle'))
    out.append(sd.text(q.px0, y0 + 8, label, size=F['small'], fill=sd.INK, weight='600'))
    out.append(sd.text(q.px0, y0 + 18, f"mean {st['mean']:,.{dg}f}; n = {int(st['n_participants']):,}", size=F['small'], fill=sd.INK_SECONDARY))
    return sd.group(out)


def panel_distributions(x0, y0, w, summ, hist, n_primary):
    """Row 1: five primary-cohort analytes; row 2: creatinine, three behavior streams and the key."""
    ncol, gap, ch = 5, 12.0, 66.0
    cw = (w - (ncol - 1) * gap) / ncol
    cx = lambda i: x0 + i * (cw + gap)  # noqa: E731
    out, y = [], y0
    out.append(sd.text(x0, y + 8, f'Laboratory: primary cohort ({n_primary:,}); value nearest the anchor HbA1c draw (within ±7 days); '
                                  'n per analyte shown in each panel', size=F['small'], fill=sd.INK_MUTED, weight='600'))
    y += 12
    for i, spec in enumerate(DIST_LAB[:5]):
        out.append(hist_cell(cx(i), y, cw, ch, *spec, summ, hist))
    y += ch + 6
    out.append(sd.text(cx(0), y + 8, 'Primary cohort, continued', size=F['small'], fill=sd.INK_MUTED, weight='600'))
    out.append(sd.text(cx(1), y + 8, 'Behavior: consenting participants with ≥3 recorded days (exercise: ≥28-day span)',
                       size=F['small'], fill=sd.INK_MUTED, weight='600'))
    y += 12
    for i, spec in enumerate([DIST_LAB[5]] + DIST_BEH):
        out.append(hist_cell(cx(i), y, cw, ch, *spec, summ, hist))
    kx, ky = cx(4) + 4, y + 10
    out.append(sd.line(kx, ky - 3, kx + 14, ky - 3, stroke=sd.INK, stroke_width=1.1))
    out.append(sd.text(kx + 19, ky, 'line: mean', size=F['small'], fill=sd.INK_SECONDARY))
    out.append(sd.rect(kx, ky + 6, 14, 7, fill=sd.BLUE_RAMP[0]))
    out.append(sd.text(kx + 19, ky + 12.5, 'shading: IQR', size=F['small'], fill=sd.INK_SECONDARY))
    out.append(sd.text(kx, ky + 26, 'bars: share of', size=F['small'], fill=sd.INK_SECONDARY))
    out.append(sd.text(kx, ky + 35.5, 'participants', size=F['small'], fill=sd.INK_SECONDARY))
    out.append(sd.text(kx, ky + 48, 'median (IQR): legend', size=F['small'], fill=sd.INK_SECONDARY))
    return sd.group(out), y + ch + 6


# ============================================================================= figure
def draw(out_dir):
    q = source_data.values('fig1ab_streams')
    fn, c, d = source_data.read('fig1b_funnel'), source_data.read('fig1c_age_sex'), source_data.read('fig1d_glycemic_range')
    summ, hist = source_data.read('fig1e_distributions').set_index('measure'), source_data.read('fig1e_histograms')
    full_hist, teens = restricted_inputs()
    if full_hist is not None:
        hist = full_hist
    ya = sd.TOP + 2
    g_a, ha = panel_schematic(6, ya + 12, 628, q['median consented interval (days)'])
    yb = ya + 12 + ha + 10
    eh = 212
    g_b = panel_euler(6, yb + 14, 392, eh, q)
    g_f, yf_end = panel_funnel(414, yb + 22, 220, fn)
    yc = max(yb + 14 + eh, yf_end) + 12
    hc = 168
    g_c = panel_age_sex(6, yc + 2, 300, hc, c, int(q['consenting participants']), teens)
    g_d, yd_end = panel_composition(330, yc + 18, 304, d)
    ye = max(yc + 2 + hc, yd_end) + 10
    n_primary = int(fn.loc[fn.stage == 'primary_no_diabetes_range_labs', 'n_participants'].iloc[0])
    g_e, ye_end = panel_distributions(6, ye + 14, 628, summ, hist, n_primary)
    body = [sd.plabel(6, ya, 'a', 'One participant, several observation streams (schematic)'), g_a,
            sd.plabel(6, yb, 'b', 'Which records participants contributed'), g_b,
            sd.text(414, yb + 9, 'From consent to the analysis cohort', size=F['ptitle'], fill=sd.INK, weight='600'), g_f,
            sd.plabel(6, yc, 'c', 'Age and recorded sex'), g_c,
            sd.plabel(330, yc, 'd', 'Laboratory glycemic range'), g_d,
            sd.plabel(6, ye, 'e', 'Laboratory and behavior streams across participants'), g_e]
    return sd.write(sd.render_svg(ye_end + 2 + 6, [], body, TITLE), out_dir, NAME)


if __name__ == '__main__':
    cli.run(__doc__, make_source_data, draw, FIGURES_OUT)
