"""Loads the canonical lab/CGM panel and applies the published eligibility gates.

Loads the canonical panel written by analysis/canonical_panel/build_panel.py,
verifies required structural gates (one row per participant, finite anchor
values, the observed-grid stage complete), enforces the no-diabetes-range
cohort rules (HbA1c in [3, 20), at least 14 dense days, no-diabetes flag
consistent with anchor and ever-high-lab HbA1c), and applies the standard
HbA1c rounding convention. Also provides a helper to write metadata describing
an analysis run alongside its outputs.

Inputs: OUTPUTS/canonical_panel/canonical_lab_cgm_panel.csv (and the
preceding-90-day variant), both produced by
analysis/canonical_panel/build_panel.py.

Usage: imported by analysis producers; not run directly.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .cross_validation import validate_publication_panel
from .lab_normalization import round_a1c

from .paths import OUTPUTS  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
PUBLICATION = OUTPUTS
DEFAULT_PANEL = PUBLICATION / 'canonical_panel/canonical_lab_cgm_panel.csv'
DEFAULT_PRECEDING = PUBLICATION / 'canonical_panel/canonical_lab_cgm_panel_preceding90.csv'


def boolean_flag(values, name):
    result = values.astype(str).str.lower().map({'true': True, 'false': False})
    if result.isna().any():
        raise ValueError(f'Invalid canonical boolean flag: {name}')
    return result.astype(bool)


def load_panel(path=DEFAULT_PANEL, cohort='no_diabetes_range', required=(), expected_window=None):
    path = Path(path)
    meta = validate_publication_panel(path)
    if meta.get('dense_grid_complete') is not True:
        raise ValueError('Canonical observed-grid stage must be complete')
    if expected_window is not None and meta.get('window') != expected_window:
        raise ValueError(f'Expected canonical {expected_window} window')
    data = pd.read_csv(path, dtype={'USER_ID': str}, low_memory=False)
    columns = {'USER_ID', 'a1c', 'mean_glucose', 'n_dense_days', 'no_diabetes_range', *required}
    if not columns <= set(data):
        raise ValueError(f'Missing canonical columns: {sorted(columns-set(data))}')
    if data.USER_ID.isna().any() or data.USER_ID.duplicated().any():
        raise ValueError('Canonical panel must contain one identified row per participant')
    flag = boolean_flag(data.no_diabetes_range, 'no_diabetes_range')
    for name in ('a1c', 'mean_glucose', 'n_dense_days'):
        data[name] = pd.to_numeric(data[name], errors='coerce')
        if not np.isfinite(data[name]).all():
            raise ValueError(f'Invalid canonical {name}')
    data['a1c'] = round_a1c(data.a1c)
    if not data.a1c.between(3, 20).all() or not data.n_dense_days.ge(14).all():
        raise ValueError('Canonical anchor or coverage gate failed')
    if (flag & data.a1c.ge(6.5)).any():
        raise ValueError('No-diabetes flag conflicts with anchor HbA1c')
    if 'diabetes_range_ever' in data:
        ever = boolean_flag(data.diabetes_range_ever, 'diabetes_range_ever')
        if (flag & ever).any():
            raise ValueError('No-diabetes flag conflicts with accepted raw ever-high labs')
    data['no_diabetes_range'] = flag
    if cohort == 'no_diabetes_range':
        data = data.loc[flag].copy()
    elif cohort != 'all_lab_linked':
        raise ValueError(f'Unknown cohort: {cohort}')
    data.attrs['panel_metadata'] = meta
    return data.reset_index(drop=True)


def write_analysis_metadata(path, *, script, panels, outputs, details):
    metadata = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'panels': {name: {'filename': Path(source).name, **validate_publication_panel(source)}
                   for name, source in panels.items()},
        **details,
    }
    Path(path).write_text(json.dumps(metadata, indent=2, default=str, allow_nan=False)+'\n')
