"""Read and write the source data tables in source_data/ (one tidy CSV per display item or panel).

Values are written at full precision so that a figure drawn from source_data/ is identical to one drawn from the
analysis outputs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cgmpaper.paths import SOURCE_DATA


def path(name: str) -> Path:
    return SOURCE_DATA / f'{name}.csv'


def write(name: str, df: pd.DataFrame) -> Path:
    """Write one source data table (no index) and return its path."""
    p = path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, lineterminator='\n')
    return p


def read(name: str) -> pd.DataFrame:
    """Read one source data table with exact float round-tripping."""
    p = path(name)
    if not p.is_file():
        raise SystemExit(f'{p} not found. Run the script with the source-data command first (requires the analysis outputs).')
    return pd.read_csv(p, float_precision='round_trip', keep_default_na=False, na_values=[''])


def values(name: str, key: str = 'quantity', value: str = 'value') -> dict:
    """Read a two-column (quantity, value) table as a dict."""
    df = read(name)
    return dict(zip(df[key], df[value]))


SMALL_COUNT_MAX = 10   # counts of 1-10 are not released in source data


def merge_small_bins(bins: pd.DataFrame, lo='bin_lo', hi='bin_hi', count='count') -> pd.DataFrame:
    """Merge histogram bins holding 1-10 participants into their neighbour on the side with more participants.

    Returns the merged bins with a column n_bins_merged (1 for bins left unchanged). Totals are preserved.
    """
    rows = [dict(bin_lo=float(r[lo]), bin_hi=float(r[hi]), count=int(r[count]), n_bins_merged=1) for _, r in bins.iterrows()]
    while True:
        small = [i for i, r in enumerate(rows) if 1 <= r['count'] <= SMALL_COUNT_MAX]
        if not small or len(rows) == 1:
            break
        i = small[0]
        left, right = sum(r['count'] for r in rows[:i]), sum(r['count'] for r in rows[i + 1:])
        j = i - 1 if (i == len(rows) - 1 or (i > 0 and left > right)) else i + 1
        a, b = sorted((i, j))
        rows[a:b + 1] = [dict(bin_lo=rows[a]['bin_lo'], bin_hi=rows[b]['bin_hi'], count=rows[a]['count'] + rows[b]['count'],
                              n_bins_merged=rows[a]['n_bins_merged'] + rows[b]['n_bins_merged'])]
    return pd.DataFrame(rows)
