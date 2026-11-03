"""Gap-aware primitives for equally spaced 15-minute CGM grids.

Only bounded, wholly missing runs of at most two cells may be interpolated.
Two missing cells means 30 minutes of missing grid support (45 minutes between
the two observed endpoints). Longer runs remain entirely missing. Dynamic
lagged pairs require finite support throughout the interval. Same-clock MODD
is a separate comparison of observed clock cells on consecutive calendar days
and explicitly requests endpoint-only matching.
"""
from __future__ import annotations

import numpy as np

GRID_FEATURE_VERSION = "gap-aware-15m"


def interpolate_short_gaps(values: np.ndarray, limit: int = 2) -> np.ndarray:
    """Fill a whole bounded missing run iff its length <= limit; no edge fill."""
    if int(limit) != limit or limit < 0:
        raise ValueError("Interpolation limit must be a nonnegative integer")
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 1:
        raise ValueError("CGM grid must be one-dimensional")
    result[~np.isfinite(result)] = np.nan
    missing = np.isnan(result)
    boundaries = np.diff(np.r_[False, missing, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
        if start > 0 and stop < len(result) and stop-start <= limit:
            result[start:stop] = np.interp(np.arange(start,stop),[start-1,stop],[result[start-1],result[stop]])
    return result


def lag_pair_mask(values: np.ndarray, lag: int, *, require_contiguous: bool = True) -> np.ndarray:
    """Pairs at exactly lag cells, optionally restricted to a complete interval."""
    if int(lag) != lag or lag < 1:
        raise ValueError("Lag must be a positive integer number of grid cells")
    lag = int(lag)
    x = np.asarray(values,dtype=float)
    if x.ndim != 1:
        raise ValueError("CGM grid must be one-dimensional")
    if len(x) <= lag:
        return np.zeros(0,dtype=bool)
    finite = np.isfinite(x)
    if not require_contiguous:
        return finite[:-lag] & finite[lag:]
    missing_prefix = np.r_[0,np.cumsum(~finite)]
    return (missing_prefix[lag+1:] - missing_prefix[:-lag-1]) == 0


def paired_difference(values: np.ndarray, lag: int, absolute: bool = False,
                      *, require_contiguous: bool = True) -> np.ndarray:
    """Dynamic differences never cross unresolved gaps; MODD opts out explicitly."""
    values=np.asarray(values,dtype=float)
    mask=lag_pair_mask(values,lag,require_contiguous=require_contiguous)
    lag=int(lag)
    difference=values[lag:][mask]-values[:-lag][mask] if len(values)>lag else np.array([],dtype=float)
    return np.abs(difference) if absolute else difference


def autocorrelation(values: np.ndarray, lag: int, min_pairs: int = 100) -> float:
    """Pearson lag correlation over complete intervals, with >=100 pairs by default."""
    x=np.asarray(values,dtype=float)
    mask=lag_pair_mask(x,lag)
    lag=int(lag)
    if int(mask.sum()) < min_pairs:
        return float("nan")
    left,right=x[:-lag][mask],x[lag:][mask]
    if np.std(left)==0 or np.std(right)==0:
        return float("nan")
    return float(np.corrcoef(left,right)[0,1])
