"""Training-only numeric preprocessing and participant-disjoint CV primitives.

Targets are never winsorized or standardized here: a held-out endpoint must retain
its prespecified physical/log scale. No target values are used by preprocessing.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold

CV_DEFINITION_VERSION = "participant-training-only"


def validate_publication_panel(path, require_rich=False):
    """Check that the canonical panel was built with the current definitions; return its metadata."""
    from .cgm_trace_features import FEATURE_VERSION
    from .lab_normalization import LAB_DEFINITION_VERSION
    from .cgm_grid_metrics import GRID_FEATURE_VERSION as CORE_GRID_FEATURE_VERSION
    from analysis.canonical_panel._canonical_grid import DENSE_DAY_VERSION, MAGE_VERSION
    path = Path(path)
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise ValueError(f"Canonical panel metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    required = {
        "canonical_panel_version": "publication",
        "lab_definition_version": LAB_DEFINITION_VERSION,
        "core_grid_feature_version": CORE_GRID_FEATURE_VERSION,
        "dense_day_count_version": DENSE_DAY_VERSION,
        "daily_mage_version": MAGE_VERSION,
    }
    if require_rich:
        required["trace_feature_version"] = FEATURE_VERSION
        required["rich_features_complete"] = True
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Canonical panel built with other definitions: {key} must be {expected!r}")
    return metadata


class TrainingOnlyNumeric(BaseEstimator, TransformerMixin):
    """Median imputation, optional clipping, and scaling learned on fit rows only.

    An entirely missing training column is retained as zero. Held-out observations
    cannot make such a column informative. Constant training columns are likewise
    set to zero. This also preserves column alignment across folds.
    """

    def __init__(self, lower=0.005, upper=0.995, standardize=True):
        self.lower = lower
        self.upper = upper
        self.standardize = standardize

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=float).copy()
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("Expected a nonempty two-dimensional training array")
        x[~np.isfinite(x)] = np.nan
        self.n_features_in_ = x.shape[1]
        self.empty_ = np.isnan(x).all(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median_ = np.nanmedian(x, axis=0)
            self.low_ = np.nanquantile(x, self.lower, axis=0)
            self.high_ = np.nanquantile(x, self.upper, axis=0)
        self.median_[self.empty_] = 0.0
        self.low_[self.empty_] = self.high_[self.empty_] = 0.0
        x = self._clip_impute(x)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.constant_ = self.scale_ == 0
        self.scale_[self.constant_] = 1.0
        return self

    def _clip_impute(self, x):
        x = np.clip(x, self.low_, self.high_)
        return np.where(np.isfinite(x), x, self.median_)

    def transform(self, X):
        x = np.asarray(X, dtype=float).copy()
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("Prediction columns differ from training columns")
        x[~np.isfinite(x)] = np.nan
        x = self._clip_impute(x)
        if self.standardize:
            x = (x - self.mean_) / self.scale_
            x[:, self.constant_] = 0.0
        return x


def participant_folds(user_ids, n_splits=5, random_state=42, strata=None):
    """Stable splits of sorted participant IDs, invariant to incoming row order.

    Repeated observations of a participant remain together. A classification
    stratum must be constant within participant; conflicting labels fail.
    """
    ids = np.asarray(user_ids).astype(str)
    users, inverse = np.unique(ids, return_inverse=True)
    if len(users) < n_splits:
        raise ValueError("Fewer participants than requested CV folds")
    if strata is None:
        splitter = KFold(n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(users)
    else:
        strata = np.asarray(strata)
        labels = []
        for i in range(len(users)):
            values = np.unique(strata[inverse == i])
            if len(values) != 1:
                raise ValueError("Classification label changes within participant")
            labels.append(values[0])
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(users, labels)
    return [
        (np.flatnonzero(np.isin(inverse, train)), np.flatnonzero(np.isin(inverse, test)))
        for train, test in splits
    ]


def linear_oof_predictions(X, y, folds):
    """Unclipped-target OOF OLS; preprocessing is fit separately in each fold."""
    x = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("CV endpoints must be finite on their prespecified scale")
    predictions = np.full(len(y), np.nan)
    for train, test in folds:
        if not x.shape[1]:
            predictions[test] = y[train].mean()
        else:
            transform = TrainingOnlyNumeric().fit(x[train])
            model = LinearRegression().fit(transform.transform(x[train]), y[train])
            predictions[test] = model.predict(transform.transform(x[test]))
    if not np.isfinite(predictions).all():
        raise ValueError("CV did not produce one finite prediction for every row")
    return predictions
