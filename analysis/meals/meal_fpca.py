#!/usr/bin/env python3
"""Functional PCA basis for post-meal glucose trace shapes.

Requires complete observed traces (by default no interior gap filling; up to two
interior gap cells may optionally be linearly interpolated), subtracts BASELINE_PRE
from each trace, fits a PCA basis on a seeded random subsample of the centered
traces, and projects all retained traces onto that basis to obtain per-meal
component scores. Each component is oriented so that its largest absolute loading
is positive. An optional comparison against an external per-user score file is a
diagnostic only and never affects fitting or orientation.

Inputs:
  OUTPUTS/meals/meal_traces_observed.parquet - per-meal glucose grid traces with
    BASELINE_PRE, produced by analysis/meals/build_meals.py.

Outputs (OUTPUTS/meals/, per-meal/per-user; keep out of version control):
  meal_fpca_scores.parquet - per-meal component scores.
  meal_fpca_per_user.csv - per-user summary statistics of component scores.
  meal_fpca_components.npz - fitted PCA components, mean, explained variance.
  meal_fpca_metadata.json - run parameters and counts.
  meal_fpca_summary.md - human-readable summary.

Usage: python analysis/meals/meal_fpca.py [--traces PATH] [--output-dir PATH]
  [--components N] [--fit-subsample N] [--seed N] [--max-gap-cells {0,1,2}]
  [--historical-per-user PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cgmpaper.paths import OUTPUTS  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_TRACES = OUTPUTS / 'meals/meal_traces_observed.parquet'
DEFAULT_OUT = OUTPUTS / 'meals'
N_COMP = 10
FIT_SUBSAMPLE = 500_000
SEED = 20260825
DEFINITION_VERSION = "meal-pca-observed-cell"


def complete_traces(frame, max_gap_cells=0):
    """One common observed-trace completion rule, with deterministic row order."""
    grid_columns = sorted((c for c in frame if re.fullmatch(r"g\d+", c)), key=lambda c: int(c[1:]))
    if len(grid_columns)<4 or grid_columns != [f"g{i}" for i in range(len(grid_columns))]:
        raise ValueError("Trace grid must have consecutive g0..gN columns")
    if max_gap_cells not in (0, 1, 2):
        raise ValueError("Only zero, one, or two interior gap cells are supported")
    required = ["USER_ID", "MEAL_EVENT_ID", "BASELINE_PRE"] + grid_columns
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing trace columns: {missing}")
    if frame[["USER_ID", "MEAL_EVENT_ID"]].isna().any().any():
        raise ValueError("Trace identifiers must be present")
    if frame.duplicated(["USER_ID", "MEAL_EVENT_ID"]).any():
        raise ValueError("Duplicate participant/meal trace keys")
    frame = frame.sort_values(["USER_ID", "MEAL_EVENT_ID"]).reset_index(drop=True)
    grid = frame[grid_columns].to_numpy(dtype=float)
    if np.isinf(grid).any():
        raise ValueError("Infinite glucose values in trace input")
    filled = (pd.DataFrame(grid).interpolate(axis=1, method="linear",
        limit=max_gap_cells, limit_area="inside").to_numpy() if max_gap_cells else grid)
    baseline = frame["BASELINE_PRE"].to_numpy(dtype=float)
    complete = np.isfinite(filled).all(axis=1) & np.isfinite(baseline)
    retained = frame.loc[complete].reset_index(drop=True)
    centered = filled[complete] - baseline[complete, None]
    return retained, centered, {
        "input_traces": int(len(frame)), "retained_traces": int(complete.sum()),
        "excluded_incomplete_or_invalid_baseline": int((~complete).sum()),
        "imputed_cells_in_retained_traces": int(np.isnan(grid[complete]).sum()),
        "grid_cells": len(grid_columns), "max_interior_gap_cells": max_gap_cells,
    }


def fit_basis(centered, n_components=N_COMP, fit_subsample=FIT_SUBSAMPLE, seed=SEED):
    if n_components < 1 or fit_subsample < n_components + 1:
        raise ValueError("Need a positive component count and at least components+1 fit rows")
    if len(centered) < n_components + 1 or centered.shape[1] < n_components:
        raise ValueError("Insufficient complete traces or grid cells for requested components")
    if not np.isfinite(centered).all() or np.var(centered, axis=0).sum() <= 0:
        raise ValueError("PCA requires finite traces with nonzero between-trace variance")
    rng = np.random.default_rng(seed)
    indices = (rng.choice(len(centered), fit_subsample, replace=False)
               if len(centered) > fit_subsample else np.arange(len(centered)))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(centered[indices])
    components = pca.components_.copy()
    pivots = np.argmax(np.abs(components), axis=1)
    signs = np.where(components[np.arange(n_components), pivots] >= 0, 1., -1.)
    pca.components_ *= signs[:, None]
    scores = pca.transform(centered)
    evr = pca.explained_variance_ratio_
    if not (np.isfinite(evr).all() and (evr >= 0).all() and evr.sum() <= 1 + 1e-10):
        raise ValueError("Invalid explained-variance fractions")
    if not np.allclose(pca.components_ @ pca.components_.T, np.eye(n_components), atol=1e-8):
        raise ValueError("PCA components are not orthonormal")
    # A projection must not increase the centered residual sum of squares.
    check = centered[indices[:min(1000, len(indices))]] - pca.mean_
    reconstructed = (check @ pca.components_.T) @ pca.components_
    if np.sum((check - reconstructed)**2) > np.sum(check**2) + 1e-7:
        raise ValueError("PCA reconstruction increased residual energy")
    return pca, scores, signs, int(len(indices))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--components", type=int, default=N_COMP)
    parser.add_argument("--fit-subsample", type=int, default=FIT_SUBSAMPLE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-gap-cells", type=int, choices=[0,1,2], default=0,
                        help="Explicit historical sensitivity only; primary corrected traces use zero")
    parser.add_argument("--historical-per-user", type=Path,
                        help="Optional diagnostic comparison; never used to orient or fit the basis")
    args = parser.parse_args(argv)
    started = time.time()
    frame, centered, counts = complete_traces(pd.read_parquet(args.traces), args.max_gap_cells)
    source_metadata = args.traces.with_suffix(".metadata.json")
    if source_metadata.exists():
        trace_metadata = json.loads(source_metadata.read_text())
        grid_minutes = np.asarray(trace_metadata["grid_minutes"], dtype=float)
    else:
        # Inputs without accompanying metadata use the standard 5-minute cell-start grid.
        if counts["grid_cells"] != 55:
            raise ValueError("Trace grids require producer metadata")
        trace_metadata = {"definition_version":"unverified-cell-grid"}
        grid_minutes = np.arange(-30, 245, 5, dtype=float)
    if len(grid_minutes) != centered.shape[1] or not np.isfinite(grid_minutes).all():
        raise ValueError("Trace grid coordinates do not match observed cells")
    pca, scores, signs, n_fit = fit_basis(centered, args.components, args.fit_subsample, args.seed)
    score_cols = [f"fpc{i+1}" for i in range(args.components)]
    sdf = pd.DataFrame(scores, columns=score_cols)
    sdf.insert(0, "USER_ID", frame.USER_ID.to_numpy())
    sdf.insert(1, "MEAL_EVENT_ID", frame.MEAL_EVENT_ID.to_numpy())
    per_user = sdf.groupby("USER_ID")[score_cols].agg(["mean", "std", "median"])
    per_user.columns = [f"{c}_{stat}" for c, stat in per_user.columns]
    per_user["n_fpca_meals"] = sdf.groupby("USER_ID").size()
    comparisons = []
    if args.historical_per_user is not None:
        historical = pd.read_csv(args.historical_per_user).set_index("USER_ID")
        overlap = per_user.index.intersection(historical.index)
        for name in score_cols[:3]:
            column = name + "_mean"
            if column not in historical:
                continue
            paired = pd.concat([per_user.loc[overlap, column], historical.loc[overlap, column]], axis=1).dropna()
            r = float(np.corrcoef(paired.to_numpy().T)[0, 1]) if len(paired) >= 3 else np.nan
            comparisons.append({"component": name, "n_overlap": len(paired),
                                "pearson_r": r if np.isfinite(r) else None})
    metadata = {
        "definition_version": DEFINITION_VERSION, **counts,
        "participants": len(per_user), "fit_traces": n_fit, "seed": args.seed,
        "components": args.components, "grid_minutes": grid_minutes.tolist(),
        "trace_definition_version": trace_metadata["definition_version"],
        "component_orientation": "largest absolute loading positive",
        "historical_comparison_used_for_fitting_or_orientation": False,
        "historical_comparisons": comparisons,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "scope": "Descriptive pooled meal basis; biomarker comparisons require independently matched windows and cohorts.",
    }
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sdf.to_parquet(out / "meal_fpca_scores.parquet", index=False)
    per_user.reset_index().to_csv(out / "meal_fpca_per_user.csv", index=False)
    np.savez_compressed(out / "meal_fpca_components.npz", components=pca.components_,
                        mean=pca.mean_, explained_variance_ratio=pca.explained_variance_ratio_,
                        score_std=sdf[score_cols].std().to_numpy(),
                        grid_minutes=grid_minutes, sign_alignment=signs)
    (out / "meal_fpca_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    lines = ["# Meal-trace PCA", "",
             f"{counts['input_traces']:,} input traces; {len(sdf):,} complete retained traces; {len(per_user):,} participants.",
             f"Basis fitted on {n_fit:,} meals with seed {args.seed}; {args.components} components.",
             "Signs follow the largest absolute loading; no historical participant file is required.",
             "Explained-variance fractions: " + ", ".join(f"{x:.6f}" for x in pca.explained_variance_ratio_) + ".", "",
             "This is a descriptive pooled basis. Its orientation and variance fractions are not clinical validation.",
             "Full input, output and definition details are in meal_fpca_metadata.json."]
    (out / "meal_fpca_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({**counts, "participants": len(per_user), "elapsed_seconds": round(time.time()-started, 2)}))


if __name__ == "__main__":
    main()
