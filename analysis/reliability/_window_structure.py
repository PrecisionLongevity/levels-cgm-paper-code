"""Shared definitions for the seven-day CGM windows and the factor analysis of CGM metric axes.

Constants: RAW_DAILY_GATE (>= 230 accepted raw readings per local day), CELLS_PER_DAY (96 fifteen-minute cells),
MIN_15M_CELLS (>= 77 observed cells for a dense day), LAB_SPECS / DERIVED_SPECS (laboratory analytes and accepted
ranges), METRIC_COLUMNS and AXES (CGM metric axes). factor_analysis() returns the PCA variance, PCA loadings and the
three-factor principal-axis loadings (varimax) of the standardized metric panel.
Used by analysis/canonical_panel/_window_features.py and analysis/reliability/repeatability.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cgmpaper.lab_normalization import BOUNDS as LAB_VALIDITY_BOUNDS

RAW_DAILY_GATE = 230
CELLS_PER_DAY = 96
MIN_15M_CELLS = 77
METRIC_COLUMNS = [
    "mean_glucose", "median_glucose", "p10_glucose", "p90_glucose",
    "sd_glucose", "cv_glucose", "iqr_glucose", "tar140", "tbr70",
    "modd", "day_mean_sd", "conga_2h", "gvp", "ac_mean", "ac_var",
]
AXES = ["mean_glucose", "sd_glucose", "modd", "ac_var"]
LAB_SPECS = {
    "A1C": ("a1c", *LAB_VALIDITY_BOUNDS["a1c"]),
    "FASTING_GLUCOSE": ("fasting_glucose", 30.0, 600.0),
    "FASTING_INSULIN": ("fasting_insulin", 0.5, 500.0),
    "TRIGLYCERIDES": ("triglycerides", 10.0, 5000.0),
    "HDL": ("hdl_cholesterol", 5.0, 200.0),
    "LDL": ("ldl_cholesterol", 10.0, 500.0),
    "TOTAL_CHOL": ("total_cholesterol", 50.0, 800.0),
    "APOB": ("apo_b", 20.0, 250.0),
    "HSCRP": ("hscrp", 0.01, 100.0),
    "URIC_ACID": ("uric_acid", 1.0, 20.0),
    "ALT": ("alt", 1.0, 500.0),
    "AST": ("ast", 1.0, 500.0),
}
DERIVED_SPECS = {
    "HOMA_IR": (0.05, 100.0),
    "TG_HDL_RATIO": (0.05, 100.0),
    "TYG": (4.0, 15.0),
}




def factor_analysis(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = primary[METRIC_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna().copy()
    for column in panel:
        low, high = panel[column].quantile([0.005, 0.995])
        panel[column] = panel[column].clip(low, high)
    scaled = StandardScaler().fit_transform(panel)
    pca_model = PCA().fit(scaled)
    pca_rows = pd.DataFrame(
        {
            "component": np.arange(1, len(METRIC_COLUMNS) + 1),
            "explained_variance": pca_model.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca_model.explained_variance_ratio_),
        }
    )
    pca_loadings = pd.DataFrame(
        pca_model.components_[:5].T * np.sqrt(pca_model.explained_variance_[:5]),
        index=METRIC_COLUMNS,
        columns=[f"PC{i}" for i in range(1, 6)],
    ).reset_index(names="metric")
    # Principal-axis factoring is used instead of sklearn's maximum-likelihood
    # FactorAnalysis because the intentionally redundant CGM panel drives the ML
    # optimizer to a Heywood boundary. PAF is deterministic and converges on the
    # correlation matrix without pretending that redundant transforms are
    # independent measurements.
    correlation = np.corrcoef(scaled, rowvar=False)
    inverse = np.linalg.pinv(correlation)
    communalities = np.clip(1.0 - 1.0 / np.diag(inverse), 0.05, 0.99)
    unrotated = None
    for _ in range(1000):
        reduced = correlation.copy()
        np.fill_diagonal(reduced, communalities)
        eigenvalues, eigenvectors = np.linalg.eigh(reduced)
        order = np.argsort(eigenvalues)[::-1][:3]
        positive = np.maximum(eigenvalues[order], 0)
        unrotated = eigenvectors[:, order] * np.sqrt(positive)
        updated = np.clip(np.sum(np.square(unrotated), axis=1), 0.05, 0.99)
        if np.max(np.abs(updated - communalities)) < 1e-6:
            communalities = updated
            break
        communalities = updated
    if unrotated is None:
        raise AssertionError("Principal-axis factor extraction failed")

    # Orthogonal varimax rotation (Kaiser 1958).
    rotation = np.eye(3)
    previous = 0.0
    for _ in range(1000):
        loadings_rotated = unrotated @ rotation
        u, singular, vh = np.linalg.svd(
            unrotated.T
            @ (np.power(loadings_rotated, 3)
               - (1.0 / unrotated.shape[0])
               * loadings_rotated
               @ np.diag(np.diag(loadings_rotated.T @ loadings_rotated)))
        )
        rotation = u @ vh
        objective = float(singular.sum())
        if previous and objective / previous < 1.0 + 1e-7:
            break
        previous = objective
    rotated = unrotated @ rotation
    efa_loadings = pd.DataFrame(
        rotated,
        index=METRIC_COLUMNS,
        columns=["Factor1", "Factor2", "Factor3"],
    )
    # Orient each factor so its largest absolute loading is positive. This does
    # not change the solution and makes the table stable/readable.
    for column in efa_loadings:
        strongest = efa_loadings[column].abs().idxmax()
        if efa_loadings.loc[strongest, column] < 0:
            efa_loadings[column] *= -1
    efa_loadings = efa_loadings.reset_index(names="metric")
    return pca_rows, pca_loadings, efa_loadings
