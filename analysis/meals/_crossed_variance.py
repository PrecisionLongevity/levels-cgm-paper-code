"""Method-of-moments crossed (person x food) variance-component estimation.

Estimates variance components for person, food, person-by-food interaction,
and residual (occasion) effects from a crossed person/food design, using
seeded Monte Carlo estimation of expected mean squares (EMS), since exact
finite-draw EMS quantities are not tractable for this design. Negative signed
variance estimates are retained in diagnostics and truncated at zero only for
the conventional nonnegative descriptive shares. Also provides a null-based
correction that subtracts a Monte Carlo null mean from the raw person-food
component before computing derived ratios (e.g. personal share of stable-food
variance, food-to-personal ratio, single-meal personal reliability).
"""
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

@dataclass
class Design:
    person: np.ndarray
    food: np.ndarray
    cell: np.ndarray
    person_count: np.ndarray
    food_count: np.ndarray
    cell_count: np.ndarray
    df: np.ndarray
    n: int
    nperson: int
    nfood: int
    ncell: int
    connected_components: int

def remove_group_means(
    values: np.ndarray, codes: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    means = np.bincount(codes, weights=values, minlength=len(counts)) / counts
    return values - means[codes]


def additive_person_food_residual(
    values: np.ndarray,
    design: Design,
    tolerance: float = 1e-10,
    max_iterations: int = 1_000,
) -> np.ndarray:
    """Residual from values ~ person + food via alternating projections.

    This is algebraically the same least-squares projection used in
    analysis/meals/food_variance.py, but exploits the two categorical factors
    instead of building a large sparse matrix.
    """
    residual = values - values.mean()
    previous_ss = np.inf
    for _ in range(max_iterations):
        residual = remove_group_means(
            residual, design.person, design.person_count
        )
        residual = remove_group_means(residual, design.food, design.food_count)
        ss = float(residual @ residual)
        if abs(previous_ss - ss) <= tolerance * max(ss, 1.0):
            return residual
        previous_ss = ss
    raise RuntimeError("Alternating person/food projections did not converge")


def prepare_design(person: np.ndarray, food: np.ndarray) -> Design:
    if len(person) != len(food) or len(person) == 0 or pd.isna(person).any() or pd.isna(food).any():
        raise ValueError("Crossed design requires aligned nonempty nonmissing identities")
    person = pd.factorize(person, sort=False)[0].astype(np.int64, copy=False)
    food = pd.factorize(food, sort=False)[0].astype(np.int64, copy=False)
    nperson = int(person.max()) + 1
    nfood = int(food.max()) + 1
    cell_key = person * nfood + food
    cell = pd.factorize(cell_key, sort=False)[0].astype(np.int64, copy=False)
    ncell = int(cell.max()) + 1
    n = len(person)
    person_count = np.bincount(person, minlength=nperson).astype(float)
    food_count = np.bincount(food, minlength=nfood).astype(float)
    cell_count = np.bincount(cell, minlength=ncell).astype(float)
    # Rank(person+food) = nperson+nfood-number of connected bipartite components.
    row = np.r_[person, food+nperson]
    column = np.r_[food+nperson, person]
    graph = coo_matrix((np.ones(len(row)),(row,column)),shape=(nperson+nfood,nperson+nfood)).tocsr()
    components = connected_components(graph, directed=False, return_labels=False)
    additive_rank = nperson+nfood-components
    df = np.array([nperson-1,nfood-components,ncell-additive_rank,n-ncell],dtype=float)
    if (df <= 0).any():
        raise ValueError("Crossed person/food/interaction/occasion variances are not identifiable: zero design degrees of freedom")
    return Design(
        person,
        food,
        cell,
        person_count,
        food_count,
        cell_count,
        df,
        n,
        nperson,
        nfood,
        ncell,
        int(components),
    )


def mean_squares(design: Design, y: np.ndarray) -> np.ndarray:
    centered = y - y.mean()
    person_residual = remove_group_means(y, design.person, design.person_count)
    additive_residual = additive_person_food_residual(y, design)
    cell_residual = remove_group_means(y, design.cell, design.cell_count)
    r0 = float(centered @ centered)
    r1 = float(person_residual @ person_residual)
    r2 = float(additive_residual @ additive_residual)
    r3 = float(cell_residual @ cell_residual)
    return np.array([r0 - r1, r1 - r2, r2 - r3, r3]) / design.df


def ems_coefficients(design: Design, n_trace: int, seed: int) -> np.ndarray:
    if n_trace < 1:
        raise ValueError("Positive Monte Carlo trace count required")
    rng = np.random.default_rng(seed)
    coefficients = np.zeros((4, 4))
    factor_codes = [design.person, design.food, design.cell, None]
    factor_sizes = [design.nperson, design.nfood, design.ncell, design.n]
    for column, (codes, size) in enumerate(zip(factor_codes, factor_sizes)):
        accumulated = np.zeros(4)
        for _ in range(n_trace):
            if codes is None:
                simulated = rng.standard_normal(design.n)
            else:
                simulated = rng.standard_normal(size)[codes]
            accumulated += mean_squares(design, simulated)
        coefficients[:, column] = accumulated / n_trace
    return coefficients


def fit_from_coefficients(
    design: Design, y: np.ndarray, coefficients: np.ndarray
) -> dict[str, float]:
    y=np.asarray(y,float)
    if y.shape != (design.n,) or not np.isfinite(y).all():
        raise ValueError("Finite response for every crossed-design row required")
    condition=float(np.linalg.cond(coefficients))
    if not np.isfinite(condition) or condition>1e12:
        raise ValueError("Monte Carlo EMS matrix is singular or too ill-conditioned")
    ms=mean_squares(design,y)
    signed=np.linalg.solve(coefficients,ms)
    s2=np.maximum(signed,0)
    total=float(s2.sum())
    labels=['person','food','person_food','occasion']
    return {**{f's2_{name}':float(s2[i]) for i,name in enumerate(labels)},
            **{f's2_{name}_signed':float(signed[i]) for i,name in enumerate(labels)},
            **{f'pct_{name}_raw':float(100*s2[i]/total) if total>0 else np.nan for i,name in enumerate(labels)},
            'ems_condition_number':condition,'n_boundary_components':int((signed<0).sum()),
            'raw_personal_share_stable_food':ratio(100*s2[2],s2[1]+s2[2]),
            'raw_food_to_personal_ratio':ratio(s2[1],s2[2]),
            'raw_single_meal_personal_reliability':ratio(s2[2],s2[2]+s2[3])}


def ratio(numerator,denominator):
    return float(numerator/denominator) if denominator>0 else np.nan


def corrected_statistics(observed: dict[str, float], null_values: Iterable[float]) -> dict[str, float]:
    null_array = np.asarray(list(null_values), dtype=float)
    if len(null_array)==0 or not np.isfinite(null_array).all():
        raise ValueError("Finite nonempty null estimates required")
    null_mean = float(np.mean(null_array))
    corrected_pf = max(observed["s2_person_food"] - null_mean, 0.0)
    corrected = np.array(
        [
            observed["s2_person"],
            observed["s2_food"],
            corrected_pf,
            observed["s2_occasion"],
        ]
    )
    total = float(corrected.sum())
    stable_food = observed["s2_food"] + corrected_pf
    return {
        **observed,
        "s2_person_food_null_mean": null_mean,
        "s2_person_food_corrected": corrected_pf,
        "null_excess_at_zero_boundary": bool(corrected_pf == 0),
        "food_to_personal_ratio_defined": bool(corrected_pf > 0),
        "pct_person_corrected": ratio(100 * corrected[0], total),
        "pct_food_corrected": ratio(100 * corrected[1], total),
        "pct_person_food_corrected": ratio(100 * corrected[2], total),
        "pct_occasion_corrected": ratio(100 * corrected[3], total),
        "personal_share_stable_food": ratio(100 * corrected_pf, stable_food),
        "food_to_personal_ratio": ratio(observed["s2_food"], corrected_pf),
        "single_meal_personal_reliability": ratio(corrected_pf, corrected_pf + observed["s2_occasion"]),
    }

