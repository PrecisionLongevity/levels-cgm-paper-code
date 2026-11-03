"""Shared daily-panel paths, dense-day contract, and within-person estimator.

Helper module for analysis/daily_glucose/build_day_panel.py. Defines the dense-day
contract (minimum readings/cells per day, day definition version), reconstructs
event exposures (steps, weight, alcohol, sleep, exercise) on the CGM clock, and
provides shared within-person statistical estimators used by daily-glucose
analyses: a participant-cluster bootstrap mean, participant/calendar
fixed-effects fits with cluster-robust inference, and an atlas assembly helper
for combining multiple contrast estimates.

Inputs: OUTPUTS/daily_glucose/_cache files built by build_day_panel.py
(gluc_local_day.parquet, user_tz_offset.parquet, local_day_metadata.json) and
the daily_exposures database table.

Outputs: OUTPUTS/daily_glucose/_cache/daily_exposures_unique.parquet and
figure6_<code>_estimates.csv files consumed by the atlas assembly.
"""
from pathlib import Path
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cgmpaper.paths import OUTPUTS  # noqa: E402
from cgmpaper import local_exposure_dates
EXPOSURE_VERSION = local_exposure_dates.EXPOSURE_VERSION
OUT = Path(os.environ.get("LEVELS_TEMPORAL_OUTPUT_DIR", OUTPUTS / 'daily_glucose'))
CACHE = OUT / "_cache"
DAY_VERSION = "local-dense-day"
MIN_READINGS, MIN_CELLS = 230, 77
EXPOSURES = CACHE / "daily_exposures_unique.parquet"
EXPOSURE_COLUMNS = ("TOTAL_STEPS WEIGHT ALC_ANY ALC_LOG_COUNT SLEEP_DURATION_MIN "
    "SLEEP_MIDPOINT_HR SLEEP_LOG_COUNT SLEEP_REGULARITY_7D EX_ANY EX_LOG_COUNT "
    "EX_TOTAL_DURATION_MIN EX_AVG_METS EX_SESSIONS_STRENGTH EX_SESSIONS_VIGOROUS_CARDIO "
    "EX_SESSIONS_WALKING EX_SESSIONS_MODERATE_CARDIO").split()


def exposure_fingerprint():
    return {"exposure_version": EXPOSURE_VERSION}


def build_exposure_cache(con):
    """Reconstruct event exposures on the CGM clock; retain date-only ancillary data.

    Duplicate/conflict counts of the database's day-level summaries are kept as diagnostics;
    event labels and sleep timing come from timestamp-based source aggregation instead.
    Only steps and weight retain their supplied date-only alignment.
    """
    fields = ",".join(f'CASE WHEN min(d.{c}) IS NOT DISTINCT FROM max(d.{c}) THEN max(d.{c}) ELSE NULL END AS {c}' for c in EXPOSURE_COLUMNS)
    conflicts = ",".join(f'(min(d.{c}) IS DISTINCT FROM max(d.{c}))::INTEGER AS conflict_{c}' for c in EXPOSURE_COLUMNS)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE unique_exposures AS
      SELECT d.USER_ID,CAST(d.DAY AS DATE) AS DAY,count(*) AS source_rows,{fields},{conflicts}
      FROM daily_exposures d JOIN read_parquet('{CACHE/'gluc_local_day.parquet'}') g
        ON d.USER_ID=g.USER_ID AND CAST(d.DAY AS DATE)=g.LOCAL_DAY
      GROUP BY d.USER_ID,CAST(d.DAY AS DATE)""")
    conflict_counts = con.sql("SELECT " + ",".join(f"coalesce(sum(conflict_{c}),0) AS {c}" for c in EXPOSURE_COLUMNS) + " FROM unique_exposures").df().iloc[0].to_dict()
    counts = con.sql("SELECT count(*) AS unique_days,coalesce(sum(source_rows-1),0) AS duplicate_rows_removed FROM unique_exposures").df().iloc[0].to_dict()
    event_audit = local_exposure_dates.build_event_daily(con, f"read_parquet('{CACHE/'user_tz_offset.parquet'}')")
    zeros = {"ALC_ANY": "false", "EX_ANY": "false", "ALC_LOG_COUNT": "0", "EX_LOG_COUNT": "0",
             "EX_TOTAL_DURATION_MIN": "0"}
    zeros.update({c:"0" for c in EXPOSURE_COLUMNS if c.startswith("EX_SESSIONS_")})
    selected = []
    for column in EXPOSURE_COLUMNS:
        if column in {"TOTAL_STEPS", "WEIGHT"}:
            selected.append(f"a.{column}")
        elif column in zeros:
            selected.append(f"coalesce(e.{column},{zeros[column]}) AS {column}")
        else:
            selected.append(f"e.{column}")
    temporary = EXPOSURES.with_suffix(".building.parquet")
    con.execute(f"""COPY (
        SELECT g.USER_ID,g.LOCAL_DAY AS DAY,{','.join(selected)}
        FROM read_parquet('{CACHE/'gluc_local_day.parquet'}') g
        LEFT JOIN unique_exposures a ON a.USER_ID=g.USER_ID AND a.DAY=g.LOCAL_DAY
        LEFT JOIN publication_event_daily e ON e.USER_ID=g.USER_ID AND e.DAY=g.LOCAL_DAY
      ) TO '{temporary}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    temporary.replace(EXPOSURES)
    con.execute("DROP TABLE unique_exposures")
    return {**{k:int(v) for k,v in counts.items()},
            "conflicting_days_by_field": {k:int(v) for k,v in conflict_counts.items()},
            "duplicate_diagnostic_scope": "duplicate/conflict counts describe the database's source-day summaries; only their steps/weight values are used",
            **exposure_fingerprint(),
            "event_reconstruction": event_audit,
            "ancillary_rule": "steps and weight use stored date-only daily summaries; agreeing nonmissing duplicates retained, conflicts missing; timestamps unavailable for realignment",
            "rule": "one participant dense date; event-derived fields rebuilt from fixed-local source timestamps"}


def require_cache():
    meta = json.loads((CACHE / "local_day_metadata.json").read_text())
    if meta.get("day_definition_version") != DAY_VERSION or not meta.get("dense_only"):
        raise ValueError("Rebuild dense-day substrate with analysis/daily_glucose/build_day_panel.py")
    exposure = meta.get("exposure_audit", {})
    if any(exposure.get(key) != value for key, value in exposure_fingerprint().items()):
        raise ValueError("Rebuild dense-day exposure cache: version changed")
    return meta


def write_result_provenance(script, outputs):
    meta = require_cache()
    record = {"day_definition_version": DAY_VERSION, "input": meta,
              "script": str(script),
              "outputs": [Path(p).name for p in outputs]}
    (OUT / (Path(script).stem + ".metadata.json")).write_text(json.dumps(record, indent=2) + "\n")


def write_figure6_estimates(script, rows):
    """Write aggregate estimates consumed by the atlas, never transcribed numbers."""
    frame = pd.DataFrame(rows)
    required = ["finding", "effect", "ci_low", "ci_high", "n_obs", "n_users", "design"]
    if any(c not in frame for c in required) or not np.isfinite(frame[["effect", "ci_low", "ci_high", "n_obs", "n_users"]]).all().all():
        raise ValueError("Day-level contrasts require finite source estimates and explicit denominators")
    if ((frame.ci_low > frame.effect) | (frame.effect > frame.ci_high)).any():
        raise ValueError("Invalid effect-size confidence interval")
    frame["day_definition_version"] = DAY_VERSION
    frame["source_script"] = Path(script).name
    path = OUT / ("figure6_" + Path(script).name[:2] + "_estimates.csv")
    frame.to_csv(path, index=False)
    return path


def atlas_source_rows(directory):
    """Load the seven independently produced contrasts."""
    directory = Path(directory)
    frames = []
    for code in ["01", "02", "03", "04", "05", "06", "09"]:
        path = directory / f"figure6_{code}_estimates.csv"
        frame = pd.read_csv(path)
        if len(frame) != 1 or frame.day_definition_version.iloc[0] != DAY_VERSION:
            raise ValueError("Each day-level source must provide one dense-day primary contrast")
        frame["source"] = path.name
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.finding.duplicated().any() or not np.isfinite(result[["effect", "ci_low", "ci_high", "n_obs", "n_users"]]).all().all():
        raise ValueError("Duplicate or nonfinite day-level source estimates")
    return result.to_dict("records")


def cluster_mean_interval(data, value, *, group="USER_ID", n_boot=5000, seed=20260916):
    """Row-weighted mean with participant bootstrap retaining every repeated row."""
    grouped = data.dropna(subset=[value, group]).groupby(group)[value].agg(["sum", "count"])
    if len(grouped) < 2:
        raise ValueError("At least two participants are required")
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, len(grouped), len(grouped))
        boots[b] = sums[ix].sum()/counts[ix].sum()
    return tuple(np.quantile(boots, [.025, .975]))


def calendar_effects(u, c, y, w, n_c, iters=1000):
    """Weighted participant/category FE, equal-category zero centered.

    Missing calendar categories, disconnected participant/category graphs and
    unconverged fits fail; none silently defines an arbitrary seasonal level.
    """
    n_u = int(u.max()) + 1
    wu = np.bincount(u, w, minlength=n_u)
    wc = np.bincount(c, w, minlength=n_c)
    if np.any(wu <= 0) or np.any(wc <= 0) or not np.isfinite(y).all():
        raise ValueError("Every fitted participant and calendar category needs finite support")
    first = np.full(n_u, n_c)
    np.minimum.at(first, u, c)
    adjacent = np.zeros((n_c,n_c),dtype=bool)
    adjacent[first[u],c] = True
    adjacent |= adjacent.T.copy()
    reached = {0}
    for _ in range(n_c):
        reached |= set(np.flatnonzero(adjacent[list(reached)].any(axis=0)))
    if len(reached) != n_c:
        raise ValueError("Disconnected calendar design cannot identify a common contrast")
    b = np.zeros(n_c)
    for _ in range(iters):
        previous = b.copy()
        a = np.bincount(u,w*(y-b[c]),minlength=n_u)/wu
        b = np.bincount(c,w*(y-a[u]),minlength=n_c)/wc
        b -= b.mean()
        if np.max(np.abs(b-previous)) < 1e-10:
            return b
    raise ValueError("Calendar fixed-effect fit did not converge")


def calendar_interval(cells, cal_col, n_c, rng, n_boot=500):
    """Day-weighted calendar effects and participant-cluster percentile intervals."""
    users, u = np.unique(cells.USER_ID, return_inverse=True)
    c, y, w = cells[cal_col].to_numpy(int), cells.y.to_numpy(float), cells.w.to_numpy(float)
    point = calendar_effects(u,c,y,w,n_c)
    draws = np.empty((n_boot,n_c))
    for b in range(n_boot):
        multiplicity = np.bincount(rng.integers(0,len(users),len(users)),minlength=len(users))
        keep = multiplicity[u] > 0
        _, ub = np.unique(u[keep],return_inverse=True)
        # Identical copies of a participant share the same fitted intercept;
        # multiplying its cell weights equals relabeling/resampling its rows.
        draws[b] = calendar_effects(ub,c[keep],y[keep],w[keep]*multiplicity[u[keep]],n_c)
    lo,hi=np.percentile(draws,[2.5,97.5],axis=0)
    return pd.DataFrame({cal_col:np.arange(n_c),"effect":point,"ci_low":lo,"ci_high":hi}),len(users)


def fixed_effects(data, outcome, terms, *, controls=("DOW", "MONTH"), min_days=5):
    """Person FE, calendar indicators, participant-cluster CR1 with absorbed df.

    Calendar columns are demeaned with exposures. Rank determines residual df;
    an unavailable or unidentified exposure is returned as NaN, never zero.
    """
    d = data.dropna(subset=[outcome, "USER_ID"] + list(terms) + list(controls)).copy()
    d = d[d.groupby("USER_ID")[outcome].transform("size").ge(min_days)].reset_index(drop=True)
    ids, codes = np.unique(d.USER_ID.to_numpy(), return_inverse=True)
    if len(ids) < 2 or len(d) <= len(ids) + len(terms):
        raise ValueError("Insufficient independent groups/within-person observations")
    pieces = [d[list(terms)].astype(float)]
    pieces += [pd.get_dummies(d[c].astype(int), prefix=c, drop_first=True).astype(float) for c in controls]
    x = pd.concat(pieces, axis=1)
    names = list(x.columns)
    xd = x - x.groupby(codes).transform("mean")
    yd = d[outcome].astype(float) - d[outcome].astype(float).groupby(codes).transform("mean")
    within_sd = xd[list(terms)].pow(2).mean().pow(.5)
    keep = xd.std(ddof=0).gt(1e-12).to_numpy()
    matrix = xd.to_numpy()[:, keep]
    kept = np.array(names)[keep].tolist()
    gram = matrix.T @ matrix
    values, vectors = np.linalg.eigh(gram)
    positive = values > (max(float(values.max(initial=0)), 1.0)*1e-12)
    rank = int(positive.sum())
    identified = np.sum(vectors[:, ~positive]**2, axis=1) < 1e-10
    bread = (vectors[:, positive]/values[positive]) @ vectors[:, positive].T
    beta = bread @ (matrix.T @ yd.to_numpy())
    residual = yd.to_numpy() - matrix @ beta
    scores = np.zeros((len(ids), matrix.shape[1]))
    np.add.at(scores, codes, matrix * residual[:, None])
    n, groups = len(d), len(ids)
    dof = n - groups - rank
    if dof <= 0:
        raise ValueError("No residual degrees of freedom after participant effects")
    covariance = groups / (groups-1) * (n-1) / dof * bread @ (scores.T @ scores) @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    rows = []
    for term in terms:
        if term in kept and identified[kept.index(term)]:
            i = kept.index(term)
            b, s = float(beta[i]), float(se[i])
        else:
            b = s = float("nan")
        rows.append({"term": term, "beta": b, "se": s, "ci_low": b-1.96*s, "ci_high": b+1.96*s,
                     "n_obs": n, "n_users": groups, "within_sd": float(within_sd[term]),
                     "n_term_positive_obs": int(d[term].gt(0).sum()),
                     "n_term_zero_obs": int(d[term].eq(0).sum()),
                     "n_users_with_term_variation": int(d.groupby("USER_ID")[term].nunique().gt(1).sum()),
                     "residual_df": dof, "within_design_rank": rank})
    return pd.DataFrame(rows)
