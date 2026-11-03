#!/usr/bin/env python3
"""Run the whole analysis: producers -> source data -> figures and tables.

Stages
  producers    analysis/<group>/*.py: read the study database (LEVELS_DB) and write analysis outputs to outputs/
               (or LEVELS_OUTPUTS). Outputs include participant-level files and stay on the secure machine.
  source-data  figures/*.py and tables/*.py `source-data`: summarize the outputs into source_data/*.csv (aggregate only)
  draw         figures/*.py and tables/*.py `draw`: render each display item into outputs/figures and outputs/tables
               from source_data/ (Fig. 6 and Supplementary Fig. 1 also need restricted analysis outputs and are
               skipped without them, see README.md)

Usage (from the repository root)
  export LEVELS_DB=/secure/location/levels_irb_v2_2026-10-16.duckdb
  python run_all.py --list
  python run_all.py                          # all stages
  python run_all.py --stage draw             # only re-render from source_data/
  python run_all.py --stage producers --start-at meals/spike_facets
  python run_all.py --rscript /path/to/Rscript   # R with iglu 4.2.2, for analysis/clinical/mage_standard.py

Steps run in order and the first failure stops the run; a draw skipped for want of restricted analysis outputs is
reported and the run continues. Logs: outputs/logs/<step>.log
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cgmpaper.cli import RESTRICTED_EXIT  # noqa: E402
from cgmpaper.paths import OUTPUTS  # noqa: E402

# (step, script, arguments); order respects every producer's inputs
PRODUCERS = [
    # participant-level substrates
    ('canonical_panel/build_panel_symmetric', 'analysis/canonical_panel/build_panel.py', ['--window', 'symmetric']),
    ('canonical_panel/build_panel_preceding', 'analysis/canonical_panel/build_panel.py', ['--window', 'preceding']),
    ('meals/build_meals', 'analysis/meals/build_meals.py', ['--stage', 'all']),
    ('meals/meal_fpca', 'analysis/meals/meal_fpca.py', []),
    ('daily_glucose/build_day_panel', 'analysis/daily_glucose/build_day_panel.py', []),
    ('reliability/repeatability', 'analysis/reliability/repeatability.py', ['--stage', 'all']),
    ('meals/meal_shape_qualification', 'analysis/meals/meal_shape_qualification.py', []),
    ('discordance/discordance_synthesis', 'analysis/discordance/discordance_synthesis.py', []),
    ('meals/carb_sensitivity_index', 'analysis/meals/carb_sensitivity_index.py', []),
    ('meals/spike_exposure', 'analysis/meals/spike_exposure.py', []),
    ('meals/spike_facets', 'analysis/meals/spike_facets.py', []),
    ('reliability/context', 'analysis/reliability/context.py', []),
    ('cohort/cohort_reference', 'analysis/cohort/cohort_reference.py', []),
    ('meals/food_variance', 'analysis/meals/food_variance.py', []),
    ('meals/split_half_reliability', 'analysis/meals/split_half_reliability.py', []),
    ('meals/meal_populations', 'analysis/meals/meal_populations.py', []),
    # analyses
    ('metabolic/cgm_features', 'analysis/metabolic/cgm_features.py', []),
    ('metabolic/insulin_markers', 'analysis/metabolic/insulin_markers.py', []),
    ('meals/same_food_averaging', 'analysis/meals/same_food_averaging.py', []),
    ('metabolic/hba1c_stratified_spikes', 'analysis/metabolic/hba1c_stratified_spikes.py', []),
    ('validation/diabetes_range_exclusion', 'analysis/validation/diabetes_range_exclusion.py', []),
    ('meals/food_variance_log_scale', 'analysis/meals/food_variance_log_scale.py', []),
    ('cohort/glycemic_composition', 'analysis/cohort/glycemic_composition.py', []),
    ('metabolic/level_quarters', 'analysis/metabolic/level_quarters.py', []),
    ('metabolic/observability_inputs', 'analysis/metabolic/observability_inputs.py', []),
    ('metabolic/discordance_behavior', 'analysis/metabolic/discordance_behavior.py', []),
    ('metabolic/within_range_associations', 'analysis/metabolic/within_range_associations.py', []),
    ('metabolic/bmi_same_sample', 'analysis/metabolic/bmi_same_sample.py', []),
    ('metabolic/hba1c_range_resolution', 'analysis/metabolic/hba1c_range_resolution.py', []),
    ('metabolic/meal_rows_no_zero_fill', 'analysis/metabolic/meal_rows_no_zero_fill.py', []),
    ('metabolic/cgm_vs_hba1c', 'analysis/metabolic/cgm_vs_hba1c.py', []),
    ('cohort/stream_overlap', 'analysis/cohort/stream_overlap.py', []),
    ('metabolic/discordance_behavior_no_zero_fill', 'analysis/metabolic/discordance_behavior_no_zero_fill.py', []),
    ('cohort/multimodal_distributions', 'analysis/cohort/multimodal_distributions.py', []),
    ('metabolic/hba1c_within_range_slopes', 'analysis/metabolic/hba1c_within_range_slopes.py', []),
    ('reliability/reliability_by_gap', 'analysis/reliability/reliability_by_gap.py', []),
    ('metabolic/unadjusted_insulin_markers', 'analysis/metabolic/unadjusted_insulin_markers.py', []),
    ('metabolic/hba1c_unadjusted_r2', 'analysis/metabolic/hba1c_unadjusted_r2.py', []),
    ('canonical_panel/insulin_value_sensitivity', 'analysis/canonical_panel/insulin_value_sensitivity.py', []),
    ('meals/same_food_averaging_preventive', 'analysis/meals/same_food_averaging_preventive.py', []),
    ('cohort/lab_distributions', 'analysis/cohort/lab_distributions.py', []),
    ('metabolic/carb_sensitivity_continuous', 'analysis/metabolic/carb_sensitivity_continuous.py', []),
    ('metabolic/spike_width_check', 'analysis/metabolic/spike_width_check.py', []),
    ('meals/same_food_raw_vs_within', 'analysis/meals/same_food_raw_vs_within.py', []),
    ('meals/meal_shape_macro_adjusted', 'analysis/meals/meal_shape_macro_adjusted.py', []),
    ('meals/same_food_direct', 'analysis/meals/same_food_direct.py', []),
    # summaries behind Figs 3-5 and the Supplementary Tables
    ('clinical/clinical_summaries', 'analysis/clinical/clinical_summaries.py', []),
    ('clinical/spike_width', 'analysis/clinical/spike_width.py', []),
    ('clinical/markers', 'analysis/clinical/markers.py', []),
    ('clinical/mage_standard', 'analysis/clinical/mage_standard.py', []),
    ('clinical/spike_profile', 'analysis/clinical/spike_profile.py', []),
    ('clinical/discordance_groups', 'analysis/clinical/discordance_groups.py', []),
    ('clinical/gmi_calibration', 'analysis/clinical/gmi_calibration.py', []),
    ('clinical/contrast_pvalues', 'analysis/clinical/contrast_pvalues.py', []),
    # example participants for Figs 1 and 6 (restricted outputs)
    ('exemplars/exemplar_candidates', 'analysis/exemplars/exemplar_candidates.py', []),
    ('exemplars/exemplar_traces', 'analysis/exemplars/exemplar_traces.py', []),
]
FIGURES = ['figures/fig1_cohort.py', 'figures/fig2_reliability.py', 'figures/fig3_average_glucose.py',
           'figures/fig4_spike_width.py', 'figures/fig5_hba1c.py', 'figures/fig6_meals.py',
           'figures/sfig1_gmi_calibration.py', 'figures/sfig2_discordant_groups.py', 'figures/sfig3_age_offset.py']
TABLES = ['tables/table1_cohort.py', 'tables/stable1a_markers.py', 'tables/stable1b_groups.py', 'tables/stable2_meal_width.py',
          'tables/stable3_hba1c_sensitivity.py', 'tables/stable4_lab_retest.py', 'tables/stable5_cgm_measures.py',
          'tables/stable6_lab_units.py', 'tables/stable7_age_sex.py']


def steps(stage):
    out = []
    if stage in ('all', 'producers'):
        out += [(name, script, args) for name, script, args in PRODUCERS]
    for command in ('source-data', 'draw'):
        if stage in ('all', command):
            out += [(f'{command}/{Path(s).stem}', s, [command]) for s in FIGURES + TABLES]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['all', 'producers', 'source-data', 'draw'], default='all')
    ap.add_argument('--start-at', help='step name to start from (see --list)')
    ap.add_argument('--only', help='run only this step')
    ap.add_argument('--list', action='store_true', help='print the steps and exit')
    ap.add_argument('--rscript', help='Rscript of an R installation with iglu 4.2.2 (analysis/clinical/mage_standard.py)')
    a = ap.parse_args(argv)
    plan = steps(a.stage)
    names = [s[0] for s in plan]
    for opt in (a.start_at, a.only):
        if opt and opt not in names:
            raise SystemExit(f'unknown step {opt}; see --list')
    if a.start_at:
        plan = plan[names.index(a.start_at):]
    if a.only:
        plan = [s for s in plan if s[0] == a.only]
    if a.list:
        for name, script, args in plan:
            print(f"{name:48s} {script} {' '.join(args)}")
        return 0
    missing = [script for _, script, _ in plan if not (ROOT / script).is_file()]
    if missing:
        raise SystemExit('missing scripts: ' + ', '.join(missing))
    logs = OUTPUTS / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    skipped = []
    for name, script, args in plan:
        if script.endswith('mage_standard.py') and a.rscript:
            args = args + ['--rscript', a.rscript]
        t0 = time.monotonic()
        log = logs / (name.replace('/', '__') + '.log')
        print(f'[{time.strftime("%H:%M:%S")}] {name}', flush=True)
        with log.open('w') as fh:
            r = subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode == RESTRICTED_EXIT and name.startswith('draw/'):
            skipped.append(name)
            print(f'    skipped: needs restricted analysis outputs; see {log}', flush=True)
            continue
        if r.returncode:
            raise SystemExit(f'{name} failed (exit {r.returncode}); see {log}')
        print(f'    done in {time.monotonic() - t0:.0f} s', flush=True)
    if skipped:
        print(f'skipped (restricted analysis outputs not available): {", ".join(skipped)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
