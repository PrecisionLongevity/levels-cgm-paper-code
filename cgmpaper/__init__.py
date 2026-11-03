"""Shared code for the analysis, figures and tables of the CGM paper (Ginart & Loesche).

Modules
  paths                 repository locations, the database location (LEVELS_DB) and the analysis-output directory
  lab_normalization     laboratory unit conversion, accepted ranges, daily aggregation and diabetes-range flags
  lab_reference         normalized laboratory references and first/last laboratory events
  cgm_grid_metrics      gap-aware metrics on 15-minute CGM grids
  cgm_trace_features    gap-aware CGM dynamics features on observed 5-minute cells
  local_exposure_dates  participant-local dates of lifestyle events
  cross_validation      participant-disjoint cross-validation and training-only preprocessing
  publication_panels    loading the canonical laboratory-CGM panel with the analysis eligibility rules
  consent_reference     reading the aggregate consent-duration summary
  source_data           reading and writing the per-display-item source data in source_data/
  cli                   the command-line entry point shared by every figure and table script
  svgfig, mpl_style     figure drawing (SVG primitives and layout; matplotlib style)
  tables                writing the formatted tables
"""
