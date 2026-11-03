"""Repository locations, the study database and the directory that analysis outputs are written to.

LEVELS_DB       path to the study DuckDB database (required by the analysis scripts; never distributed)
LEVELS_OUTPUTS  directory for analysis outputs (default: outputs/ in the repository). Outputs are participant-derived
                and stay on the machine that holds the database.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_DATA = REPO / 'source_data'


def outputs_dir() -> Path:
    """Directory holding the analysis outputs (LEVELS_OUTPUTS, default outputs/ in the repository)."""
    return Path(os.environ.get('LEVELS_OUTPUTS', REPO / 'outputs')).expanduser()


OUTPUTS = outputs_dir()
FIGURES_OUT = OUTPUTS / 'figures'
TABLES_OUT = OUTPUTS / 'tables'


def database_path() -> Path:
    """Path of the study database, read from the LEVELS_DB environment variable."""
    value = os.environ.get('LEVELS_DB')
    if not value:
        raise SystemExit('LEVELS_DB is not set. Set it to the path of the study DuckDB database, '
                         'for example: export LEVELS_DB=/secure/location/levels.duckdb')
    path = Path(value).expanduser()
    if not path.is_file():
        raise SystemExit(f'LEVELS_DB points to {path}, which does not exist.')
    return path


def connect(read_only: bool = True, **config):
    """Open the study database with DuckDB (read-only by default)."""
    import duckdb
    return duckdb.connect(str(database_path()), read_only=read_only, config=config or None)
