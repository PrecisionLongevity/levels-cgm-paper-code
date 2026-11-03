"""Command-line entry point shared by the figure and table scripts.

Every display-item script defines make_source_data() (analysis outputs -> source_data/) and draw(out_dir)
(source_data/ -> figure or table) and calls run(). Usage:

    python figures/fig5_hba1c.py source-data   # write source_data/fig5_*.csv from the analysis outputs
    python figures/fig5_hba1c.py draw          # draw the figure from source_data/ (default command)
    python figures/fig5_hba1c.py all           # both

A draw that needs restricted analysis outputs that are not present (Fig. 6, Supplementary Fig. 1) stops with a message
and exit status RESTRICTED_EXIT, which run_all.py reports as skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

RESTRICTED_EXIT = 3


def restricted_outputs_missing(message: str) -> NoReturn:
    """Stop a draw that needs the restricted analysis outputs (not distributed with the repository)."""
    print(message, file=sys.stderr)
    sys.exit(RESTRICTED_EXIT)


def run(doc, make_source_data, draw, default_out):
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', nargs='?', default='draw', choices=['source-data', 'draw', 'all'])
    ap.add_argument('--out-dir', type=Path, default=default_out, help=f'output directory for drawn files (default {default_out})')
    a = ap.parse_args()
    if a.command in ('source-data', 'all'):
        for p in make_source_data():
            print(f'wrote {p}')
    if a.command in ('draw', 'all'):
        a.out_dir.mkdir(parents=True, exist_ok=True)
        for p in draw(a.out_dir):
            print(f'wrote {p}')
