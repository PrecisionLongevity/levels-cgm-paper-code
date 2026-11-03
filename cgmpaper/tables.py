"""Writing the formatted tables (Markdown and CSV of the printed cells)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)] +
                     ['| ' + ' | '.join('' if pd.isna(v) else str(v) for v in r) + ' |' for r in df.itertuples(index=False)])


def write(parts, out_dir, name, title, note=''):
    """Write the printed table as <name>.md (all parts) and <name>.csv (one part) or <name>_<part>.csv (several parts).

    parts: a DataFrame, or a list of (part label, DataFrame).
    """
    out_dir = Path(out_dir)
    if isinstance(parts, pd.DataFrame):
        parts = [('', parts)]
    md, paths = [f'# {title}', ''], []
    for label, df in parts:
        if label:
            md += [f'## {label}', '']
        md += [md_table(df), '']
        p = out_dir / (f'{name}.csv' if len(parts) == 1 else f"{name}_{label.split(' ')[0].lower()}.csv")
        df.to_csv(p, index=False)
        paths.append(p)
    if note:
        md += [note, '']
    p = out_dir / f'{name}.md'
    p.write_text('\n'.join(md), encoding='utf-8')
    return [p] + paths
