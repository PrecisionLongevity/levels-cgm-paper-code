"""Matplotlib style for the supplementary figures drawn with matplotlib (Supplementary Figs. 1 and 3).

Three font sizes (9, 8 and 7 pt), open frame, outward ticks, editable text in SVG, 300-dpi PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BASE, SEC, TICK = 9.0, 8.0, 7.0

BLUE = '#0072B2'
ORANGE = '#D55E00'
INK = '#1B2430'
MUTED = '#5B6573'


def apply_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': BASE,
        'axes.titlesize': BASE,
        'axes.labelsize': BASE,
        'axes.titlelocation': 'left',
        'axes.titleweight': 'normal',
        'axes.titlepad': 6.0,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': INK,
        'axes.labelcolor': INK,
        'text.color': INK,
        'xtick.color': INK,
        'ytick.color': INK,
        'xtick.labelsize': TICK,
        'ytick.labelsize': TICK,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'legend.fontsize': SEC,
        'legend.frameon': False,
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.6,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'svg.hashsalt': 'cgm-paper',
    })


def header(fig, text, x0=0.012, top=0.985, max_frac=0.976):
    """Draw a bold figure title wrapped to the figure width; return the bottom of the title (figure fraction)."""
    renderer = fig.canvas.get_renderer()
    words, lines, cur = text.split(), [], ''
    for w in words:
        trial = (cur + ' ' + w).strip()
        t = fig.text(0, 0, trial, fontsize=BASE, fontweight='bold')
        width = t.get_window_extent(renderer).width / fig.bbox.width
        t.remove()
        if width > max_frac - x0 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    line_h = BASE * 1.32 / 72 / fig.get_figheight()
    for i, ln in enumerate(lines):
        fig.text(x0, top - i * line_h, ln, fontsize=BASE, fontweight='bold', ha='left', va='top', color=INK)
    return top - len(lines) * line_h


def save(fig, out_dir: Path, stem: str):
    """Save the figure as 300-dpi PNG and SVG; return both paths."""
    png, svg = Path(out_dir) / f'{stem}.png', Path(out_dir) / f'{stem}.svg'
    fig.savefig(png, dpi=300, facecolor='white', metadata={'Software': None})
    fig.savefig(svg, facecolor='white', metadata={'Date': None})
    plt.close(fig)
    return [svg, png]


def fmt_n(n):
    return f'{int(n):,}'
