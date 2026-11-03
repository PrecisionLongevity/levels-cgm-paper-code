"""SVG drawing primitives and shared layout for the main figures and Supplementary Fig. 2.

The figures are written as SVG directly (text stays live and editable). One canvas unit is one point at 100% scale; the
canvas is 640 units wide and is declared at its print width in millimetres. PNG files are rendered from the SVG with
cairosvg at 2,400 pixels wide.
"""
from __future__ import annotations

import html
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


# Validated categorical order. Do not reorder or cycle.
CATEGORICAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

INK = "#132238"
INK_SECONDARY = "#475467"
INK_MUTED = "#667085"
RULE = "#D0D5DD"
GRID = "#EAECF0"
SURFACE = "#FFFFFF"
WASH = "#F7F9FC"
ACCENT = "#D55E00"

# Sequential ramp, single hue, monotonic in lightness (light -> dark).
BLUE_RAMP = ["#EAF3FA", "#C6E0F0", "#93C6E3", "#56A5D2", "#2C82BC", "#0072B2", "#08507E"]

FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

TYPE = {
    "figure_title": 15.5,
    "figure_subtitle": 10.0,
    "panel_title": 10.5,
    "panel_label": 12.0,
    "axis_title": 9.0,
    "tick": 8.0,
    "annotation": 7.6,
    "footer": 6.6,
}


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def fmt(v: float) -> str:
    """Trim floats so the SVG source stays readable and diffable."""
    r = round(float(v), 2)
    return str(int(r)) if r == int(r) else str(r)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def _attrs(**kw) -> str:
    out = []
    for k, v in kw.items():
        if v is None:
            continue
        out.append(f'{k.replace("_", "-")}="{esc(v) if isinstance(v, str) else fmt(v)}"')
    return " ".join(out)


def rect(x, y, w, h, fill=INK, rx=0, **kw) -> str:
    return f"<rect {_attrs(x=x, y=y, width=w, height=h, fill=fill, rx=rx, **kw)}/>"


def line(x1, y1, x2, y2, stroke=RULE, stroke_width=1, **kw) -> str:
    return f"<line {_attrs(x1=x1, y1=y1, x2=x2, y2=y2, stroke=stroke, stroke_width=stroke_width, **kw)}/>"


def circle(cx, cy, r, fill=INK, **kw) -> str:
    return f"<circle {_attrs(cx=cx, cy=cy, r=r, fill=fill, **kw)}/>"


def path(d, fill="none", stroke=None, stroke_width=1.6, **kw) -> str:
    return f"<path {_attrs(d=d, fill=fill, stroke=stroke, stroke_width=stroke_width, **kw)}/>"


def text(
    x,
    y,
    s,
    size=TYPE["tick"],
    fill=INK_SECONDARY,
    anchor="start",
    weight=None,
    baseline=None,
    **kw,
) -> str:
    return (
        f"<text {_attrs(x=x, y=y, font_size=size, fill=fill, text_anchor=anchor, font_family=FONT, font_weight=weight, dominant_baseline=baseline, **kw)}>"
        f"{esc(s)}</text>"
    )


def group(children, transform=None, **kw) -> str:
    body = "".join(c for c in children if c)
    return f"<g {_attrs(transform=transform, **kw)}>{body}</g>"


# --------------------------------------------------------------------------
# scales
# --------------------------------------------------------------------------


@dataclass
class Linear:
    d0: float
    d1: float
    r0: float
    r1: float

    def __call__(self, v: float) -> float:
        return self.r0 + (float(v) - self.d0) / (self.d1 - self.d0) * (self.r1 - self.r0)


@dataclass
class Band:
    keys: list
    r0: float
    r1: float
    pad: float = 0.22
    _step: float = field(init=False, default=0.0)

    def __post_init__(self):
        n = max(1, len(self.keys))
        self._step = (self.r1 - self.r0) / n

    def __call__(self, k) -> float:
        i = self.keys.index(k)
        return self.r0 + i * self._step + self._step * self.pad / 2

    def center(self, k) -> float:
        return self(k) + self.bandwidth / 2

    @property
    def bandwidth(self) -> float:
        return self._step * (1 - self.pad)


def nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    raw = (hi - lo) / max(1, count)
    mag = 10 ** math.floor(math.log10(abs(raw))) if raw else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw / mag <= m:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + step * 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def nice_bounds(lo: float, hi: float, pad: float = 0.06) -> tuple[float, float]:
    span = hi - lo
    return lo - span * pad, hi + span * pad


# --------------------------------------------------------------------------
# panel + axes
# --------------------------------------------------------------------------


@dataclass
class Panel:
    """A plotting rectangle with margins. x0/y0 is the top-left of the whole box."""

    x0: float
    y0: float
    w: float
    h: float
    ml: float = 54
    mr: float = 14
    mt: float = 26
    mb: float = 34

    @property
    def px0(self) -> float:
        return self.x0 + self.ml

    @property
    def py0(self) -> float:
        return self.y0 + self.mt

    @property
    def px1(self) -> float:
        return self.x0 + self.w - self.mr

    @property
    def py1(self) -> float:
        return self.y0 + self.h - self.mb

    @property
    def ph(self) -> float:
        return self.py1 - self.py0


def axis_bottom(p: Panel, sc, ticks, label="", fmt_fn=None, rotate=0) -> str:
    fmt_fn = fmt_fn or (lambda v: fmt(v))
    out = [line(p.px0, p.py1, p.px1, p.py1, stroke=RULE)]
    for t in ticks:
        x = sc(t) if not isinstance(sc, Band) else sc.center(t)
        out.append(line(x, p.py1, x, p.py1 + 3.5, stroke=RULE))
        out.append(
            text(x, p.py1 + 13, fmt_fn(t), size=TYPE["tick"], fill=INK_MUTED, anchor="middle")
        )
    if label:
        out.append(
            text(
                (p.px0 + p.px1) / 2,
                p.py1 + (30 if rotate else 27),
                label,
                size=TYPE["axis_title"],
                fill=INK_SECONDARY,
                anchor="middle",
            )
        )
    return group(out)


def axis_left(p: Panel, sc, ticks, label="", fmt_fn=None, grid=True) -> str:
    fmt_fn = fmt_fn or (lambda v: fmt(v))
    out = []
    for t in ticks:
        y = sc(t) if not isinstance(sc, Band) else sc.center(t)
        if grid:
            out.append(line(p.px0, y, p.px1, y, stroke=GRID))
        out.append(
            text(p.px0 - 6, y + 2.8, fmt_fn(t), size=TYPE["tick"], fill=INK_MUTED, anchor="end")
        )
    if label:
        cy = (p.py0 + p.py1) / 2
        out.append(
            text(
                p.x0 + 11,
                cy,
                label,
                size=TYPE["axis_title"],
                fill=INK_SECONDARY,
                anchor="middle",
                transform=f"rotate(-90,{fmt(p.x0 + 11)},{fmt(cy)})",
            )
        )
    return group(out)




def polyline(pts, stroke, width=2.0, fill="none") -> str:
    d = "M " + " L ".join(f"{fmt(x)} {fmt(y)}" for x, y in pts)
    return path(d, fill=fill, stroke=stroke, stroke_width=width, stroke_linejoin="round")


def band(pts_lo, pts_hi, fill, opacity=0.16) -> str:
    d = (
        "M "
        + " L ".join(f"{fmt(x)} {fmt(y)}" for x, y in pts_lo)
        + " L "
        + " L ".join(f"{fmt(x)} {fmt(y)}" for x, y in reversed(pts_hi))
        + " Z"
    )
    return path(d, fill=fill, stroke=None, opacity=opacity)


def arrow_h(x0, y, length, stroke=INK_MUTED, width=1.2, head=3.4) -> str:
    """Horizontal arrow drawn as geometry.

    Arrow glyphs are not reliably present in the fonts a renderer or a
    typesetter may substitute, and a missing glyph prints as tofu. Drawing the
    arrow keeps the figure independent of font coverage.
    """
    x1 = x0 + length
    d = -1 if length < 0 else 1
    return group(
        [
            line(x0, y, x1, y, stroke=stroke, stroke_width=width),
            path(
                f"M {fmt(x1)} {fmt(y)} L {fmt(x1 - head * d)} {fmt(y - head * 0.62)} "
                f"L {fmt(x1 - head * d)} {fmt(y + head * 0.62)} Z",
                fill=stroke,
                stroke=None,
            ),
        ]
    )


def legend_row(x, y, items, size=TYPE["annotation"], swatch=8, pitch=None) -> str:
    out = []
    cx = x
    for lab, col in items:
        out.append(rect(cx, y - swatch + 2, swatch, swatch, fill=col, rx=1.6))
        out.append(text(cx + swatch + 5, y + 1, lab, size=size, fill=INK_SECONDARY))
        cx += (pitch or (swatch + 12 + len(lab) * size * 0.52))
    return group(out)


# ----------------------------------------------------------------------------------------------- page layout and text
W = 640.0                 # canvas width (units)
PRINT_MM = 183.0          # printed width of the main figures
F = dict(title=12.4, subtitle=8.4, ptitle=9.6, plabel=11.4, head=8.4, body=7.8, small=7.6)
S = F['small']
TOP = 6.0                 # top margin when no figure title is drawn
CHAR_W = {None: 0.53, '400': 0.53, '600': 0.56, '700': 0.57, 'bold': 0.57}


def tw(s, size, weight=None):
    """Estimated width of a text string (units)."""
    return len(str(s)) * float(size) * CHAR_W.get(weight, 0.55)


def wrap(s, width_units, size, weight=None):
    return textwrap.wrap(s, width=max(8, int(width_units / (size * CHAR_W.get(weight, 0.55)))))


def plabel(x, y, letter, title, size=None):
    """Panel letter (bold) followed by the panel title."""
    return group([text(x, y + 9, letter, size=F['plabel'], fill=INK, weight='bold'),
                  text(x + 14, y + 9, title, size=size or F['ptitle'], fill=INK, weight='600')])


def header(title, subtitle=''):
    """Figure title (and optional subtitle) drawn at the top of the canvas; returns (elements, y below the header)."""
    els, y = [], 18.0
    for t in wrap(title, W - 12, F['title'], 'bold'):
        els.append(text(6, y, t, size=F['title'], fill=INK, weight='bold'))
        y += 15.0
    y += 0.5
    for t in wrap(subtitle, W - 12, F['subtitle']):
        els.append(text(6, y, t, size=F['subtitle'], fill=INK_MUTED))
        y += 10.5
    return els, y + 2.0


def render_svg(h, head_els, body, title, footer='', print_mm=PRINT_MM):
    """Assemble the SVG document: white background, header elements, body elements and optional footer lines."""
    lines = wrap(footer, W - 12, S) if footer else []
    yf = h - 5 - 9.4 * (len(lines) - 1)
    foot = []
    for s in lines:
        foot.append(text(6, yf, s, size=S, fill=INK_MUTED))
        yf += 9.4
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{fmt(print_mm)}mm" height="{fmt(print_mm * h / W)}mm" '
            f'viewBox="0 0 {fmt(W)} {fmt(h)}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title>'
            + rect(0, 0, W, h, fill=SURFACE) + ''.join(head_els) + ''.join(b for b in body if b) + ''.join(foot) + '</svg>')


def write(svg_text, out_dir, stem):
    """Write the SVG and a 2,400-pixel-wide PNG rendered from it; return both paths."""
    import cairosvg
    svg = Path(out_dir) / f'{stem}.svg'
    svg.write_text(svg_text, encoding='utf-8')
    png = svg.with_suffix('.png')
    cairosvg.svg2png(url=str(svg), write_to=str(png), scale=2400 / W, background_color='white')
    return [svg, png]


# ----------------------------------------------------------------------------------------------- numbers
def wilson(k, n, z=1.959963984540054):
    """Wilson score 95% interval, in percent."""
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def excl0(lo, hi):
    """True when the interval excludes 0."""
    return bool(lo > 0 or hi < 0)


def sg(v, d=1):
    """Signed number with a typographic minus."""
    return f'{v:+,.{d}f}'.replace('-', '−')


def pct(v, d=1):
    return sg(v, d) + '%'


def ci_txt(e, lo, hi, d=1):
    return f'{pct(e, d)} ({sg(lo, d)} to {sg(hi, d)})'


def pick(frame, **kw):
    """The single row of a data frame matching every column=value pair."""
    s = frame
    for k, v in kw.items():
        s = s.loc[s[k] == v]
    if len(s) != 1:
        raise ValueError(f'expected one row for {kw}, found {len(s)}')
    return s.iloc[0]


# ----------------------------------------------------------------------------------------------- marks
def dot(x, y, col, filled, r=3.0):
    return circle(x, y, r, fill=col if filled else SURFACE, stroke=col, **{'stroke-width': 1.2})


def ci_mark(x, lim, y, e, lo, hi, col, r=3.0, caps=False):
    """Interval and point estimate; filled when the 95% CI excludes 0. caps=True adds end caps."""
    out = [line(x(max(lo, lim[0])), y, x(min(hi, lim[1])), y, stroke=col, stroke_width=1.4)]
    if caps:
        for v in (lo, hi):
            if lim[0] <= v <= lim[1]:
                out.append(line(x(v), y - 3.2, x(v), y + 3.2, stroke=col, stroke_width=1.2))
    out.append(dot(x(e), y, col, excl0(lo, hi), r))
    return out


def wrap_lines(x, y, s, width, size=S, lead=9.6, fill=INK_SECONDARY, weight=None, **kw):
    out = []
    for part in wrap(s, width, size, weight):
        out.append(text(x, y, part, size=size, fill=fill, weight=weight, **kw))
        y += lead
    return out, y


def key_filled(x, y, col=INK_SECONDARY):
    """Key: filled marker = 95% CI excludes 0; open marker = includes 0."""
    return [dot(x + 3, y - 2.6, col, True), text(x + 9, y, '95% CI excludes 0', size=S, fill=INK_SECONDARY),
            dot(x + 88, y - 2.6, col, False),
            text(x + 94, y, '95% CI includes 0 (not detected)', size=S, fill=INK_SECONDARY)]


def vbar_ci(cx, y, lo, hi, col=INK):
    return [line(cx, y(lo), cx, y(hi), stroke=col, stroke_width=1.0),
            line(cx - 4, y(lo), cx + 4, y(lo), stroke=col, stroke_width=1.0),
            line(cx - 4, y(hi), cx + 4, y(hi), stroke=col, stroke_width=1.0)]
