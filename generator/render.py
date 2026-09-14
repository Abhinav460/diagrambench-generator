"""Turn a ``geometry_spec`` into a PNG. The only module that imports matplotlib.

Rendering is centralized here, rather than left to each family, for two reasons.
The labeling policy -- which measurements appear in the figure and which are
withheld -- is the property the benchmark actually measures, so it must be
reviewable in one place instead of reconstructed from a dozen families. And
byte-level determinism is fragile enough that it needs a single controlled path.

Determinism
-----------
Two renders of the same spec must produce byte-identical PNGs, so that a run is
reproducible from its seed and so that re-running the generator does not churn a
directory of images that are visually identical but differ in bytes. The measures:

- The ``Agg`` backend, selected before ``pyplot`` is imported, so no display
  backend can vary the rasterization.
- Fixed ``figsize`` and ``dpi``, and explicit ``set_xlim``/``set_ylim``. Autoscaling
  derives limits from artist extents, which shift with font metrics.
- No ``tight_layout``. It reflows the axes based on rendered text extents, which
  can differ between matplotlib versions and even between fonts of the same name.
- PNG ``Software`` metadata suppressed, since it otherwise embeds the matplotlib
  version and makes output differ across environments for no visual reason.

Limits are computed as a *square* window around the content so that
``set_aspect("equal")`` never has to adjust them, which would reintroduce the
autoscaling this is avoiding.
"""

from __future__ import annotations

import math

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import Circle, PathPatch  # noqa: E402

from generator.geometry import (  # noqa: E402
    GeometryError,
    region_geometries,
)
from generator.registry import GeometrySpec  # noqa: E402

__all__ = ["render", "contact_sheet", "RenderError"]

#: Output geometry. Fixed so image dimensions carry no signal about problem type.
FIGSIZE = (5.0, 5.5)
DPI = 150

#: Fraction of the content's half-width added as margin on every side.
PADDING = 0.14

#: Visual style. Deliberately plain: a benchmark figure should look like a textbook
#: diagram, not like a plot.
SHADE_COLOR = "#7ab8e0"
SHADE_ALPHA = 1.0
LINE_COLOR = "#1a1a1a"
LINE_WIDTH = 1.6
LABEL_SIZE = 11
TITLE_SIZE = 13

#: Gap under the title. Labels now reach the top of the window, so the title needs
#: clearance from them rather than from the polygon.
TITLE_PAD = 18

#: How far a measurement label sits off its segment, as a fraction of that
#: segment's distance from its own polygon's centre. Scaled to the owner rather
#: than to the window so a small inner polygon gets a correspondingly small
#: offset; a window-scaled offset collapses every label on a small polygon onto
#: its centre.
LABEL_OFFSET = 0.32

#: Style for the coordinate-layout primitives. A grid has to be visible enough to
#: count squares against and faint enough not to compete with the figure, since
#: reading vertices off it is the task.
GRID_COLOR = "#c8c8c8"
GRID_WIDTH = 0.8
AXIS_COLOR = "#5a5a5a"
AXIS_WIDTH = 1.2
POINT_SIZE = 26


class RenderError(ValueError):
    """Raised when a spec cannot be drawn -- a dangling id, or an unknown kind.

    Distinct from a matplotlib error so a malformed spec is not mistaken for a
    plotting bug.
    """


def _polygons(spec: GeometrySpec) -> dict[str, list[list[float]]]:
    """Index polygon primitives by id so regions and labels can resolve them."""
    out: dict[str, list[list[float]]] = {}
    for item in spec:
        if item.get("kind") == "polygon":
            pid = item.get("id")
            if not isinstance(pid, str):
                raise RenderError(f"polygon primitive needs a string id; got {pid!r}")
            if pid in out:
                raise RenderError(f"duplicate polygon id {pid!r}")
            out[pid] = [list(map(float, p)) for p in item["points"]]
    return out


def _signed_area(points: Sequence[Sequence[float]]) -> float:
    """Shoelace signed area; positive when the ring winds counter-clockwise."""
    total = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _oriented(points: list[list[float]], counter_clockwise: bool) -> list[list[float]]:
    """Force a ring's winding direction.

    Matplotlib fills compound paths by the nonzero winding rule, so a hole must
    wind opposite to the ring containing it. Normalizing here rather than trusting
    the family means a family that emits clockwise vertices still renders correctly.
    """
    is_ccw = _signed_area(points) > 0
    return points if is_ccw == counter_clockwise else points[::-1]


def _region_patch(outer: list[list[float]], holes: Iterable[list[list[float]]]) -> PathPatch:
    """Build a filled patch with holes, as one compound path."""
    vertices: list[list[float]] = []
    codes: list[int] = []

    def add_ring(ring: list[list[float]], ccw: bool) -> None:
        ring = _oriented(ring, ccw)
        vertices.extend(ring + [ring[0]])
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1) + [MplPath.CLOSEPOLY])

    add_ring(outer, ccw=True)
    for hole in holes:
        add_ring(hole, ccw=False)

    return PathPatch(
        MplPath(vertices, codes),
        facecolor=SHADE_COLOR,
        alpha=SHADE_ALPHA,
        edgecolor="none",
        zorder=1,
    )


def _window(
    polygons: dict[str, list[list[float]]],
    labels: list[tuple[float, float, str]],
    extra: Sequence[tuple[float, float]] = (),
) -> tuple[float, float, float, float]:
    """A square window around all content, padded. Square so equal aspect is free.

    Label positions are included in the bounds. They sit outside the polygons by
    construction, so framing on the polygons alone pushes them against the edge of
    the image or out of it entirely -- and into the title.
    """
    xs = (
        [x for pts in polygons.values() for x, _ in pts]
        + [x for x, _, _ in labels]
        + [x for x, _ in extra]
    )
    ys = (
        [y for pts in polygons.values() for _, y in pts]
        + [y for _, y, _ in labels]
        + [y for _, y in extra]
    )
    if not xs:
        raise RenderError("spec contains nothing to frame")

    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * (1 + PADDING)
    return cx - half, cx + half, cy - half, cy + half


def _label_positions(
    spec: GeometrySpec, polys: dict[str, list[list[float]]]
) -> list[tuple[float, float, str]]:
    """Where each drawn measurement label goes, as (x, y, text).

    Computed in one place because two callers need it and must agree: the window
    calculation has to include label positions in its bounds, and the drawing pass
    has to put text at exactly those points. Deriving them twice is how labels end
    up outside the frame.

    Every label is pushed along its edge's **outward normal**, by a distance scaled
    to the owner polygon's size.

    An earlier version pushed labels radially away from the polygon's centroid.
    That is equivalent for a convex figure, and wrong for a concave one: the edges
    of an L-shaped notch face inward relative to the centroid, so their labels were
    pushed onto the outline instead of away from it. The normal is defined by the
    edge itself and so does not care about the shape's overall form.

    Scaling by the owner's size rather than the window's is what keeps a small
    inner polygon's labels just off its edges instead of flung across the figure.
    """
    positions: list[tuple[float, float, str]] = []
    for item in spec:
        if item.get("kind") != "length_label" or not item.get("draw"):
            continue

        owner = item.get("owner")
        if owner is not None and owner not in polys:
            raise RenderError(f"length_label names unknown owner {owner!r}")

        (x1, y1), (x2, y2) = item["segment"]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        points = polys.get(owner) if owner in polys else None
        if points is None:
            positions.append((mx, my, item["text"]))
            continue

        nx, ny = _outward_normal(points, (x1, y1), (x2, y2))
        gap = _label_gap(points, math.hypot(x2 - x1, y2 - y1))
        positions.append((mx + nx * gap, my + ny * gap, item["text"]))
    return positions


def _signed_area_of(points: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for i, p1 in enumerate(points):
        p2 = points[(i + 1) % len(points)]
        total += p1[0] * p2[1] - p2[0] * p1[1]
    return total / 2


def _outward_normal(
    points: Sequence[Sequence[float]],
    start: Sequence[float],
    end: Sequence[float],
) -> tuple[float, float]:
    """Unit normal of an edge, pointing out of the polygon.

    For a counter-clockwise ring the outward normal of the edge from ``start`` to
    ``end`` is ``(dy, -dx)`` normalized. The winding is measured rather than
    assumed, so a family that hands over a clockwise ring still gets labels on the
    outside instead of silently inside.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return (0.0, 0.0)
    nx, ny = dy / length, -dx / length
    if _signed_area_of(points) < 0:
        nx, ny = -nx, -ny
    return (nx, ny)


def _label_gap(points: Sequence[Sequence[float]], edge_length: float) -> float:
    """How far off its edge a label sits, scaled to that edge rather than the figure.

    Scaling to the whole figure crowds the labels of a concave notch: both of the
    notch's edges are short and face into the same small void, so a gap sized for
    the figure pushes their labels on top of each other -- two adjacent digits that
    read as one number.

    A floor proportional to the figure keeps a very short edge on a large polygon
    from having its label sit on the outline.
    """
    cx, cy = _centroid(points)
    figure_size = sum(math.hypot(x - cx, y - cy) for x, y in points) / len(points)
    return max(LABEL_OFFSET * edge_length, 0.15 * figure_size)


def _draw_labels(ax: Any, positions: list[tuple[float, float, str]]) -> None:
    """Render measurement text at precomputed positions."""
    for x, y, text in positions:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=LABEL_SIZE,
            color=LINE_COLOR,
            zorder=3,
            # A background box keeps a label legible where it lands on the shaded
            # band rather than on the white ground.
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"),
        )


def _centroid(points: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Vertex-average centre. Exact for the regular polygons this draws."""
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _as_polygons(shape: Any) -> list[Any]:
    """Flatten a shapely result into drawable polygons.

    A boolean operation can return a MultiPolygon (a difference that splits a
    figure in two) or an empty geometry, and matplotlib needs one path per piece.
    Non-areal leftovers -- the stray lines an intersection of touching edges can
    produce -- carry no fill and are dropped.
    """
    if shape.is_empty:
        return []
    parts = list(getattr(shape, "geoms", [shape]))
    return [part for part in parts if part.geom_type == "Polygon" and not part.is_empty]


def _circle_extents(spec: GeometrySpec) -> list[tuple[float, float]]:
    """The four extreme points of each circle, so framing includes curved shapes.

    A circle contributes no vertices to the polygon bounds, so a figure whose
    circle extends past its polygons would otherwise be cropped.
    """
    out: list[tuple[float, float]] = []
    for item in spec:
        if item.get("kind") != "circle":
            continue
        cx, cy = float(item["center"][0]), float(item["center"][1])
        r = float(item["radius"])
        out.extend([(cx - r, cy), (cx + r, cy), (cx, cy - r), (cx, cy + r)])
    return out


#: How far a coordinate label sits from its vertex, as a fraction of the figure's
#: size. Enough to clear the dot and the outline without detaching from the vertex
#: it names.
POINT_LABEL_OFFSET = 0.10


def _point_positions(spec: GeometrySpec) -> list[tuple[float, float, str, bool]]:
    """Marked points and their labels, as (x, y, text, draw)."""
    out: list[tuple[float, float, str, bool]] = []
    for item in spec:
        if item.get("kind") != "point_label":
            continue
        x, y = item["point"]
        out.append((float(x), float(y), str(item.get("text", "")), bool(item.get("draw", True))))
    return out


def _point_label_anchors(
    spec: GeometrySpec, polys: dict[str, list[list[float]]]
) -> list[tuple[float, float, float, float, str, bool]]:
    """Dot position and text position per marked point: (px, py, tx, ty, text, draw).

    Text is pushed away from the figure's centre so it never lands on the shaded
    fill, and so two adjacent vertices' labels diverge instead of colliding. The
    text anchor is returned separately from the dot because the window has to be
    framed around the text -- a label placed outside the polygon is exactly the
    thing that gets clipped by the frame otherwise.
    """
    points = _point_positions(spec)
    if not points:
        return []

    all_vertices = [v for pts in polys.values() for v in pts] or [(x, y) for x, y, _, _ in points]
    cx, cy = _centroid(all_vertices)
    span = max(
        (math.hypot(x - cx, y - cy) for x, y, _, _ in points),
        default=1.0,
    ) or 1.0
    gap = POINT_LABEL_OFFSET * span * 2

    out: list[tuple[float, float, float, float, str, bool]] = []
    for x, y, text, draw in points:
        dx, dy = x - cx, y - cy
        length = math.hypot(dx, dy)
        if length == 0:
            dx, dy, length = 0.0, 1.0, 1.0
        out.append((x, y, x + dx / length * gap, y + dy / length * gap, text, draw))
    return out


def _draw_axes(ax: Any, spec: GeometrySpec, x0: float, x1: float, y0: float, y1: float) -> None:
    """Draw a unit grid and the two axes, for coordinate-layout problems.

    The grid is the measuring instrument: with vertices unlabelled, counting
    squares is how a solver recovers coordinates, so the spacing has to be exactly
    the unit the family reasons in.
    """
    axes_items = [item for item in spec if item.get("kind") == "axes"]
    if not axes_items:
        return

    step = float(axes_items[0].get("step", 1.0))
    if step <= 0:
        raise RenderError(f"axes step must be positive; got {step}")

    start = math.floor(x0 / step) * step
    while start <= x1:
        ax.plot([start, start], [y0, y1], color=GRID_COLOR, linewidth=GRID_WIDTH, zorder=0)
        start += step

    start = math.floor(y0 / step) * step
    while start <= y1:
        ax.plot([x0, x1], [start, start], color=GRID_COLOR, linewidth=GRID_WIDTH, zorder=0)
        start += step

    ax.plot([x0, x1], [0, 0], color=AXIS_COLOR, linewidth=AXIS_WIDTH, zorder=1)
    ax.plot([0, 0], [y0, y1], color=AXIS_COLOR, linewidth=AXIS_WIDTH, zorder=1)


def _draw_points(ax: Any, anchors: list[tuple[float, float, float, float, str, bool]]) -> None:
    """Mark vertices, and label those whose coordinates are given.

    A point with ``draw`` false is still marked but not labelled: the dot says
    "this vertex matters", while withholding the text is what forces the solver to
    read its position off the grid.
    """
    for x, y, tx, ty, text, draw in anchors:
        ax.scatter([x], [y], s=POINT_SIZE, color=LINE_COLOR, zorder=4)
        if draw and text:
            ax.text(
                tx,
                ty,
                text,
                ha="center",
                va="center",
                fontsize=LABEL_SIZE,
                color=LINE_COLOR,
                zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"),
            )


def _draw_into(ax: Any, spec: GeometrySpec, stem: Optional[str]) -> None:
    """Draw one problem into an existing axes. Shared by render and contact_sheet."""
    polys = _polygons(spec)
    labels = _label_positions(spec, polys)
    anchors = _point_label_anchors(spec, polys)
    extra = (
        _circle_extents(spec)
        + [(x, y) for x, y, _, _, _, _ in anchors]
        + [(tx, ty) for _, _, tx, ty, _, draw in anchors if draw]
    )
    x0, x1, y0, y1 = _window(polys, labels, extra)

    known = {"polygon", "circle", "region", "length_label", "axes", "point_label"}
    for item in spec:
        kind = item.get("kind")
        if kind not in known:
            raise RenderError(f"unknown primitive kind {kind!r}")

    _draw_axes(ax, spec, x0, x1, y0, y1)

    # Regions are filled from the same geometry verify measures, so a figure cannot
    # be shaded differently from the region the answer describes.
    try:
        for _, shape in region_geometries(spec):
            for part in _as_polygons(shape):
                ax.add_patch(
                    _region_patch(
                        [list(c) for c in part.exterior.coords[:-1]],
                        [[list(c) for c in ring.coords[:-1]] for ring in part.interiors],
                    )
                )
    except GeometryError as exc:
        raise RenderError(str(exc)) from exc

    for pts in polys.values():
        closed = pts + [pts[0]]
        ax.plot(
            [p[0] for p in closed],
            [p[1] for p in closed],
            color=LINE_COLOR,
            linewidth=LINE_WIDTH,
            solid_joinstyle="miter",
            zorder=2,
        )

    for item in spec:
        if item.get("kind") == "circle":
            cx, cy = item["center"]
            ax.add_patch(
                Circle(
                    (float(cx), float(cy)),
                    float(item["radius"]),
                    fill=False,
                    edgecolor=LINE_COLOR,
                    linewidth=LINE_WIDTH,
                    zorder=2,
                )
            )

    _draw_points(ax, anchors)

    _draw_labels(ax, labels)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if stem:
        ax.set_title(stem, fontsize=TITLE_SIZE, color=LINE_COLOR, pad=TITLE_PAD)


def render(spec: GeometrySpec, path: str | Path, *, stem: Optional[str] = None) -> Path:
    """Draw a spec to ``path`` as a PNG and return the path.

    ``stem`` is burned into the image as a title rather than left to the prompt,
    because the benchmark's prompt tells the model that everything it needs is in
    the image. A question that lived only in the surrounding text would break that
    contract.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    try:
        _draw_into(ax, spec, stem)
        # Software metadata would otherwise embed the matplotlib version, making
        # output differ across environments with no visual difference.
        fig.savefig(path, dpi=DPI, format="png", metadata={"Software": None})
    finally:
        plt.close(fig)
    return path


def contact_sheet(
    problems: Sequence[tuple[str, GeometrySpec]],
    path: str | Path,
    *,
    cols: int = 3,
) -> Path:
    """Tile several problems into one image for eyeballing rendering quality.

    A review aid, not part of the generation path: judging whether figures look
    like plausible benchmark items needs them side by side, not one at a time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = (len(problems) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(FIGSIZE[0] * cols, FIGSIZE[1] * rows), dpi=DPI
    )
    try:
        flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
        for ax, (stem, spec) in zip(flat, problems):
            _draw_into(ax, spec, stem)
        for ax in flat[len(problems):]:
            ax.axis("off")
        fig.savefig(path, dpi=DPI, format="png", metadata={"Software": None})
    finally:
        plt.close(fig)
    return path
