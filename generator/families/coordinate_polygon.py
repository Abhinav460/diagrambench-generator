"""Coordinate layouts: a lattice polygon on a grid, with some vertices unlabelled.

Covers the paper's *coordinate-based layouts* configuration and targets the
coordinate-misassignment error category -- incorrect placement or reading of points
in the plane.

The withheld quantities
-----------------------
Every vertex is marked with a dot, but only some carry their coordinate pair as
text. The rest must be read off the grid by counting squares. A model that
misplaces one of them gets a wrong area, and the error is a pure coordinate
misassignment rather than an arithmetic slip -- which is what makes this family a
clean probe of that category.

The origin is drawn but the polygon is not anchored to it. That is the
``offset_origin`` trap: a layout whose natural reference point is not the obvious
corner, so a model that assumes the figure starts at (0, 0) misreads every vertex.

The area is exact and rational by the shoelace formula::

    A = |sum over edges of (x_i * y_{i+1} - x_{i+1} * y_i)| / 2

Lattice vertices make that a half-integer at worst, so answers are exact rationals
rather than radicals -- a third answer shape alongside the polygon families'
cotangents and the composite family's integers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.random import Generator as RNG
    from sympy import Expr
else:
    RNG = Any
    Expr = Any

from generator.params import parse_params
from generator.registry import (
    GEOMETRY,
    READABILITY,
    GeometrySpec,
    Issue,
    Params,
    ValidationResult,
    first_issue,
)

NAME = "coordinate_polygon"

#: Integer vertices only, including for structured input. This is a lattice family:
#: the withheld vertices are recovered by counting grid squares, which a vertex at
#: (2.5, 1) defeats. A non-lattice polygon belongs in a different family.
PARAM_TYPES = {"points": "lattice_points"}

#: Lattice bounds. Kept small so the grid stays countable at figure scale -- the
#: whole task is counting squares, and a 40-wide grid renders as hatching.
MIN_COORD = -6
MAX_COORD = 6

#: Vertex count. Three and four only: beyond that a randomly drawn simple polygon
#: is usually a spiky non-convex shape that reads as a mistake rather than a figure.
MIN_VERTICES = 3
MAX_VERTICES = 4

#: Smallest acceptable area, in grid squares. Anything less is a sliver whose
#: vertices cannot be told apart visually.
MIN_AREA = 6.0

#: How many vertices keep their printed coordinates. The rest must be read off the
#: grid. One withheld vertex is the minimum that makes the figure load-bearing.
LABELLED_VERTICES = 2


def shoelace(points: list[list[float]]) -> float:
    """Twice the signed area, halved and absolute. The numeric cross-check."""
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def exact_area(points: list[list[int]]) -> Expr:
    """Shoelace area as an exact sympy Rational over integer lattice points."""
    import sympy as sp

    total = sp.Integer(0)
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += sp.Integer(x1) * sp.Integer(y2) - sp.Integer(x2) * sp.Integer(y1)
    return sp.Abs(total) / 2


def _is_simple_and_convex(points: list[list[float]]) -> bool:
    """Whether the vertices wind consistently, making a simple convex polygon.

    Convexity is checked via the sign of every consecutive cross product. A
    consistent sign means no reflex vertex and, for a closed ring, no
    self-intersection -- so one test rules out both the bowtie shapely would refuse
    and the spiky shape that reads as a rendering bug.
    """
    n = len(points)
    signs = []
    for i in range(n):
        ax, ay = points[i]
        bx, by = points[(i + 1) % n]
        cx, cy = points[(i + 2) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if cross == 0:
            return False  # three collinear vertices: a degenerate corner
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def _counter_clockwise(points: list[list[int]]) -> list[list[int]]:
    """Force positive winding so every family hands the renderer the same convention."""
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return points if total > 0 else list(reversed(points))


def sample(rng: RNG) -> Params:
    """Draw three or four lattice points."""
    count = int(rng.integers(MIN_VERTICES, MAX_VERTICES + 1))
    points = [
        [int(rng.integers(MIN_COORD, MAX_COORD + 1)), int(rng.integers(MIN_COORD, MAX_COORD + 1))]
        for _ in range(count)
    ]
    return {"points": _counter_clockwise(points)}


def _segments_intersect(a: list[int], b: list[int], c: list[int], d: list[int]) -> bool:
    """Whether closed segments ab and cd share any point. Exact on integers."""

    def orient(p: list[int], q: list[int], r: list[int]) -> int:
        cross = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (cross > 0) - (cross < 0)

    def on_segment(p: list[int], q: list[int], r: list[int]) -> bool:
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(a, c, b))
        or (o2 == 0 and on_segment(a, d, b))
        or (o3 == 0 and on_segment(c, a, d))
        or (o4 == 0 and on_segment(c, b, d))
    )


def _is_simple(points: list[list[int]]) -> bool:
    """Whether the vertices form a simple polygon of positive area, convex or not.

    The geometric requirement the shoelace answer actually has. Weaker than
    ``_is_simple_and_convex``: an arrowhead or an L on the lattice is simple, and its
    shoelace area is its true area.
    """
    n = len(points)
    if shoelace([[float(x), float(y)] for x, y in points]) == 0:
        return False
    for i in range(n):
        a, b, c = points[i], points[(i + 1) % n], points[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        dot = (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1])
        if cross == 0 and dot < 0:
            return False  # the outline doubles back on itself
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # adjacent through the closing edge
            if _segments_intersect(points[i], points[(i + 1) % n], points[j], points[(j + 1) % n]):
                return False
    return True


def validation_issues(params: Params) -> Iterator[Issue]:
    """Every problem with ``params``, by severity, in the order ``is_valid`` checks.

    Geometry: at least three distinct vertices forming a simple polygon of positive
    area. Readability: the vertex-count ceiling, the lattice bounds, convexity, and
    the minimum area -- the filters that keep random draws from looking like
    rendering bugs, none of which a real figure needs to satisfy.
    """
    try:
        points = parse_params(PARAM_TYPES, params)["points"]
    except (KeyError, TypeError, ValueError) as exc:
        yield GEOMETRY, f"malformed params: {exc}"
        return

    count_reason = f"vertex count {len(points)} outside [{MIN_VERTICES}, {MAX_VERTICES}]"
    if len(points) < MIN_VERTICES:
        yield GEOMETRY, count_reason
        return
    if len(points) > MAX_VERTICES:
        yield READABILITY, count_reason
    if len({tuple(p) for p in points}) != len(points):
        yield GEOMETRY, "duplicate vertices"
        return
    for x, y in points:
        if not (MIN_COORD <= x <= MAX_COORD and MIN_COORD <= y <= MAX_COORD):
            yield READABILITY, f"vertex ({x}, {y}) outside the lattice bounds"
            break

    shape_reason = "vertices are collinear or wind inconsistently (non-convex or self-intersecting)"
    if not _is_simple(points):
        yield GEOMETRY, shape_reason
        return
    if not _is_simple_and_convex(points):
        yield READABILITY, shape_reason

    area = shoelace([[float(x), float(y)] for x, y in points])
    if area < MIN_AREA:
        yield READABILITY, f"area {area:.1f} is below the readable minimum of {MIN_AREA}"


def is_valid(params: Params) -> ValidationResult:
    """Reject degenerate, non-convex, or unreadably small polygons."""
    return first_issue(validation_issues(params))


def solve(params: Params) -> tuple[str, Expr, GeometrySpec]:
    """Return the question, the exact area, and what to draw."""
    import sympy as sp

    points = parse_params(PARAM_TYPES, params)["points"]
    answer = sp.simplify(exact_area(points))
    stem = "Find the area of the shaded region."

    float_points = [[float(x), float(y)] for x, y in points]

    spec: list[dict[str, Any]] = [
        {"kind": "axes", "step": 1.0},
        {"kind": "polygon", "id": "figure", "points": float_points, "role": "outer"},
        {"kind": "region", "operation": "union", "of": ["figure"], "style": "shaded"},
    ]

    # The first LABELLED_VERTICES vertices keep their coordinates; the rest are
    # marked but unlabelled, and must be counted off the grid.
    for index, (x, y) in enumerate(points):
        spec.append(
            {
                "kind": "point_label",
                "point": [float(x), float(y)],
                "text": f"({x}, {y})",
                "draw": index < LABELLED_VERTICES,
            }
        )
    return stem, answer, spec
