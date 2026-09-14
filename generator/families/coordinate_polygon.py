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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.random import Generator as RNG
    from sympy import Expr
else:
    RNG = Any
    Expr = Any

from generator.registry import GeometrySpec, Params, ValidationResult

NAME = "coordinate_polygon"

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


def is_valid(params: Params) -> ValidationResult:
    """Reject degenerate, non-convex, or unreadably small polygons."""
    try:
        points = [[int(x), int(y)] for x, y in params["points"]]
    except (KeyError, TypeError, ValueError) as exc:
        return f"malformed params: {exc}"

    if not (MIN_VERTICES <= len(points) <= MAX_VERTICES):
        return f"vertex count {len(points)} outside [{MIN_VERTICES}, {MAX_VERTICES}]"
    if len({tuple(p) for p in points}) != len(points):
        return "duplicate vertices"
    for x, y in points:
        if not (MIN_COORD <= x <= MAX_COORD and MIN_COORD <= y <= MAX_COORD):
            return f"vertex ({x}, {y}) outside the lattice bounds"

    if not _is_simple_and_convex(points):
        return "vertices are collinear or wind inconsistently (non-convex or self-intersecting)"

    area = shoelace([[float(x), float(y)] for x, y in points])
    if area < MIN_AREA:
        return f"area {area:.1f} is below the readable minimum of {MIN_AREA}"
    return True


def solve(params: Params) -> tuple[str, Expr, GeometrySpec]:
    """Return the question, the exact area, and what to draw."""
    import sympy as sp

    points = [[int(x), int(y)] for x, y in params["points"]]
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
