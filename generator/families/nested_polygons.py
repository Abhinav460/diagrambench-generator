"""Nested regular polygons: a smaller regular polygon centered inside a larger one,
with the annular region between them shaded.

Chosen as the first family because the parameter space is small and discrete, so
every edge case surfaces immediately rather than after a thousand draws.

The geometry_spec format
------------------------
This module is the first producer of a ``geometry_spec``, so the format is defined
here and consumed by both ``render`` (Stage 4) and ``verify`` (Stage 5). A spec is
a flat list of primitive dicts, each with a ``kind``:

``{"kind": "polygon", "id": str, "points": [[x, y], ...], "role": str}``
    A closed polygon in figure coordinates. ``id`` is referenced by regions.

``{"kind": "region", "operation": "difference", "of": [outer_id, inner_id],
   "style": "shaded"}``
    A derived area, described *declaratively* rather than as precomputed geometry.
    This is the primitive that makes Stage 5 work: ``verify`` rebuilds the region
    in shapely from the polygon coordinates in this same spec and compares its
    numeric area against the symbolic answer. Because the operation is declared
    rather than baked in, ``verify`` needs no per-family knowledge, and a family
    that emits coordinates disagreeing with its own algebra is caught.

``{"kind": "length_label", "segment": [[x1, y1], [x2, y2]], "value": float,
   "text": str, "draw": bool, "owner": str}``
    A measurement annotation attached to a segment. ``draw`` carries the labeling
    policy: ``True`` means the renderer prints the value, ``False`` means the
    quantity exists in the geometry but is deliberately withheld, forcing the
    solver to recover it from the figure. ``owner`` names the polygon the segment
    belongs to, which is what lets the renderer decide a side to offset the text
    toward without having to infer ownership from coordinates.

Quantities with no ``length_label`` at all -- here, the side counts ``n`` and ``m``
-- are *implicit*: they are readable only by inspecting the drawing. That is the
diagram-dependence property the benchmark measures.

Design decisions
----------------
Labeling policy: every side of both polygons carries its length (``LABEL_EVERY_SIDE``).
Since both polygons are regular, this repeats one value per polygon rather than
adding information. Side counts stay implicit.

Answers are left as exact sympy expressions in whatever form they take. Regular
polygon area carries a ``cot(pi/k)`` factor, which reduces to a radical only for
k in {3, 4, 6, 8, 12}; for k = 9 it stays as ``cot(pi/9)``, which is exact and
evaluable but not a radical. Restricting k to the radical cases would collapse the
space to about ten (n, m) pairs.

Side lengths are integers, so answers read as a rational coefficient on an exact
trig term -- the same shape as the benchmark's real answers ("The area is
8 + 4*sqrt(3) cm^2") -- without forcing an integral decimal, which regular-polygon
areas essentially never have.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.random import Generator as RNG
    from sympy import Expr
else:
    RNG = Any
    Expr = Any

from generator.params import exact, length_text, parse_params
from generator.registry import (
    GEOMETRY,
    READABILITY,
    GeometrySpec,
    Issue,
    Params,
    ValidationResult,
    first_issue,
)

NAME = "nested_polygons"

#: Side counts are counts; sides are lengths. Random draws use integer sides, but
#: the geometry accepts any positive real.
PARAM_TYPES = {"n": "integer", "m": "integer", "side_outer": "length", "side_inner": "length"}

#: Side-count bounds. The upper bound keeps side-counting a reasonable visual task;
#: a 20-gon is indistinguishable from a circle at figure scale, which would make the
#: implicit quantity unreadable rather than merely unstated.
MIN_SIDES = 3
MAX_SIDES = 12

#: Integer side lengths, per the sampling decision.
MIN_SIDE_LENGTH = 1
MAX_SIDE_LENGTH = 20

#: Containment margins, expressed as (inner circumradius) / (outer apothem).
#: The upper bound keeps the shaded annulus from collapsing into a hairline the
#: reader cannot resolve; the lower bound keeps the inner polygon from shrinking to
#: an indistinct dot. Both are readability constraints standing in for the spec's
#: "resulting region non-degenerate".
MAX_FILL_RATIO = 0.90
MIN_FILL_RATIO = 0.15

#: Labeling policy switch. Flip to False for one label per polygon.
LABEL_EVERY_SIDE = True


# --- geometry helpers -----------------------------------------------------


def _start_angle(k: int) -> float:
    """Angle of the first vertex, chosen so every polygon rests on a flat edge.

    A flat bottom reads as deliberate rather than arbitrarily rotated, and fixing
    it here keeps orientation out of the parameter space -- two records differing
    only by rotation would be near-duplicates carrying no new visual information.
    """
    return -math.pi / 2 + math.pi / k


def _vertices(k: int, side: float) -> list[list[float]]:
    """Vertices of a regular k-gon with the given side length, centered on origin."""
    radius = circumradius(k, side)
    start = _start_angle(k)
    return [
        [radius * math.cos(start + 2 * math.pi * i / k), radius * math.sin(start + 2 * math.pi * i / k)]
        for i in range(k)
    ]


def circumradius(k: int, side: float) -> float:
    """Center-to-vertex distance. The inner polygon's binding dimension."""
    return side / (2 * math.sin(math.pi / k))


def apothem(k: int, side: float) -> float:
    """Center-to-edge-midpoint distance. The outer polygon's binding dimension."""
    return side / (2 * math.tan(math.pi / k))


def fill_ratio(params: Params) -> float:
    """How much of the outer polygon's inradius the inner polygon consumes.

    Comparing the inner *circumradius* against the outer *apothem* is deliberately
    conservative: it guarantees containment at any relative rotation, so validity
    never depends on the orientation convention chosen in ``_start_angle``.
    """
    values = parse_params(PARAM_TYPES, params)
    inner = circumradius(values["m"], float(values["side_inner"]))
    outer = apothem(values["n"], float(values["side_outer"]))
    return inner / outer


def inner_fits(n: int, m: int, side_outer: int | Fraction, side_inner: int | Fraction) -> bool:
    """Whether the inner polygon, as drawn, lies inside the outer one.

    The exact containment test, in the orientation ``_vertices`` actually draws.
    ``fill_ratio <= 1`` guarantees containment at any rotation and so is sufficient
    but not necessary: a hexagon inside a square can fit with a ratio above 1. The
    answer ``area(outer) - area(inner)`` is correct exactly when this holds, so it
    is the geometry check; the ratio remains a readability check.

    The outer polygon is convex and wound counter-clockwise, so a point is inside
    when it is on the left of (or on) every edge. Touching the boundary counts as
    fitting: the shaded region's area is unchanged.
    """
    outer = _vertices(n, float(side_outer))
    inner = _vertices(m, float(side_inner))
    tolerance = 1e-9 * circumradius(n, float(side_outer)) ** 2
    for i in range(n):
        (ax, ay), (bx, by) = outer[i], outer[(i + 1) % n]
        for px, py in inner:
            if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < -tolerance:
                return False
    return True


def exact_area(k: int, side: int | Fraction) -> Expr:
    """Exact area of a regular k-gon: (k/4) * s^2 * cot(pi/k)."""
    import sympy as sp

    return sp.Rational(k, 4) * exact(side) ** 2 * sp.cot(sp.pi / sp.Integer(k))


# --- family contract ------------------------------------------------------


def sample(rng: RNG) -> Params:
    """Draw side counts and integer side lengths, unfiltered.

    Draws freely and lets ``is_valid`` reject, rather than sampling from the
    constrained space directly. That is the right tradeoff only while the
    acceptance rate stays high; if it drops, the fix is a smarter draw here, not a
    looser predicate there.
    """
    n = int(rng.integers(MIN_SIDES + 1, MAX_SIDES + 1))
    m = int(rng.integers(MIN_SIDES, n))
    return {
        "n": n,
        "m": m,
        "side_outer": int(rng.integers(MIN_SIDE_LENGTH, MAX_SIDE_LENGTH + 1)),
        "side_inner": int(rng.integers(MIN_SIDE_LENGTH, MAX_SIDE_LENGTH + 1)),
    }


def parameter_space() -> Iterator[Params]:
    """Every parameter set ``sample`` can draw, unfiltered; ``registry.ceiling`` counts
    the valid ones so a run can be told when it asks for more than the family has."""
    for n in range(MIN_SIDES + 1, MAX_SIDES + 1):
        for m in range(MIN_SIDES, n):
            for side_outer in range(MIN_SIDE_LENGTH, MAX_SIDE_LENGTH + 1):
                for side_inner in range(MIN_SIDE_LENGTH, MAX_SIDE_LENGTH + 1):
                    yield {"n": n, "m": m, "side_outer": side_outer, "side_inner": side_inner}


def validation_issues(params: Params) -> Iterator[Issue]:
    """Every problem with ``params``, by severity, in the order ``is_valid`` checks.

    Geometry: side counts of at least 3, positive sides, and an inner polygon that
    actually lies inside the outer one. Readability: the side-count ceiling, the
    inner-fewer-sides rule, and both fill-ratio margins. A square inside a hexagon
    fails none of the geometry, and a real problem may well draw one.
    """
    try:
        values = parse_params(PARAM_TYPES, params)
    except (KeyError, TypeError, ValueError) as exc:
        yield GEOMETRY, f"malformed params: {exc}"
        return
    n, m = values["n"], values["m"]
    side_outer, side_inner = values["side_outer"], values["side_inner"]

    for label, count in (("outer", n), ("inner", m)):
        reason = f"{label} side count {count} outside [{MIN_SIDES}, {MAX_SIDES}]"
        if count < MIN_SIDES:
            yield GEOMETRY, reason
            return
        if count > MAX_SIDES:
            yield READABILITY, reason
    if m >= n:
        yield READABILITY, f"inner side count {m} is not less than outer {n}"
    if side_outer <= 0 or side_inner <= 0:
        yield GEOMETRY, "side lengths must be positive"
        return

    ratio = fill_ratio(params)
    clearance_reason = (
        f"inner polygon does not fit with clearance (fill ratio {ratio:.3f} > {MAX_FILL_RATIO})"
    )
    if not inner_fits(n, m, side_outer, side_inner):
        yield GEOMETRY, clearance_reason
        return
    if ratio > MAX_FILL_RATIO:
        yield READABILITY, clearance_reason
    if ratio < MIN_FILL_RATIO:
        yield READABILITY, (
            f"inner polygon too small to read (fill ratio {ratio:.3f} < {MIN_FILL_RATIO})"
        )


def is_valid(params: Params) -> ValidationResult:
    """Reject configurations that are malformed, non-containing, or unreadable.

    Returns a specific reason string on every rejection path so that a low
    acceptance rate can be attributed to a cause instead of guessed at.
    """
    return first_issue(validation_issues(params))


def solve(params: Params) -> tuple[str, Expr, GeometrySpec]:
    """Return the question, the exact shaded area, and what to draw."""
    values = parse_params(PARAM_TYPES, params)
    n, m = values["n"], values["m"]
    side_outer, side_inner = values["side_outer"], values["side_inner"]

    import sympy as sp

    answer = sp.simplify(exact_area(n, side_outer) - exact_area(m, side_inner))
    stem = "Find the area of the shaded region."

    outer_points = _vertices(n, float(side_outer))
    inner_points = _vertices(m, float(side_inner))

    spec: list[dict[str, Any]] = [
        {"kind": "polygon", "id": "outer", "points": outer_points, "role": "outer"},
        {"kind": "polygon", "id": "inner", "points": inner_points, "role": "inner"},
        {
            "kind": "region",
            "operation": "difference",
            "of": ["outer", "inner"],
            "style": "shaded",
        },
    ]
    spec.extend(_length_labels(outer_points, side_outer, owner="outer"))
    spec.extend(_length_labels(inner_points, side_inner, owner="inner"))
    return stem, answer, spec


def _length_labels(points: list[list[float]], side: int | Fraction, owner: str) -> list[dict[str, Any]]:
    """Emit one label per side, or a single label, per ``LABEL_EVERY_SIDE``.

    All sides of a regular polygon are equal, so labelling every one repeats a
    value rather than revealing a new one; the switch exists so the policy can be
    revisited against rendered output without touching the geometry code.
    """
    count = len(points) if LABEL_EVERY_SIDE else 1
    labels = []
    for i in range(count):
        start = points[i]
        end = points[(i + 1) % len(points)]
        labels.append(
            {
                "kind": "length_label",
                "segment": [start, end],
                "value": float(side),
                "text": length_text(side),
                "draw": True,
                "owner": owner,
            }
        )
    return labels
