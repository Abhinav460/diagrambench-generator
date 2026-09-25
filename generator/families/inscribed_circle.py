"""Circular regions: a circle inscribed in a regular polygon, tangent to every side.

Covers the paper's *circular regions* configuration, and targets the error category
its taxonomy ranks second-largest -- circle and tangency confusion, over 20% of all
errors across the three models tested.

Tangency is the point. The circle touches each side at exactly one point, so the
radius is not drawn and cannot be measured off the figure: it must be *inferred*
from the tangency, as the polygon's apothem. A model that reads the circle as
merely "inside" the polygon rather than inscribed in it has no way to recover the
radius, which is precisely the confusion the taxonomy names.

What is given and what is withheld
----------------------------------
Given: the side length, labelled on every side.
Withheld: the side count (countable only from the figure) and the radius (derivable
only from the tangency). Two implicit quantities rather than one, which makes this
family a harder perceptual task than ``nested_polygons``.

The answer is the area between the polygon and the circle::

    A = (k/4) * s^2 * cot(pi/k)  -  pi * (s / (2 * tan(pi/k)))^2

Both terms carry ``cot(pi/k)``, so the answer stays exact but is rarely a radical:
the circle contributes a ``pi`` and the polygon a cotangent, and they do not
combine. That is the same shape as the benchmark's real answers.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from numpy.random import Generator as RNG
    from sympy import Expr
else:
    RNG = Any
    Expr = Any

from generator.params import exact, length_text, parse_params, parse_value
from generator.registry import (
    GEOMETRY,
    READABILITY,
    GeometrySpec,
    Issue,
    Params,
    ValidationResult,
    first_issue,
)

NAME = "inscribed_circle"

#: Side count is a count; the side is a length, and may be any positive real.
PARAM_TYPES = {"k": "integer", "side": "length"}

#: Side-count bounds. The lower bound is 3; the upper is held below the point where
#: a polygon and its inscribed circle become visually indistinguishable, which would
#: leave the shaded ring too thin to read as a region at all.
MIN_SIDES = 3
MAX_SIDES = 8

MIN_SIDE_LENGTH = 1
MAX_SIDE_LENGTH = 20

#: Smallest fraction of the polygon the ring may occupy and still be readable.
#: 5% admits k up to 8 and rejects nothing below it, so it is a backstop against a
#: future widening of MAX_SIDES rather than an active constraint today.
MIN_RING_FRACTION = 0.05


def apothem(k: int, side: float) -> float:
    """Centre-to-edge distance, which for an inscribed circle is the radius."""
    return side / (2 * math.tan(math.pi / k))


def circumradius(k: int, side: float) -> float:
    return side / (2 * math.sin(math.pi / k))


def _vertices(k: int, side: float) -> list[list[float]]:
    """Vertices of a regular k-gon, flat side down, centred on the origin."""
    radius = circumradius(k, side)
    start = -math.pi / 2 + math.pi / k
    return [
        [
            radius * math.cos(start + 2 * math.pi * i / k),
            radius * math.sin(start + 2 * math.pi * i / k),
        ]
        for i in range(k)
    ]


def ring_fraction(params: Params) -> float:
    """Fraction of the polygon's area left uncovered by the inscribed circle.

    The readability measure for this family. The inscribed circle covers
    ``pi / (k * tan(pi/k))`` of the polygon, so the ring is what is left. It depends
    only on the side count and shrinks fast as the polygon rounds off: 39.5% at
    k=3, 5.2% at k=8, 2.3% at k=12. That decay is what the side-count ceiling and
    ``MIN_RING_FRACTION`` are protecting against -- past a point the ring is a
    hairline and the figure reads as a circle with a faint outline.
    """
    k = parse_value("integer", "k", params["k"])
    return 1 - math.pi / (k * math.tan(math.pi / k))


def exact_area(k: int, side: int | Fraction) -> Expr:
    """Exact area of a regular k-gon: (k/4) * s^2 * cot(pi/k)."""
    import sympy as sp

    return sp.Rational(k, 4) * exact(side) ** 2 * sp.cot(sp.pi / sp.Integer(k))


def exact_circle_area(k: int, side: int | Fraction) -> Expr:
    """Exact area of the inscribed circle: pi * apothem^2, apothem = s/(2 tan(pi/k))."""
    import sympy as sp

    r = exact(side) / (2 * sp.tan(sp.pi / sp.Integer(k)))
    return sp.pi * r ** 2


def sample(rng: RNG, pinned: Optional[Params] = None) -> Params:
    """Draw a side count and an integer side length.

    A key in ``pinned`` is taken as given and not drawn, so an empty ``pinned``
    consumes ``rng`` exactly as a plain draw does.
    """
    pinned = pinned or {}
    k = pinned["k"] if "k" in pinned else int(rng.integers(MIN_SIDES, MAX_SIDES + 1))
    side = (
        pinned["side"]
        if "side" in pinned
        else int(rng.integers(MIN_SIDE_LENGTH, MAX_SIDE_LENGTH + 1))
    )
    return {"k": k, "side": side}


def parameter_space(pinned: Optional[Params] = None) -> Iterator[Params]:
    """Every parameter set ``sample`` can draw with ``pinned``, unfiltered;
    ``registry.ceiling`` counts the valid ones so a run can be told when it asks for
    more than the family has."""
    pinned = pinned or {}
    for k in [pinned["k"]] if "k" in pinned else range(MIN_SIDES, MAX_SIDES + 1):
        for side in (
            [pinned["side"]] if "side" in pinned else range(MIN_SIDE_LENGTH, MAX_SIDE_LENGTH + 1)
        ):
            yield {"k": k, "side": side}


def validation_issues(params: Params) -> Iterator[Issue]:
    """Every problem with ``params``, by severity, in the order ``is_valid`` checks.

    Geometry: a real side count of at least 3 and a positive side. Readability: the
    side-count ceiling and the ring fraction, both of which exist to keep the ring
    visible rather than to keep the figure well-formed.
    """
    try:
        values = parse_params(PARAM_TYPES, params)
    except (KeyError, TypeError, ValueError) as exc:
        yield GEOMETRY, f"malformed params: {exc}"
        return
    k, side = values["k"], values["side"]

    if k < MIN_SIDES:
        yield GEOMETRY, f"side count {k} outside [{MIN_SIDES}, {MAX_SIDES}]"
        return
    if k > MAX_SIDES:
        yield READABILITY, f"side count {k} outside [{MIN_SIDES}, {MAX_SIDES}]"
    if side <= 0:
        yield GEOMETRY, "side length must be positive"
        return

    fraction = ring_fraction(params)
    if fraction < MIN_RING_FRACTION:
        yield READABILITY, (
            f"ring too thin to read (occupies {fraction:.3f} of the polygon, "
            f"below {MIN_RING_FRACTION})"
        )


def is_valid(params: Params) -> ValidationResult:
    """Reject malformed or visually unreadable configurations."""
    return first_issue(validation_issues(params))


def solve(params: Params) -> tuple[str, Expr, GeometrySpec]:
    """Return the question, the exact ring area, and what to draw."""
    import sympy as sp

    values = parse_params(PARAM_TYPES, params)
    k, side = values["k"], values["side"]

    answer = sp.simplify(exact_area(k, side) - exact_circle_area(k, side))
    stem = "Find the area of the shaded region."

    points = _vertices(k, float(side))
    radius = apothem(k, float(side))

    spec: list[dict[str, Any]] = [
        {"kind": "polygon", "id": "outer", "points": points, "role": "outer"},
        {
            "kind": "circle",
            "id": "hole",
            "center": [0.0, 0.0],
            "radius": radius,
            "role": "inner",
        },
        {
            "kind": "region",
            "operation": "difference",
            "of": ["outer", "hole"],
            "style": "shaded",
        },
    ]

    for i in range(k):
        start, end = points[i], points[(i + 1) % k]
        spec.append(
            {
                "kind": "length_label",
                "segment": [start, end],
                "value": float(side),
                "text": length_text(side),
                "draw": True,
                "owner": "outer",
            }
        )
    return stem, answer, spec
