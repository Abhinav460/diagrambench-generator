"""Composite figures: an L-shaped outline whose area needs decomposition.

Covers the paper's *composite figures* configuration. The figure is a rectangle
with a rectangular notch cut from its top-right corner, drawn as a single six-sided
outline -- not as two overlapping pieces, since showing the construction would give
away the decomposition that is the task.

The withheld quantities
-----------------------
Four of the six edges carry their length. The two that do not -- the short vertical
below the notch and the top edge left of it -- are each the difference of two
labelled edges::

    unlabelled vertical = height - notch_height
    unlabelled top      = width  - notch_width

This is the ``unlabeled_inferrable`` trap from the spec: a length that must be
derived sits directly adjacent to one that is given, inviting a model to assume
they are equal. It probes measurement misassignment, and it is the reason this
family labels four edges rather than all six. Labelling all six would leave nothing
to infer; labelling two would make the problem underdetermined rather than merely
hard.

The area is exact and integral::

    A = width * height - notch_width * notch_height

Integral answers are unusual in this package -- the polygon families produce
radicals and cotangents -- which is itself useful. A benchmark whose answers all
look alike lets a model pattern-match the *form* of an answer rather than derive it.
"""

from __future__ import annotations

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

NAME = "composite_rectilinear"

#: All four are lengths. Random draws use integers, but nothing in the geometry
#: needs them: a 12.5 by 6 figure with a 5 by 2.5 notch is an ordinary L.
PARAM_TYPES = {"width": "length", "height": "length", "notch_w": "length", "notch_h": "length"}

Length = int | Fraction

#: Outer dimensions, in the same integer units the labels display.
MIN_SIDE = 4
MAX_SIDE = 20

#: The notch must leave a genuine L: big enough to see, small enough that the
#: remaining arms do not degenerate into slivers. Expressed as a fraction of the
#: edge it is cut from.
MIN_NOTCH_FRACTION = 0.25
MAX_NOTCH_FRACTION = 0.75


def _outline(width: Length, height: Length, notch_w: Length, notch_h: Length) -> list[list[float]]:
    """The six vertices of the L, counter-clockwise from the origin.

    Ordered counter-clockwise so the shoelace area is positive and the outline
    winds the same way as every other family's polygons, which is what lets the
    renderer treat them all identically.
    """
    w, h, a, b = float(width), float(height), float(notch_w), float(notch_h)
    return [
        [0.0, 0.0],
        [w, 0.0],
        [w, h - b],
        [w - a, h - b],
        [w - a, h],
        [0.0, h],
    ]


def exact_area(width: Length, height: Length, notch_w: Length, notch_h: Length) -> Expr:
    """Outer rectangle minus the notch. Exact, and integral for integer sides."""
    return exact(width) * exact(height) - exact(notch_w) * exact(notch_h)


def sample(rng: RNG) -> Params:
    """Draw outer dimensions, then a notch sized as a fraction of each."""
    #find the use of the random number generator(rng)
    width = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    height = int(rng.integers(MIN_SIDE, MAX_SIDE + 1))
    return {
        "width": width,
        "height": height,
        "notch_w": int(rng.integers(1, width)),
        "notch_h": int(rng.integers(1, height)),
    }


def parameter_space() -> Iterator[Params]:
    """Every parameter set ``sample`` can draw, unfiltered; ``registry.ceiling`` counts
    the valid ones so a run can be told when it asks for more than the family has."""
    for width in range(MIN_SIDE, MAX_SIDE + 1):
        for height in range(MIN_SIDE, MAX_SIDE + 1):
            for notch_w in range(1, width):
                for notch_h in range(1, height):
                    yield {"width": width, "height": height, "notch_w": notch_w, "notch_h": notch_h}


def validation_issues(params: Params) -> Iterator[Issue]:
    """Every problem with ``params``, by severity, in the order ``is_valid`` checks.

    Geometry: positive outer dimensions, and a positive notch strictly inside them
    -- anything else is not an L. Readability: the outer-dimension range and the
    notch fractions, which keep a random draw from producing slivers.
    """
    try:
        values = parse_params(PARAM_TYPES, params)
    except (KeyError, TypeError, ValueError) as exc:
        yield GEOMETRY, f"malformed params: {exc}"
        return
    width, height = values["width"], values["height"]
    notch_w, notch_h = values["notch_w"], values["notch_h"]
    dims = f"{length_text(width)}x{length_text(height)}"

    if width <= 0 or height <= 0:
        yield GEOMETRY, f"outer dimensions {dims} outside [{MIN_SIDE}, {MAX_SIDE}]"
        return
    if not (MIN_SIDE <= width <= MAX_SIDE) or not (MIN_SIDE <= height <= MAX_SIDE):
        yield READABILITY, f"outer dimensions {dims} outside [{MIN_SIDE}, {MAX_SIDE}]"
    if notch_w <= 0 or notch_h <= 0:
        yield GEOMETRY, "notch must have positive dimensions"
        return
    if notch_w >= width or notch_h >= height:
        yield GEOMETRY, (
            f"notch {length_text(notch_w)}x{length_text(notch_h)} is not strictly inside {dims}"
        )
        return

    w_fraction = float(notch_w / width)
    h_fraction = float(notch_h / height)
    if not (MIN_NOTCH_FRACTION <= w_fraction <= MAX_NOTCH_FRACTION):
        yield READABILITY, (
            f"notch width is {w_fraction:.2f} of the figure, outside "
            f"[{MIN_NOTCH_FRACTION}, {MAX_NOTCH_FRACTION}]"
        )
    if not (MIN_NOTCH_FRACTION <= h_fraction <= MAX_NOTCH_FRACTION):
        yield READABILITY, (
            f"notch height is {h_fraction:.2f} of the figure, outside "
            f"[{MIN_NOTCH_FRACTION}, {MAX_NOTCH_FRACTION}]"
        )


def is_valid(params: Params) -> ValidationResult:
    """Reject notches that leave no L, or that swallow the figure."""
    return first_issue(validation_issues(params))


def solve(params: Params) -> tuple[str, Expr, GeometrySpec]:
    """Return the question, the exact area, and what to draw."""
    import sympy as sp

    values = parse_params(PARAM_TYPES, params)
    width, height = values["width"], values["height"]
    notch_w, notch_h = values["notch_w"], values["notch_h"]

    answer = sp.simplify(exact_area(width, height, notch_w, notch_h))
    stem = "Find the area of the shaded figure."

    points = _outline(width, height, notch_w, notch_h)

    # Edge i runs from points[i] to points[i+1]. The two withheld edges are the
    # ones whose length is a difference of two labelled edges.
    edges = [
        (width, True),           # bottom
        (height - notch_h, False),  # withheld: height - notch_h
        (notch_w, True),         # notch horizontal
        (notch_h, True),         # notch vertical
        (width - notch_w, False),   # withheld: width - notch_w
        (height, True),          # left
    ]

    spec: list[dict[str, Any]] = [
        {"kind": "polygon", "id": "figure", "points": points, "role": "outer"},
        {"kind": "region", "operation": "union", "of": ["figure"], "style": "shaded"},
    ]

    for i, (length, draw) in enumerate(edges):
        spec.append(
            {
                "kind": "length_label",
                "segment": [points[i], points[(i + 1) % len(points)]],
                "value": float(length),
                "text": length_text(length),
                "draw": draw,
                "owner": "figure",
            }
        )
    return stem, answer, spec
