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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.random import Generator as RNG
    from sympy import Expr
else:
    RNG = Any
    Expr = Any

from generator.registry import GeometrySpec, Params, ValidationResult

NAME = "composite_rectilinear"

#: Outer dimensions, in the same integer units the labels display.
MIN_SIDE = 4
MAX_SIDE = 20

#: The notch must leave a genuine L: big enough to see, small enough that the
#: remaining arms do not degenerate into slivers. Expressed as a fraction of the
#: edge it is cut from.
MIN_NOTCH_FRACTION = 0.25
MAX_NOTCH_FRACTION = 0.75


def _outline(width: int, height: int, notch_w: int, notch_h: int) -> list[list[float]]:
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


def exact_area(width: int, height: int, notch_w: int, notch_h: int) -> Expr:
    """Outer rectangle minus the notch. Exact and integral by construction."""
    import sympy as sp

    return sp.Integer(width) * sp.Integer(height) - sp.Integer(notch_w) * sp.Integer(notch_h)


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


def is_valid(params: Params) -> ValidationResult:
    """Reject notches that leave no L, or that swallow the figure."""
    try:
        width = int(params["width"])
        height = int(params["height"])
        notch_w = int(params["notch_w"])
        notch_h = int(params["notch_h"])
    except (KeyError, TypeError, ValueError) as exc:
        return f"malformed params: {exc}"

    if not (MIN_SIDE <= width <= MAX_SIDE) or not (MIN_SIDE <= height <= MAX_SIDE):
        return f"outer dimensions {width}x{height} outside [{MIN_SIDE}, {MAX_SIDE}]"
    if notch_w < 1 or notch_h < 1:
        return "notch must have positive dimensions"
    if notch_w >= width or notch_h >= height:
        return f"notch {notch_w}x{notch_h} is not strictly inside {width}x{height}"

    w_fraction = notch_w / width
    h_fraction = notch_h / height
    if not (MIN_NOTCH_FRACTION <= w_fraction <= MAX_NOTCH_FRACTION):
        return (
            f"notch width is {w_fraction:.2f} of the figure, outside "
            f"[{MIN_NOTCH_FRACTION}, {MAX_NOTCH_FRACTION}]"
        )
    if not (MIN_NOTCH_FRACTION <= h_fraction <= MAX_NOTCH_FRACTION):
        return (
            f"notch height is {h_fraction:.2f} of the figure, outside "
            f"[{MIN_NOTCH_FRACTION}, {MAX_NOTCH_FRACTION}]"
        )
    return True


def solve(params: Params) -> tuple[str, Expr, GeometrySpec]:
    """Return the question, the exact area, and what to draw."""
    import sympy as sp

    width = int(params["width"])
    height = int(params["height"])
    notch_w = int(params["notch_w"])
    notch_h = int(params["notch_h"])

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
                "text": str(length),
                "draw": draw,
                "owner": "figure",
            }
        )
    return stem, answer, spec
