"""Independently re-measure a rendered figure and confirm it matches the answer.

The check that matters is not "does the algebra agree with itself" -- it is "does
the thing that will be drawn agree with the thing that will be graded". So this
module rebuilds geometry from the ``geometry_spec``, never from ``params``.

That distinction is the whole point. Rebuilding from ``params`` would re-run the
family's own derivation and agree with it by construction, including when the
family emits coordinates that describe a different figure than its algebra
assumed. Rebuilding from the spec closes that gap: the spec is what ``render``
draws, so measuring it is measuring the actual problem.

Disagreement raises rather than warns. In a run of a thousand problems a warning
is a line nobody reads, and a benchmark with silently wrong ground truth is worse
than no benchmark.

Tolerances
----------
Polygon vertices are exact rationals-of-trig evaluated in double precision, so a
shoelace area agrees with the closed form to near machine epsilon; ``TOLERANCE``
is set at 1e-9 accordingly.

Curved boundaries cannot be exact. Shapely has no true arc, so a circle is a
polygonal approximation via ``buffer(r, quad_segs=N)``, which inscribes 4*N
segments and therefore *understates* area by roughly ``2*pi**2 / (3 * (4N)**2)``.
(The plan calls this argument ``resolution``, which is what shapely named it
before 2.1 deprecated the spelling; ``quad_segs`` is the same quantity.)
At ``ARC_RESOLUTION = 1024`` (4096 segments) that is about 3.9e-7, and specs
containing curved primitives are held to ``ARC_TOLERANCE = 1e-4``.

That margin is deliberately wide, because the relevant error is not relative to
the circle but relative to the *answer*. A shaded ring is a small difference of
two large areas, so the circle's absolute error stays fixed while the quantity it
is compared against shrinks. An inscribed circle in a decagon exceeded 1e-4 at
the previous resolution of 256 for exactly that reason. Reaching 1e-9 would need
N in the millions, which is not worth the cost.

The curved primitive
--------------------
``{"kind": "circle", "id": str, "center": [x, y], "radius": float}``
    A full circle, measured as ``Point(center).buffer(radius, quad_segs=N)``.

No family emits one yet. It is implemented anyway because Stage 5 specifies the
buffer approach, and constants describing a tolerance no code path can reach are
worse than an unused primitive: they read as a capability that exists. Center and
radius are the whole of a circle, so little is being guessed at here -- but a
family needing arcs or sectors rather than closed circles should extend the format
rather than force itself through this one.
"""

from __future__ import annotations

from typing import Any, Optional

from generator.geometry import (
    ARC_RESOLUTION,
    CURVED_KINDS,
    GeometryError,
    region_geometries,
    spec_has_curve,
)
from generator.registry import GeometrySpec, Params

__all__ = [
    "VerificationError",
    "verify",
    "measure",
    "relative_difference",
    "TOLERANCE",
    "ARC_TOLERANCE",
    "ARC_RESOLUTION",
    "CURVED_KINDS",
]

#: Relative tolerance for straight-edged figures.
TOLERANCE = 1e-9

#: Relative tolerance once a curved boundary is involved. See the module docstring.
ARC_TOLERANCE = 1e-4


class VerificationError(AssertionError):
    """Raised when the drawn geometry and the symbolic answer disagree.

    An AssertionError subclass because it signals a broken invariant in our own
    code -- a family whose figure and algebra have drifted apart -- rather than
    bad input from outside.
    """


def measure(spec: GeometrySpec) -> float:
    """Numerically measure the quantity a spec's regions describe.

    Each ``region`` primitive may carry ``measure``: ``"area"`` (the default) or
    ``"length"``. Multiple regions sum, so a family that shades two disjoint
    pieces and asks for the total needs no special handling here.

    Note that a spec containing any curved primitive is only accurate to
    ``ARC_TOLERANCE``, not ``TOLERANCE``; ``verify`` picks the right one, but a
    caller using ``measure`` directly is responsible for that itself.
    """
    try:
        regions = region_geometries(spec)
    except GeometryError as exc:
        # Translated so a malformed spec surfaces as a verification failure rather
        # than as an unrelated exception type escaping this module.
        raise VerificationError(str(exc)) from exc

    if not regions:
        raise VerificationError("spec declares no region, so there is nothing to measure")

    total = 0.0
    for region, shape in regions:
        quantity = region.get("measure", "area")
        if quantity == "area":
            total += shape.area
        elif quantity == "length":
            total += shape.length
        else:
            raise VerificationError(f"unknown region measure {quantity!r}")

    return total


def relative_difference(measured: float, expected: float) -> float:
    """Relative difference, floored at 1 so a near-zero expectation cannot blow up."""
    return abs(measured - expected) / max(abs(expected), 1.0)


def verify(
    spec: GeometrySpec,
    answer: Any,
    *,
    params: Optional[Params] = None,
    tolerance: Optional[float] = None,
) -> float:
    """Confirm the spec's geometry measures what ``answer`` says, or raise.

    Returns the relative difference so a driver can report the worst case across a
    run; a run whose maximum sits just under tolerance is worth knowing about even
    though nothing failed.
    """
    expected = float(answer.evalf()) if hasattr(answer, "evalf") else float(answer)
    measured = measure(spec)
    difference = relative_difference(measured, expected)

    if tolerance is None:
        tolerance = ARC_TOLERANCE if spec_has_curve(spec) else TOLERANCE

    if difference >= tolerance:
        raise VerificationError(
            f"geometry and answer disagree for params={dict(params) if params else '<unknown>'}: "
            f"measured={measured!r} from the spec, expected={expected!r} from the answer "
            f"({answer!r}), relative difference={difference:.3e} >= tolerance={tolerance:.1e}"
        )
    return difference
