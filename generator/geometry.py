"""Turn a ``geometry_spec`` into shapely geometry.

Extracted so that ``render`` and ``verify`` build their shapes from exactly the
same code. They ask different questions of the result -- one fills it, the other
measures it -- but if they disagreed about what a spec *means*, verification would
be checking a figure other than the one drawn, which is the failure the whole
verification step exists to prevent.

That does not weaken the check. The independent side of the comparison is the
family's sympy answer; the spec is the drawn figure, and both consumers must read
it identically.

Primitives
----------
``{"kind": "polygon", "id": str, "points": [[x, y], ...]}``
``{"kind": "circle", "id": str, "center": [x, y], "radius": float}``
    Buildable shapes, addressed by ``id``.

``{"kind": "region", "operation": "difference" | "union" | "intersection",
   "of": [id, ...]}``
    A derived shape. ``difference`` subtracts the union of everything after the
    first from the first; ``union`` and ``intersection`` fold across all of them.
    Declared rather than precomputed, so a family states what it means and both
    consumers derive the same geometry from that statement.

Circles are polygonal approximations -- shapely has no true arc. See
``verify.ARC_TOLERANCE`` for what that costs.
"""

from __future__ import annotations

from typing import Any, Optional

from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from generator.registry import GeometrySpec

__all__ = [
    "BUILDABLE_KINDS",
    "CURVED_KINDS",
    "GeometryError",
    "OPERATIONS",
    "build_shapes",
    "region_geometries",
    "spec_has_curve",
]

#: Primitive kinds that become a shapely object.
BUILDABLE_KINDS = frozenset({"polygon", "circle"})

#: Kinds that introduce curvature, and so relax numeric tolerance downstream.
#: ``arc`` and ``sector`` are named but unbuildable: an arc needs start and end
#: angles and a winding sense, which is a real format decision belonging to the
#: first family that needs one. A circle is center plus radius and guesses nothing.
CURVED_KINDS = frozenset({"arc", "circle", "sector"})

#: Supported region operations.
OPERATIONS = frozenset({"difference", "union", "intersection"})

#: Segments per quarter circle when buffering a circle into a polygon.
#: 1024 (4096 segments) rather than 256, because a shaded *ring* is a small
#: difference of two large areas: the circle's absolute approximation error is
#: unchanged but the answer it is measured against is much smaller, so the
#: relative error is amplified. At 256 an inscribed circle in a decagon exceeded
#: ARC_TOLERANCE outright. Error falls as 1/N^2, so this buys a factor of 16.
ARC_RESOLUTION = 1024


class GeometryError(ValueError):
    """Raised when a spec cannot be turned into geometry.

    Callers translate this into their own error type -- ``RenderError`` or
    ``VerificationError`` -- so that a malformed spec is reported in the terms of
    whichever stage found it.
    """


def spec_has_curve(spec: GeometrySpec) -> bool:
    """Whether any primitive in the spec has a curved boundary."""
    return any(item.get("kind") in CURVED_KINDS for item in spec)


def build_shapes(spec: GeometrySpec) -> dict[str, BaseGeometry]:
    """Build every addressable shape in the spec, keyed by id.

    Circles are buffered into polygons here rather than at the point of use, so a
    region mixing a circle and a polygon needs no special case: by the time a
    boolean operation runs, everything is the same kind of object.
    """
    out: dict[str, BaseGeometry] = {}
    for item in spec:
        kind = item.get("kind")
        if kind not in BUILDABLE_KINDS:
            if kind in CURVED_KINDS:
                raise GeometryError(
                    f"curved primitive {kind!r} has no defined format yet; "
                    f"buildable curved kinds are {sorted(BUILDABLE_KINDS & CURVED_KINDS)}"
                )
            continue

        pid = str(item.get("id"))
        if pid in out:
            raise GeometryError(f"duplicate shape id {pid!r}")

        if kind == "polygon":
            points = item["points"]
            if len(points) < 3:
                raise GeometryError(f"polygon {pid!r} needs at least three points")
            shape: BaseGeometry = Polygon([(float(x), float(y)) for x, y in points])
            if not shape.is_valid:
                raise GeometryError(
                    f"polygon {pid!r} is not a valid simple polygon (self-intersecting?)"
                )
        else:
            radius = float(item["radius"])
            if radius <= 0:
                raise GeometryError(f"circle {pid!r} has non-positive radius {radius!r}")
            cx, cy = item["center"]
            shape = Point(float(cx), float(cy)).buffer(radius, quad_segs=ARC_RESOLUTION)

        out[pid] = shape
    return out


def _combine(operation: str, parts: list[BaseGeometry]) -> BaseGeometry:
    """Fold a region's operation across its shapes."""
    if operation == "difference":
        return parts[0].difference(unary_union(parts[1:]))
    if operation == "union":
        return unary_union(parts)

    result = parts[0]
    for part in parts[1:]:
        result = result.intersection(part)
    return result


def region_geometries(
    spec: GeometrySpec, shapes: Optional[dict[str, BaseGeometry]] = None
) -> list[tuple[dict[str, Any], BaseGeometry]]:
    """Resolve every ``region`` primitive to concrete geometry, in spec order.

    Returns the region item alongside its shape so callers can read per-region
    settings -- a fill style, a ``measure`` of area versus length -- without
    walking the spec a second time and risking a different pairing.
    """
    if shapes is None:
        shapes = build_shapes(spec)

    out: list[tuple[dict[str, Any], BaseGeometry]] = []
    for item in spec:
        if item.get("kind") != "region":
            continue

        operation = item.get("operation")
        if operation not in OPERATIONS:
            raise GeometryError(
                f"unsupported region operation {operation!r}; expected one of {sorted(OPERATIONS)}"
            )

        ids = [str(i) for i in item.get("of", [])]
        missing = [i for i in ids if i not in shapes]
        if missing:
            raise GeometryError(f"region references undeclared shapes: {missing}")
        # A union of one shape is that shape, which is how a family declares "the
        # region is this single outline" -- a composite figure drawn as one path
        # rather than as its construction pieces. Difference and intersection are
        # genuinely binary, and one operand there means the family forgot one.
        minimum = 1 if operation == "union" else 2
        if len(ids) < minimum:
            raise GeometryError(
                f"{operation} needs at least {minimum} shape(s); got {ids}"
            )

        out.append((dict(item), _combine(operation, [shapes[i] for i in ids])))
    return out
