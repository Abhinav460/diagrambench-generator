"""Typed reading of a family's params, so no family truncates what it was given.

Every family used to read its params with ``int(params["side"])``. That is harmless
for a random draw, which only ever produces ints, and wrong for anything else:
``int(2.5)`` is ``2``, so a problem with a 2.5 side was solved -- and verified,
since the figure was built from the same truncated value -- as a problem with a
2 side. Verification cannot catch this, because it checks the figure against the
answer, and both were computed from the same wrong number.

So each family declares ``PARAM_TYPES``, a mapping from param name to one of the
kinds below, and reads its params through ``parse_params``. A value of the wrong
kind raises ``ParamTypeError`` instead of being coerced.

The kinds
---------
``"integer"``
    A count, such as a side count. Must be an ``int`` (not a ``bool``, and not a
    float, even ``6.0``): a polygon with 6.5 sides has no meaning to recover.
``"length"``
    A real length. An ``int`` or a finite ``float``, returned
    as an exact ``Fraction``. A float is read through its shortest ``repr``, so
    ``0.1`` becomes ``1/10`` rather than the binary approximation -- a length typed
    as 0.1 means one tenth, and the exact answer should say so.
``"lattice_points"``
    A list of ``[x, y]`` pairs of ``"integer"`` coordinates.

Stdlib only, like ``schema``, so a loader can type-check a file without importing
sympy. ``exact`` is the one exception, and imports sympy lazily.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sympy import Expr
else:
    Expr = Any

__all__ = [
    "PARAM_KINDS",
    "ParamTypeError",
    "exact",
    "length_text",
    "parse_params",
    "parse_value",
]

PARAM_KINDS = ("integer", "length", "lattice_points")


class ParamTypeError(ValueError):
    """A param is present but of the wrong kind.

    A ``ValueError`` so the families' existing ``except (KeyError, TypeError,
    ValueError)`` malformed-params paths report it without further changes.
    """


def _integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParamTypeError(f"{name} must be an integer; got {value!r}")
    return value


def _length(name: str, value: Any) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParamTypeError(f"{name} must be a number; got {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ParamTypeError(f"{name} must be finite; got {value!r}")
        return Fraction(repr(value))
    return Fraction(value)


def _lattice_points(name: str, value: Any) -> list[list[int]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ParamTypeError(f"{name} must be a list of [x, y] pairs; got {value!r}")
    points = []
    for index, point in enumerate(value):
        if isinstance(point, (str, bytes)) or not isinstance(point, Sequence) or len(point) != 2:
            raise ParamTypeError(f"{name}[{index}] must be an [x, y] pair; got {point!r}")
        points.append([_integer(f"{name}[{index}] x", point[0]), _integer(f"{name}[{index}] y", point[1])])
    return points


_PARSERS = {"integer": _integer, "length": _length, "lattice_points": _lattice_points}


def parse_value(kind: str, name: str, value: Any) -> Any:
    """Check one value against its declared kind and return it in working form."""
    try:
        parser = _PARSERS[kind]
    except KeyError:
        raise ValueError(f"unknown param kind {kind!r} for {name!r}; known: {PARAM_KINDS}") from None
    return parser(name, value)


def parse_params(param_types: Mapping[str, str], params: Mapping[str, Any]) -> dict[str, Any]:
    """Read every declared param, in declaration order.

    A missing key raises ``KeyError(key)`` -- the same exception the families'
    ``params["n"]`` lookups raised, so a malformed-params message is unchanged.
    Undeclared extra keys are ignored here, as they always were by the families;
    rejecting them is the job of the structured-input loader and the schema, which
    are the places a human-written row enters.
    """
    return {name: parse_value(kind, name, params[name]) for name, kind in param_types.items()}


def exact(value: int | Fraction) -> Expr:
    """An int or ``Fraction`` as an exact sympy number. ``exact(10)`` is ``Integer(10)``."""
    import sympy as sp

    if isinstance(value, Fraction):
        return sp.Rational(value.numerator, value.denominator)
    return sp.Integer(value)


def length_text(value: int | Fraction) -> str:
    """How a length is printed on a figure: ``10``, ``2.5``, or ``1/3``.

    Integral lengths print exactly as ``str(int)`` did before lengths became
    fractions, which keeps the labels -- and so the rendered images -- of every
    random draw byte-identical.
    """
    fraction = Fraction(value)
    if fraction.denominator == 1:
        return str(fraction.numerator)

    denominator = fraction.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if denominator != 1:
        return f"{fraction.numerator}/{fraction.denominator}"

    # A denominator of only 2s and 5s has a terminating decimal expansion.
    with localcontext() as ctx:
        ctx.prec = 60
        text = format(Decimal(fraction.numerator) / Decimal(fraction.denominator), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text
