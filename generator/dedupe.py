"""Decide whether two problems are the same problem.

Near-duplicates are the failure mode that quietly ruins a generated benchmark. A
thousand records that are really two hundred problems in five rotations each
inflates the headline count while measuring far less than it claims, and nothing
about the manifest reveals it -- every record is individually well-formed.

So identity is defined here, once, as a hash of the family name together with the
parameters that produced the problem. Two records with the same signature are the
same problem regardless of which seed drew them.

Why params rather than pixels
-----------------------------
Comparing rendered images would catch visual duplicates that params-based hashing
misses, but it costs a render per candidate and answers a slightly different
question. Params are what the family varies, so params are what "different
problem" means for generated records. (Image-based comparison is still the right
tool against the *existing* screenshot dataset, which has no params at all -- a
separate problem, in a separate package.)

The rounding precision
----------------------
Params are rounded before hashing, because two floats differing at the sixteenth
decimal are the same problem and must not hash apart. ``precision`` is the number
of decimal places kept, and it is an argument rather than a buried constant
because the right value depends on what a family's params mean -- a side length in
centimetres and a ratio in [0, 1] do not want the same granularity.

Both extremes fail, in opposite and asymmetric ways:

- **Too high** (``precision=15``): float noise survives into the digest, so two
  problems that render identically hash apart. Duplicates enter the dataset. This
  failure is silent -- the manifest looks fine and the benchmark is quietly weaker
  than its record count suggests.
- **Too low** (``precision=0``): genuinely distinct problems round together and
  the second is discarded as a duplicate. Yield collapses and the parameter space
  is explored far less than the acceptance rate implies. This failure is at least
  visible, as a collision rate that climbs toward 1.

Prefer erring low-ish. Discarding a real problem costs one draw; admitting a
duplicate costs the benchmark's integrity.

``DEFAULT_PRECISION`` is 6. The only family so far draws integer params, where any
precision at or above 0 behaves identically, so the default is chosen for the
families that come later rather than for this one.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable, Optional

from generator.schema import ParamsInput, canonical_params

__all__ = [
    "DEFAULT_PRECISION",
    "DedupeError",
    "Deduplicator",
    "normalize",
    "signature",
]

#: Decimal places retained when hashing a float param. See the module docstring.
DEFAULT_PRECISION = 6

#: Separates the family name from the payload so that a family called ``"a"`` with
#: a payload of ``"bc"`` cannot collide with family ``"ab"`` and payload ``"c"``.
_SEPARATOR = "\x00"


class DedupeError(ValueError):
    """Raised when a signature cannot be computed from the given params.

    Distinct from ``SchemaValidationError`` because a record can be schema-valid
    and still unhashable here -- a non-finite float passes ``json.dumps`` under
    default settings but has no meaningful rounded form.
    """


def _normalize_value(value: Any, precision: int) -> Any:
    """Round one param value into the form that defines identity.

    Numbers become fixed-point strings so that ``1``, ``1.0`` and ``0.9999999``
    (at precision 6) all normalize to the same token: a family that switches a
    param from int to float has not thereby invented new problems.

    Booleans are tagged separately even though ``bool`` is a subclass of ``int``,
    since ``True`` and ``1`` are the same value to Python but never the same
    parameter to a family.
    """
    if isinstance(value, bool):
        return ["b", value]
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise DedupeError(
                f"cannot hash non-finite param value {value!r}; "
                "a non-finite param means the sampler produced nonsense"
            )
        # Adding 0.0 folds -0.0 into 0.0, which would otherwise format as
        # "-0.000000" and hash apart from an identical problem.
        return ["n", format(round(number, precision) + 0.0, f".{precision}f")]
    if isinstance(value, str):
        return ["s", value]
    if value is None:
        return ["z", None]
    if isinstance(value, (list, tuple)):
        return ["l", [_normalize_value(v, precision) for v in value]]
    raise DedupeError(
        f"param value of type {type(value).__name__} cannot be normalized: {value!r}"
    )


def normalize(params: ParamsInput, *, precision: int = DEFAULT_PRECISION) -> list[Any]:
    """The rounded, sorted structure a signature is computed from.

    Exposed separately from ``signature`` so that a collision can be investigated:
    when two problems hash together unexpectedly, the question is always what they
    normalized to, and a hex digest cannot answer it.
    """
    if precision < 0:
        raise DedupeError(f"precision must be non-negative; got {precision}")
    return [[key, _normalize_value(value, precision)] for key, value in canonical_params(params)]


def signature(
    family: str,
    params: ParamsInput,
    *,
    precision: int = DEFAULT_PRECISION,
) -> str:
    """A stable hex digest identifying this problem within this family.

    Stable across processes and runs -- ``hashlib`` rather than ``hash()``, whose
    string randomization would make a signature meaningless between sessions and
    so unable to dedupe an appended manifest against an existing one.
    """
    if not isinstance(family, str) or not family.strip():
        raise DedupeError(f"family must be a non-empty string; got {family!r}")

    payload = json.dumps(
        normalize(params, precision=precision),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256((family + _SEPARATOR + payload).encode("utf-8")).hexdigest()


class Deduplicator:
    """Tracks signatures seen so far and reports how often draws repeat.

    The collision rate is the reason this is a class rather than a set at the call
    site. A rate near zero means the parameter space is far from exhausted; a rate
    climbing through a run means the sampler is running out of room and the target
    count will not be reached by drawing harder. The driver needs to be able to say
    which, and that requires counting rejections, not just suppressing them.
    """

    def __init__(self, *, precision: int = DEFAULT_PRECISION) -> None:
        self.precision = precision
        self._seen: dict[str, ParamsInput] = {}
        self.considered = 0
        self.collisions = 0

    def __len__(self) -> int:
        return len(self._seen)

    def __contains__(self, sig: str) -> bool:
        return sig in self._seen

    @property
    def signatures(self) -> frozenset[str]:
        """The accepted signatures, for writing a resume marker or an audit."""
        return frozenset(self._seen)

    @property
    def collision_rate(self) -> float:
        """Fraction of considered draws that repeated something already accepted.

        Zero for an empty run rather than undefined: a run that considered nothing
        collided on nothing, and a driver printing a report should not have to
        special-case its first line.
        """
        return self.collisions / self.considered if self.considered else 0.0

    def add(self, family: str, params: ParamsInput) -> Optional[str]:
        """Register a problem, returning its signature, or ``None`` if it repeats.

        Returning the signature rather than ``True`` saves the caller recomputing
        it for the record it is about to build -- the signature is a schema field,
        so every accepted draw needs it anyway.
        """
        self.considered += 1
        sig = signature(family, params, precision=self.precision)
        if sig in self._seen:
            self.collisions += 1
            return None
        self._seen[sig] = canonical_params(params)
        return sig

    def discard(self, sig: str) -> None:
        """Forget an accepted signature whose problem was never emitted.

        For a driver that accepts a problem and then fails to emit it without
        aborting the run: the signature must be released, or a later, correct copy of
        the same problem is reported as a duplicate of one that does not exist.
        """
        self._seen.pop(sig, None)

    def first_with(self, sig: str) -> Optional[ParamsInput]:
        """The params that originally claimed ``sig``, for explaining a collision."""
        return self._seen.get(sig)

    def prime(self, signatures: Iterable[str]) -> None:
        """Seed with signatures from an earlier run, so a resumed or appended run
        does not re-emit problems the manifest already contains.

        Primed signatures have no stored params, so ``first_with`` returns ``None``
        for them; they count toward membership but not toward the collision rate
        until something actually collides with one.
        """
        for sig in signatures:
            self._seen.setdefault(sig, None)  # type: ignore[arg-type]
