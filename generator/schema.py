"""The canonical record for a DiagramBench problem, generated or harvested.

This module exists so that every downstream module -- ``render``, ``verify``,
``dedupe``, ``emit``, and later the separate harvest package -- agrees on one
shape for a problem, and so that a malformed record is rejected before any of
the expensive machinery (sympy, shapely, matplotlib) touches it.

It is deliberately stdlib-only. Importing the record definition should not pull
in a numerics stack, and the validation here must be runnable in any context,
including a bare Colab cell before ``pip install``.

How ``params`` is stored
------------------------
``params`` is stored as a **tuple of sorted ``(key, value)`` pairs**, not as a
dict and not as a JSON string.

The spec flagged the underlying problem: a frozen dataclass holding a dict is
not really immutable, and a dict cannot be hashed, so ``dedupe`` has nothing to
key on. Of the two candidates named in the plan:

- A **JSON string** round-trips perfectly and hashes trivially, but it makes the
  emitted manifest carry an escaped JSON string inside a JSON document. Anything
  reading ``manifest.jsonl`` downstream -- ``jq``, pandas, a collaborator -- then
  has to parse twice to reach ``params.n``. For an artifact meant to be published
  alongside a paper, that cost is paid by every future reader.
- A **tuple of sorted pairs** is natively hashable and immutable, and can still
  be serialized as an ordinary JSON *object*, keeping the manifest idiomatic.

So the tuple wins, with the serialization asymmetry handled here: ``__post_init__``
accepts a plain ``Mapping`` and canonicalizes it, ``params_dict`` hands one back
for ergonomics, and ``to_json``/``from_json`` convert to and from a JSON object.
Callers never have to think about the internal form.

Sorting by key is what makes the form canonical, which is what makes the JSON
round-trip exact rather than merely equivalent.

A third option -- keep a plain dict and let ``dedupe`` canonicalize at hash time --
was rejected because it leaves the mutation hazard the spec called out: nothing
would stop a caller mutating ``params`` after ``signature`` was computed, silently
decoupling a record's identity from its content.

Param values are restricted to JSON scalars and (nested) sequences of them. That
restriction is what guarantees hashability; ``validate()`` enforces it rather than
the constructor, so that a bad record can be built, inspected, and reported on
before it is rejected.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, fields
from typing import Any, Literal, Optional, Union

__all__ = [
    "Category",
    "Datapoint",
    "ORIGINS",
    "SchemaValidationError",
    "Origin",
    "canonical_params",
    "stem_leaks_geometry",
]

Origin = Literal["generated", "harvested", "structured"]

#: Every accepted ``origin``. ``structured`` is a real problem entered by hand as a
#: family's params: it has params and a signature like a generated record, no seed
#: because nothing was drawn, and a known answer to check the family against.
ORIGINS = ("generated", "harvested", "structured")

#: Fields added after the manifest format was fixed, omitted from ``to_dict`` while
#: unset so that records of the existing origins serialize byte-identically to the
#: manifests already on disk.
_OMIT_WHEN_NONE = frozenset({"expected_answer"})

#: 1 = diagram-dependent, 2 = textual. The paper's two-category structure, and the
#: variable its central claim is measured against, so it belongs in the record
#: rather than being inferred from whether ``image_path`` happens to be set.
Category = Literal[1, 2]

JsonScalar = Union[str, int, float, bool, None]
ParamValue = Union[JsonScalar, tuple["ParamValue", ...]]
CanonicalParams = tuple[tuple[str, ParamValue], ...]
ParamsInput = Union[Mapping[str, Any], CanonicalParams]


class SchemaValidationError(ValueError):
    """Raised when a Datapoint violates an invariant downstream code relies on.

    Distinct from a bare ValueError so that a generation driver can catch schema
    problems specifically without also swallowing arithmetic errors from sympy.
    """


def _freeze(value: Any) -> Any:
    """Recursively convert sequences to tuples so a param value can be hashed.

    Strings and bytes are sequences but are already hashable and must not be
    exploded into tuples of characters, so they are passed through untouched.
    """
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Sequence):
        return tuple(_freeze(v) for v in value)
    return value


def canonical_params(params: ParamsInput) -> CanonicalParams:
    """Normalize a parameter mapping into the hashable, order-stable storage form.

    Accepts either a plain mapping (what family code naturally produces) or an
    already-canonical tuple (what a round-tripped record produces), so that
    canonicalizing twice is a no-op. Exposed rather than private because
    ``dedupe`` needs the same normalization to build a signature.
    """
    if isinstance(params, Mapping):
        items: tuple[Any, ...] = tuple(params.items())
    else:
        items = tuple(params)

    normalized: list[tuple[str, ParamValue]] = []
    for item in items:
        if not (isinstance(item, tuple) and len(item) == 2):
            raise SchemaValidationError(
                f"params must be a mapping or a sequence of (key, value) pairs; got item {item!r}"
            )
        key, value = item
        if not isinstance(key, str):
            raise SchemaValidationError(
                f"param keys must be str so they survive a JSON round-trip; "
                f"got {type(key).__name__}: {key!r}"
            )
        normalized.append((key, _freeze(value)))

    keys = [k for k, _ in normalized]
    if len(set(keys)) != len(keys):
        raise SchemaValidationError(f"duplicate param keys: {sorted(keys)}")

    return tuple(sorted(normalized, key=lambda kv: kv[0]))


def _unfreeze(value: Any) -> Any:
    """Inverse of ``_freeze`` for JSON emission: tuples become lists."""
    if isinstance(value, tuple):
        return [_unfreeze(v) for v in value]
    return value


#: Words that give away a side count, which for a Category 1 problem is exactly the
#: quantity the diagram is supposed to be the only source of. "Find the area between
#: the hexagon and the triangle" states n=6 and m=3 in words.
_SHAPE_WORDS = frozenset(
    """triangle quadrilateral square rectangle rhombus trapezoid trapezium pentagon
    hexagon heptagon octagon nonagon decagon hendecagon dodecagon""".split()
)

#: Number words, for the same reason a digit is disallowed. "two circles" is a
#: measurement stated in text.
_NUMBER_WORDS = frozenset(
    """one two three four five six seven eight nine ten eleven twelve""".split()
)


def stem_leaks_geometry(stem: str) -> Optional[str]:
    """Return why a Category 1 stem leaks geometry, or ``None`` if it is clean.

    The paper's first selection criterion requires the diagram to be "the exclusive
    or primary carrier of geometric information, with no redundant textual
    description of spatial relationships". A stem that states a measurement or names
    a polygon has broken that, and the resulting problem no longer measures what the
    benchmark claims to measure -- silently, since the record is otherwise valid.

    Deliberately crude, and deliberately run on every record. Three checks:

    - **digits**, the cheapest possible detector for a numeric leak;
    - **number words**, which a digit check would miss;
    - **polygon names**, which leak a side count -- the implicit quantity in the one
      family that exists, and the kind of leak most easily written by accident.

    False positives are possible: a legitimate stem may one day need a numeral. That
    is the right direction to fail, since the cost is rewording a stem, while the
    cost of a false negative is a benchmark that quietly measures reading instead of
    seeing. Loosen this when a real stem needs it, not in advance.
    """
    if any(ch.isdigit() for ch in stem):
        digits = "".join(sorted({ch for ch in stem if ch.isdigit()}))
        return f"stem contains digits ({digits!r}), which states a measurement in text"

    words = {word.strip(".,;:!?()").lower() for word in stem.split()}

    leaked_numbers = sorted(words & _NUMBER_WORDS)
    if leaked_numbers:
        return f"stem contains number words {leaked_numbers}, which state a count in text"

    leaked_shapes = sorted(words & _SHAPE_WORDS)
    if leaked_shapes:
        return (
            f"stem names shapes {leaked_shapes}, which gives away a side count the "
            f"diagram is supposed to be the only source of"
        )
    return None


@dataclass(frozen=True)
class Datapoint:
    """One benchmark problem: what the model sees, and what the answer is.

    Frozen because a record's ``signature`` is derived from its ``params``; if a
    record could be edited after construction those two could silently disagree,
    and dedupe would be keying on something that is no longer true.

    ``answer_exact`` holds a sympy-serializable string (e.g. ``"8 + 4*sqrt(3)"``)
    rather than a float, because the benchmark's real answers are frequently
    exact radicals and a float discards the form a model may legitimately answer
    in. ``answer_decimal`` is the float projection, kept alongside for tolerance
    comparison, not as a replacement.
    """

    problem_id: str
    family: str
    category: Category
    stem: str
    image_path: Optional[str]
    answer_exact: str
    answer_decimal: float
    origin: Origin

    # Procedural-generation provenance. Required for origin="generated", required
    # bar the seed for origin="structured", and None for origin="harvested": a
    # harvested problem was not drawn from a
    # parameter space, so it has no seed, no params, and no generator version to
    # record, and a dedupe signature over absent params would be a fiction.
    # Enforced by _validate_origin_coupling rather than by the type, so that a
    # generated record cannot quietly lose its signature.
    seed: Optional[int] = None
    params: Optional[ParamsInput] = None
    signature: Optional[str] = None
    generator_version: Optional[str] = None

    # Harvest provenance. Optional, and empty for generated records, so that the
    # harvest package can emit this same record type without a parallel schema.
    source_site: Optional[str] = None
    source_url: Optional[str] = None
    source_problem_id: Optional[str] = None
    retrieved_at: Optional[str] = None
    answer_source: Optional[str] = None
    original_answer: Optional[str] = None

    # Structured-input provenance. Required for origin="structured": the known
    # answer the source gives, as a sympy-parseable string, which the driver checked
    # ``answer_exact`` against before emitting. Optional and unset for the others.
    expected_answer: Optional[str] = None

    def __post_init__(self) -> None:
        # Canonicalize on the way in so callers may pass a dict while storage
        # stays hashable. object.__setattr__ is the standard escape hatch for
        # normalizing a field on a frozen dataclass.
        if self.params is not None:
            object.__setattr__(self, "params", canonical_params(self.params))

    @property
    def params_dict(self) -> dict[str, Any]:
        """The parameters as an ordinary dict, for code that just wants ``p["n"]``.

        Empty for a harvested record, which has no parameters at all.
        """
        if self.params is None:
            return {}
        return {k: _unfreeze(v) for k, v in self.params}

    def validate(self) -> None:
        """Raise ``SchemaValidationError`` if this record would corrupt later stages.

        Checks the invariants that are cheap here and expensive to discover later:
        a non-finite or zero answer means the solver produced nonsense, an empty
        stem means the problem has no question, and params that will not survive a
        JSON round-trip mean the manifest cannot be reloaded.
        """
        if self.origin not in ORIGINS:
            raise SchemaValidationError(f"origin must be one of {ORIGINS}; got {self.origin!r}")

        # Two traps: bool is a subclass of int, so True would pass as 1; and
        # 1.0 == 1, so a float would pass the membership test and then serialize
        # as "1.0", breaking an exact round-trip.
        if (
            isinstance(self.category, bool)
            or not isinstance(self.category, int)
            or self.category not in (1, 2)
        ):
            raise SchemaValidationError(
                f"category must be 1 (diagram-dependent) or 2 (textual); got {self.category!r}"
            )

        self._validate_category_coupling()

        for name in ("problem_id", "family", "stem", "answer_exact"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SchemaValidationError(f"{name} must be a non-empty string; got {value!r}")

        self._validate_origin_coupling()

        if isinstance(self.answer_decimal, bool) or not isinstance(
            self.answer_decimal, (int, float)
        ):
            raise SchemaValidationError(
                f"answer_decimal must be a real number; got {self.answer_decimal!r}"
            )
        if not math.isfinite(self.answer_decimal):
            raise SchemaValidationError(
                f"answer_decimal must be finite; got {self.answer_decimal!r}"
            )
        if self.answer_decimal == 0:
            raise SchemaValidationError(
                "answer_decimal is zero, which means the configuration is degenerate"
            )

        if self.params is not None:
            self._validate_params()

    def _validate_origin_coupling(self) -> None:
        """Tie the generator-only fields to ``origin``.

        A generated record missing its seed or signature cannot be reproduced or
        deduped -- a silent corruption if only the type is checked, since ``None``
        is now valid for the field. Nothing is asserted in the other direction: a
        harvested record is *expected* to leave these unset, but carrying them is
        not an error, so a record that gains a signature later still validates.
        """
        generator_only = ("seed", "params", "signature", "generator_version")

        if self.origin == "generated":
            missing = [name for name in generator_only if getattr(self, name) is None]
            if missing:
                raise SchemaValidationError(
                    f"origin='generated' requires {generator_only}; missing {missing}"
                )
            for name in ("signature", "generator_version"):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise SchemaValidationError(
                        f"{name} must be a non-empty string; got {value!r}"
                    )
            # bool is a subclass of int; a boolean seed is a bug, not a seed.
            if isinstance(self.seed, bool) or not isinstance(self.seed, int):
                raise SchemaValidationError(f"seed must be an int; got {self.seed!r}")

        elif self.origin == "structured":
            required = ("params", "signature", "generator_version", "expected_answer")
            missing = [name for name in required if getattr(self, name) is None]
            if missing:
                raise SchemaValidationError(
                    f"origin='structured' requires {required}; missing {missing}"
                )
            for name in ("signature", "generator_version", "expected_answer"):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise SchemaValidationError(
                        f"{name} must be a non-empty string; got {value!r}"
                    )
            # A seed on a record nothing was drawn for would claim a reproducibility
            # path that does not exist.
            if self.seed is not None:
                raise SchemaValidationError(
                    f"origin='structured' was not drawn from a seed; got seed={self.seed!r}"
                )
            self._validate_structured_param_keys()

    def _validate_structured_param_keys(self) -> None:
        """Require exactly the keys the family's ``solve`` reads.

        Imports the registry lazily, so this module stays importable, and every other
        origin validatable, without the family package.
        """
        from generator import registry

        try:
            family = registry.get(self.family)
        except KeyError as exc:
            raise SchemaValidationError(
                f"origin='structured' needs a registered family; {exc.args[0]}"
            ) from None
        param_types = getattr(family, "PARAM_TYPES", None)
        if param_types is None:
            raise SchemaValidationError(
                f"family {self.family!r} declares no PARAM_TYPES, so it cannot take structured input"
            )
        expected = set(param_types)
        actual = {key for key, _ in self.params}  # type: ignore[union-attr]
        if actual != expected:
            raise SchemaValidationError(
                f"params for family {self.family!r} must have exactly the keys "
                f"{sorted(expected)}; missing {sorted(expected - actual)}, "
                f"unexpected {sorted(actual - expected)}"
            )

    def _validate_params(self) -> None:
        """Confirm params are hashable and survive a JSON round-trip unchanged."""
        try:
            hash(self.params)
        except TypeError as exc:
            raise SchemaValidationError(f"params are not hashable: {exc}") from exc

        try:
            encoded = json.dumps(self.params_dict, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"params are not JSON-serializable: {exc}") from exc

        if canonical_params(json.loads(encoded)) != self.params:
            raise SchemaValidationError(
                "params do not survive a JSON round-trip unchanged; "
                f"before={self.params!r} after={canonical_params(json.loads(encoded))!r}"
            )

    def _validate_category_coupling(self) -> None:
        """Enforce the invariants that tie a category to the rest of the record.

        A Category 1 problem without a diagram is not a Category 1 problem, and a
        Category 2 problem with one has quietly become Category 1 -- which would
        corrupt the comparison the paper's central claim is measured against, since
        the between-category gap is the result.
        """
        if self.category == 1:
            if not isinstance(self.image_path, str) or not self.image_path.strip():
                raise SchemaValidationError(
                    "category 1 is diagram-dependent and requires an image_path; "
                    f"got {self.image_path!r}"
                )
            leak = stem_leaks_geometry(self.stem)
            if leak is not None:
                raise SchemaValidationError(
                    f"category 1 stem must not carry geometric information: {leak}. "
                    f"stem was {self.stem!r}"
                )
        elif self.image_path is not None:
            raise SchemaValidationError(
                "category 2 is textual and must not carry a diagram; "
                f"got image_path={self.image_path!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """The record as plain JSON-compatible types, with params as a JSON object."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None and f.name in _OMIT_WHEN_NONE:
                continue
            out[f.name] = self.params_dict if f.name == "params" else value
        return out

    def to_json(self) -> str:
        """One manifest line. ``sort_keys`` so byte-identical runs stay byte-identical."""
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Datapoint":
        """Rebuild a record, failing loudly on unknown or missing fields.

        A silently-dropped unknown field would mean a manifest written by a newer
        generator version loads as if the extra data never existed, which is the
        kind of mismatch that only surfaces much later as an unexplained result.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise SchemaValidationError(f"unknown fields in record: {sorted(unknown)}")

        required = {
            f.name
            for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing = required - set(data)
        if missing:
            raise SchemaValidationError(f"missing required fields: {sorted(missing)}")

        return cls(**data)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, line: str) -> "Datapoint":
        """Inverse of ``to_json``; ``from_json(dp.to_json()) == dp`` for valid records."""
        return cls.from_dict(json.loads(line))
