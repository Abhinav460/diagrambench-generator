"""The family contract, and the registry that dispatches on family name.

A *family* is one parametrized problem type -- "nested polygons", "circle tangent
to two square edges" -- implemented as a single module under ``generator.families``.
Families are the only part of this package that grows as configuration types are
added; everything else is written once. This module is what makes that true, by
fixing the interface a family must satisfy and refusing to register one that does
not.

The contract is checked structurally at registration time rather than trusted by
convention, so that a family with a typo'd function name fails immediately with a
message naming the missing function, instead of failing deep inside the generation
loop a hundred problems later.

Families never import matplotlib and never construct their own ``rng``. Both rules
exist for the same reason: rendering policy and randomness must be controlled
centrally, or reproducibility and the labeling policy leak into per-family code
where they cannot be reviewed as a whole.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterable, Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, Protocol, Union, runtime_checkable

if TYPE_CHECKING:  # keeps this module importable without numpy or sympy present
    from numpy.random import Generator as RNG
    from sympy import Expr
else:
    RNG = Any
    Expr = Any

__all__ = [
    "Family",
    "FamilyContractError",
    "GeometrySpec",
    "Params",
    "ValidationResult",
    "GEOMETRY",
    "Issue",
    "READABILITY",
    "available",
    "classify_issues",
    "first_issue",
    "get",
    "register",
    "rejection_reason",
]

Params = Mapping[str, Any]

#: A neutral description of what to draw: a sequence of primitives, each a mapping
#: carrying coordinates and style flags. The concrete format is defined by the
#: first family that emits one (Stage 3, ``families/nested_polygons.py``) and is
#: consumed by both ``render`` and ``verify``. Deliberately unconstrained here so
#: that Stage 2 does not pre-empt a Stage 3 decision.
GeometrySpec = Sequence[Mapping[str, Any]]

#: ``True`` means the parameters are usable. A non-empty ``str`` means rejected,
#: and the string is the reason -- which is what makes an acceptance rate
#: debuggable rather than just low. Bare ``False`` is accepted but discouraged,
#: since it discards the reason.
ValidationResult = Union[bool, str]

#: The three functions every family module must define, plus the name constant.
REQUIRED_FUNCTIONS = ("sample", "is_valid", "solve")

#: Minimum positional arity for each required function.
_REQUIRED_ARITY = {"sample": 1, "is_valid": 1, "solve": 1}


class FamilyContractError(TypeError):
    """Raised when a module does not satisfy the family contract.

    A TypeError subclass because the failure is structural -- the module is not
    the shape the registry requires -- and callers that want to distinguish a
    broken family from a missing one can catch this specifically.
    """


@runtime_checkable
class Family(Protocol):
    """The interface a family module implements.

    Declared as a Protocol so families are plain modules rather than classes that
    must inherit from anything; ``register`` checks the shape explicitly in order
    to produce an error message naming what is missing, which ``isinstance``
    against a Protocol cannot do.

    Two further attributes are optional for registration but required to accept
    structured (hand-supplied) input: ``PARAM_TYPES``, mapping each param name to a
    kind in ``generator.params``, and ``validation_issues(params)``, yielding
    ``(GEOMETRY | READABILITY, reason)`` pairs in the order ``is_valid`` checks them.
    """

    NAME: str

    def sample(self, rng: RNG) -> Params:
        """Draw one candidate parameter set from the family's space.

        May return parameters that are invalid; filtering is ``is_valid``'s job.
        Must not construct its own randomness -- ``rng`` is passed in so a run is
        reproducible from a single seed.
        """

    def is_valid(self, params: Params) -> ValidationResult:
        """Reject degenerate or out-of-range configurations.

        Returns ``True`` when usable, or a string explaining the rejection. The
        string matters: it is what turns a 4% acceptance rate from a mystery into
        a histogram of causes.
        """

    def solve(self, params: Params) -> tuple[str, Expr, GeometrySpec]:
        """Return ``(stem, answer, geometry_spec)`` for a valid parameter set.

        ``stem`` is the question text, naming only the quantity asked for.
        ``answer`` is an exact sympy expression, not a float -- the benchmark's
        real answers are frequently radicals, and a float discards the form a
        model may legitimately answer in. ``emit`` renders it into the record's
        ``answer_exact`` (string) and ``answer_decimal`` (float) fields.
        ``geometry_spec`` describes what to draw, and is what ``verify`` rebuilds
        from -- so it, not ``params``, is the thing that must agree with the answer.
        """


#: Severity of a validation issue. A *geometry* issue means the params do not
#: describe a well-formed figure whose answer the family's formula gives: a
#: polygon with fewer than three sides, a notch outside its rectangle, an inner
#: polygon that crosses the outer one. A *readability* issue means the figure is
#: well-formed but falls outside the heuristics that keep random draws legible: a
#: hairline ring, a 20-gon, a notch that swallows most of the figure.
GEOMETRY = "geometry"
READABILITY = "readability"

#: One ``(severity, reason)`` pair, as yielded by a family's ``validation_issues``.
Issue = tuple[str, str]


def first_issue(issues: Iterable[Issue]) -> ValidationResult:
    """``is_valid`` for random draws: the first issue of either severity rejects.

    Families implement ``is_valid`` as ``first_issue(validation_issues(params))``,
    yielding their checks in the order ``is_valid`` always ran them, with the same
    messages. So a random draw is rejected by exactly the checks, and with exactly
    the reason, that it was before the checks were split by severity.
    """
    for _severity, reason in issues:
        return reason
    return True


def classify_issues(family: Any, params: Params) -> tuple[Optional[str], list[str]]:
    """``(blocking_reason, readability_warnings)`` for a hand-supplied parameter set.

    The structured-input counterpart of ``first_issue``: readability issues are
    collected as warnings, and the first geometry issue stops the scan and blocks.
    Checks after a geometry issue are not run, since they may assume what it
    rejected.
    """
    issues = getattr(family, "validation_issues", None)
    if issues is None:
        raise FamilyContractError(
            f"family {getattr(family, 'NAME', family)!r} does not define "
            f"validation_issues, so it cannot accept structured input"
        )
    warnings: list[str] = []
    for severity, reason in issues(params):
        if severity == GEOMETRY:
            return reason, warnings
        if severity != READABILITY:
            raise FamilyContractError(
                f"validation issue severity must be {GEOMETRY!r} or {READABILITY!r}; "
                f"got {severity!r}"
            )
        warnings.append(reason)
    return None, warnings


def rejection_reason(result: ValidationResult) -> Optional[str]:
    """Normalize an ``is_valid`` return into ``None`` (valid) or a reason string.

    Exists so the driver loop and the acceptance-rate reporting do not each
    re-implement the bool-or-string union, and so a family returning bare
    ``False`` still produces a usable bucket label.
    """
    if result is True:
        return None
    if result is False:
        return "unspecified"
    if isinstance(result, str):
        return result or "unspecified"
    raise FamilyContractError(
        f"is_valid must return True or a reason string; got {result!r} ({type(result).__name__})"
    )


def _check_contract(module: ModuleType) -> None:
    """Raise ``FamilyContractError`` naming the first thing the module is missing."""
    origin = getattr(module, "__name__", repr(module))

    name = getattr(module, "NAME", None)
    if not isinstance(name, str) or not name.strip():
        raise FamilyContractError(
            f"family module {origin!r} must define NAME as a non-empty string; got {name!r}"
        )

    for func_name in REQUIRED_FUNCTIONS:
        func = getattr(module, func_name, None)
        if func is None:
            raise FamilyContractError(
                f"family {name!r} ({origin}) is missing required function {func_name}(); "
                f"the contract is {', '.join(f + '()' for f in REQUIRED_FUNCTIONS)}"
            )
        if not callable(func):
            raise FamilyContractError(
                f"family {name!r} ({origin}) defines {func_name} as "
                f"{type(func).__name__}, which is not callable"
            )
        _check_arity(name, origin, func_name, func)


def _check_arity(name: str, origin: str, func_name: str, func: Any) -> None:
    """Confirm a family function can accept its required positional argument.

    Caught here rather than at call time because a ``def sample():`` surfaces
    otherwise as a bare TypeError from inside the generation loop, with nothing
    in the message pointing at which family is at fault.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):  # builtins and C callables have no signature
        return

    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    accepts_varargs = any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values())
    required = _REQUIRED_ARITY[func_name]

    if len(positional) < required and not accepts_varargs:
        expected = "rng" if func_name == "sample" else "params"
        raise FamilyContractError(
            f"family {name!r} ({origin}) defines {func_name}{sig}, which cannot accept "
            f"its required argument ({expected})"
        )


_REGISTRY: dict[str, ModuleType] = {}
_DISCOVERED = False


def register(module: ModuleType) -> str:
    """Validate a family module and add it to the registry, returning its name.

    Re-registering the same module under the same name is a no-op; registering a
    *different* module under a name already taken raises, since two families
    sharing a name would make ``Datapoint.family`` ambiguous and the manifest
    unreadable after the fact.
    """
    _check_contract(module)
    name: str = module.NAME

    existing = _REGISTRY.get(name)
    if existing is not None and existing is not module:
        raise FamilyContractError(
            f"family name {name!r} is already registered to "
            f"{getattr(existing, '__name__', existing)!r}; names must be unique "
            f"because they are recorded in every emitted record"
        )

    _REGISTRY[name] = module
    return name


def discover(package: str = "generator.families") -> None:
    """Import every module in the families package and register each one.

    Auto-discovery rather than a hand-maintained list, so adding a family is one
    file and no edits elsewhere. Import errors are allowed to propagate: a family
    that fails to import is a bug to fix, not a family to quietly skip.
    """
    global _DISCOVERED
    pkg = importlib.import_module(package)
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        register(importlib.import_module(f"{package}.{info.name}"))
    _DISCOVERED = True


def _ensure_discovered() -> None:
    if not _DISCOVERED:
        discover()


def get(name: str) -> Family:
    """Look up a family by name, raising with the available names if absent."""
    _ensure_discovered()
    try:
        return _REGISTRY[name]  # type: ignore[return-value]
    except KeyError:
        raise KeyError(
            f"no family named {name!r}; available: {sorted(_REGISTRY) or '(none)'}"
        ) from None


def available() -> list[str]:
    """Every registered family name, sorted. Used by the CLI's ``--family all``."""
    _ensure_discovered()
    return sorted(_REGISTRY)
