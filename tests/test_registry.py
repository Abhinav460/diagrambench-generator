"""Stage 2 acceptance tests: dispatch works, and a broken family fails loudly
with a message naming what is missing."""

from __future__ import annotations

from types import ModuleType

import pytest

from generator import registry
from generator.registry import (
    Family,
    FamilyContractError,
    rejection_reason,
)


def make_module(name: str = "fake", **attrs) -> ModuleType:
    """Build a module satisfying the contract, so each test can break one piece."""
    mod = ModuleType(f"generator.families.{name}")
    mod.NAME = name
    mod.sample = lambda rng: {"value": 1}
    mod.is_valid = lambda params: True
    mod.solve = lambda params: ("stem", 1, [])
    for key, value in attrs.items():
        if value is _DELETE:
            delattr(mod, key)
        else:
            setattr(mod, key, value)
    return mod


class _Delete:
    pass


_DELETE = _Delete()


# --- dispatch -------------------------------------------------------------


def test_get_returns_something_satisfying_the_contract():
    fam = registry.get("nested_polygons")
    assert isinstance(fam, Family)
    for func in registry.REQUIRED_FUNCTIONS:
        assert callable(getattr(fam, func))


def test_a_real_family_dispatches_end_to_end():
    import numpy as np

    fam = registry.get("nested_polygons")
    rng = np.random.default_rng(0)
    params = next(p for p in iter(lambda: fam.sample(rng), None) if fam.is_valid(p) is True)
    stem, answer, spec = fam.solve(params)
    assert stem and answer is not None and spec


def test_families_are_discovered_without_explicit_registration():
    assert "nested_polygons" in registry.available()


def test_unknown_family_names_the_available_ones():
    with pytest.raises(KeyError, match="no family named 'nope'"):
        registry.get("nope")


# --- contract violations, one per test ------------------------------------


@pytest.mark.parametrize("missing", ["sample", "is_valid", "solve"])
def test_missing_function_fails_registration_naming_it(missing):
    mod = make_module(name=f"missing_{missing}", **{missing: _DELETE})
    with pytest.raises(FamilyContractError, match=rf"missing required function {missing}\(\)"):
        registry.register(mod)


def test_non_callable_function_fails_registration():
    mod = make_module(name="not_callable", solve="not a function")
    with pytest.raises(FamilyContractError, match="defines solve as str"):
        registry.register(mod)


@pytest.mark.parametrize("bad_name", [None, "", "   ", 42])
def test_missing_or_blank_name_fails_registration(bad_name):
    mod = make_module(name="named")
    mod.NAME = bad_name
    with pytest.raises(FamilyContractError, match="must define NAME"):
        registry.register(mod)


def test_zero_arity_sample_fails_registration():
    """def sample(): would otherwise surface as a bare TypeError mid-run."""
    mod = make_module(name="bad_arity")
    mod.sample = lambda: {"value": 1}
    with pytest.raises(FamilyContractError, match=r"cannot accept its required argument \(rng\)"):
        registry.register(mod)


def test_zero_arity_solve_fails_registration():
    mod = make_module(name="bad_arity_solve")
    mod.solve = lambda: ("stem", 1, [])
    with pytest.raises(FamilyContractError, match=r"cannot accept its required argument \(params\)"):
        registry.register(mod)


def test_varargs_sample_is_accepted():
    mod = make_module(name="varargs")
    mod.sample = lambda *args: {"value": 1}
    assert registry.register(mod) == "varargs"


def test_duplicate_name_from_a_different_module_fails():
    first = make_module(name="dupe")
    second = make_module(name="dupe")
    registry.register(first)
    with pytest.raises(FamilyContractError, match="already registered"):
        registry.register(second)


def test_reregistering_the_same_module_is_idempotent():
    mod = make_module(name="idempotent")
    assert registry.register(mod) == "idempotent"
    assert registry.register(mod) == "idempotent"


# --- is_valid return normalization ----------------------------------------


def test_rejection_reason_passes_valid_through_as_none():
    assert rejection_reason(True) is None


def test_rejection_reason_keeps_the_reason_string():
    assert rejection_reason("inner polygon does not fit") == "inner polygon does not fit"


def test_rejection_reason_labels_bare_false():
    assert rejection_reason(False) == "unspecified"


def test_rejection_reason_labels_empty_string():
    assert rejection_reason("") == "unspecified"


def test_rejection_reason_rejects_nonsense():
    with pytest.raises(FamilyContractError, match="must return True or a reason string"):
        rejection_reason(0.5)
