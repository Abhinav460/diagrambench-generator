"""Stage 6 acceptance tests: identical params collide, params differing beyond the
rounding threshold do not, and the collision rate over a real run is reported."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from generator import registry
from generator.dedupe import (
    DEFAULT_PRECISION,
    DedupeError,
    Deduplicator,
    normalize,
    signature,
)
from generator.families import nested_polygons as fam
from generator.registry import rejection_reason
from generator.schema import canonical_params

REPO = Path(__file__).resolve().parent.parent


def params(n=6, m=3, side_outer=7, side_inner=3):
    return {"n": n, "m": m, "side_outer": side_outer, "side_inner": side_inner}


# --- the two acceptance criteria ------------------------------------------


def test_identical_params_collide():
    assert signature("nested_polygons", params()) == signature("nested_polygons", params())


def test_identical_params_collide_regardless_of_key_order():
    """Params arrive as a dict, and dict order follows insertion. If ordering
    reached the digest, the same problem sampled by differently-written family code
    would hash apart."""
    a = {"n": 6, "m": 3, "side_outer": 7, "side_inner": 3}
    b = {"side_inner": 3, "side_outer": 7, "m": 3, "n": 6}
    assert signature("f", a) == signature("f", b)


def test_params_differing_beyond_the_threshold_do_not_collide():
    assert signature("f", params(m=3)) != signature("f", params(m=4))
    assert signature("f", params(side_outer=7)) != signature("f", params(side_outer=8))


@pytest.mark.parametrize("precision", [0, 2, 6, 10])
def test_a_difference_larger_than_the_threshold_survives_any_precision(precision):
    assert signature("f", {"x": 1.0}, precision=precision) != signature(
        "f", {"x": 2.0}, precision=precision
    )


def test_a_difference_below_the_threshold_is_rounded_away():
    """The point of rounding: float noise must not manufacture a new problem."""
    assert signature("f", {"x": 1.0}, precision=6) == signature(
        "f", {"x": 1.0 + 1e-12}, precision=6
    )


def test_the_threshold_is_where_the_precision_says_it_is():
    at_precision_2 = signature("f", {"x": 1.0}, precision=2)
    assert at_precision_2 == signature("f", {"x": 1.001}, precision=2)
    assert at_precision_2 != signature("f", {"x": 1.01}, precision=2)


def test_high_precision_lets_noise_through_and_low_precision_merges_problems():
    """Both documented failure modes, pinned so the docstring cannot drift from
    the behaviour it describes."""
    noisy_a, noisy_b = {"x": 1.0}, {"x": 1.0 + 1e-12}
    assert signature("f", noisy_a, precision=15) != signature("f", noisy_b, precision=15)

    distinct_a, distinct_b = {"x": 1.2}, {"x": 1.4}
    assert signature("f", distinct_a, precision=0) == signature("f", distinct_b, precision=0)


# --- the collision rate over a run of 500 ---------------------------------


def test_collision_rate_over_a_run_of_500():
    """The Stage 6 acceptance criterion. Asserts the rate is in a sane band rather
    than at a fixed value: what matters is that duplicates occur (so the mechanism
    is doing something) but do not dominate (so the space is not exhausted)."""
    rng = np.random.default_rng(0)
    dedupe = Deduplicator()
    accepted = 0

    for _ in range(500):
        candidate = fam.sample(rng)
        if rejection_reason(fam.is_valid(candidate)) is not None:
            continue
        if dedupe.add(fam.NAME, candidate) is not None:
            accepted += 1

    assert dedupe.considered == accepted + dedupe.collisions
    assert len(dedupe) == accepted
    assert 0 < dedupe.collision_rate < 0.5, (
        f"collision rate {dedupe.collision_rate:.3f} over {dedupe.considered} valid draws "
        f"({accepted} unique, {dedupe.collisions} repeats)"
    )


def test_every_accepted_signature_is_distinct():
    rng = np.random.default_rng(1)
    dedupe = Deduplicator()
    returned = []
    for _ in range(300):
        candidate = fam.sample(rng)
        if rejection_reason(fam.is_valid(candidate)) is not None:
            continue
        sig = dedupe.add(fam.NAME, candidate)
        if sig is not None:
            returned.append(sig)

    assert len(returned) == len(set(returned)) == len(dedupe)


def test_the_same_params_drawn_twice_is_reported_as_a_collision():
    dedupe = Deduplicator()
    assert dedupe.add("f", params()) is not None
    assert dedupe.add("f", params()) is None
    assert dedupe.collisions == 1
    assert dedupe.considered == 2
    assert dedupe.collision_rate == pytest.approx(0.5)


def test_collision_rate_of_an_empty_run_is_zero_not_undefined():
    assert Deduplicator().collision_rate == 0.0


def test_first_with_recovers_the_params_that_claimed_a_signature():
    """A collision is only debuggable if the winner can be named."""
    dedupe = Deduplicator()
    sig = dedupe.add("f", params())
    assert dedupe.add("f", params()) is None
    assert dedupe.first_with(sig) == canonical_params(params())


# --- family scoping --------------------------------------------------------


def test_the_same_params_in_different_families_do_not_collide():
    """Two families may well share a param name like 'n' without sharing meaning."""
    assert signature("family_a", params()) != signature("family_b", params())


def test_family_and_payload_boundary_cannot_be_confused():
    """Concatenating family and payload without a separator would let a shifted
    boundary produce the same digest from different inputs."""
    assert signature("a", {"bc": 1}) != signature("ab", {"c": 1})


# --- stability -------------------------------------------------------------


def test_signature_is_stable_across_processes():
    """hash() is randomized per process. If that ever crept in here, an appended
    run could not be deduped against an existing manifest."""
    script = (
        "from generator.dedupe import signature\n"
        "print(signature('nested_polygons', {'n': 6, 'm': 3, 'side_outer': 7, 'side_inner': 3}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == signature("nested_polygons", params())


def test_signature_is_a_hex_digest():
    sig = signature("f", params())
    assert len(sig) == 64
    assert set(sig) <= set("0123456789abcdef")


def test_signature_accepts_already_canonical_params():
    """Records round-tripped from a manifest carry the canonical tuple form, and
    must dedupe against records that carry a dict."""
    assert signature("f", canonical_params(params())) == signature("f", params())


# --- value normalization ---------------------------------------------------


def test_int_and_float_of_equal_value_are_the_same_problem():
    assert signature("f", {"n": 6}) == signature("f", {"n": 6.0})


def test_negative_zero_matches_zero():
    assert signature("f", {"x": -0.0}) == signature("f", {"x": 0.0})


def test_bool_does_not_collide_with_the_int_it_equals():
    """True == 1 in Python, but a flag and a count are never the same parameter."""
    assert signature("f", {"flag": True}) != signature("f", {"flag": 1})


def test_none_and_strings_are_distinguished_from_numbers():
    assert signature("f", {"x": None}) != signature("f", {"x": 0})
    assert signature("f", {"x": "1"}) != signature("f", {"x": 1})


def test_nested_sequences_are_normalized_elementwise():
    assert signature("f", {"pts": [1.0, 2.0]}) == signature("f", {"pts": (1, 2)})
    assert signature("f", {"pts": [1.0, 2.0]}) != signature("f", {"pts": [2.0, 1.0]})


def test_nested_floats_are_rounded_too():
    assert signature("f", {"pts": [1.0, 2.0]}, precision=6) == signature(
        "f", {"pts": [1.0 + 1e-12, 2.0]}, precision=6
    )


def test_normalize_is_inspectable_for_debugging_a_collision():
    """The reason normalize is public: a digest cannot explain why two problems
    hashed together, and the rounded form can."""
    assert normalize({"n": 6, "m": 3}, precision=2) == [
        ["m", ["n", "3.00"]],
        ["n", ["n", "6.00"]],
    ]


# --- rejected inputs -------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_param_values_raise(bad):
    """A non-finite param survives json.dumps under default settings, so it would
    otherwise reach the digest and hash as a legitimate value."""
    with pytest.raises(DedupeError, match="non-finite"):
        signature("f", {"x": bad})


def test_unhashable_param_type_raises():
    with pytest.raises(DedupeError, match="cannot be normalized"):
        signature("f", {"x": {1, 2}})


def test_negative_precision_raises():
    with pytest.raises(DedupeError, match="non-negative"):
        signature("f", params(), precision=-1)


@pytest.mark.parametrize("bad", ["", "   ", None, 3])
def test_empty_or_non_string_family_raises(bad):
    with pytest.raises(DedupeError, match="non-empty string"):
        signature(bad, params())


# --- priming from an existing manifest ------------------------------------


def test_priming_suppresses_problems_an_earlier_run_already_emitted():
    earlier = signature("f", params())
    dedupe = Deduplicator()
    dedupe.prime([earlier])

    assert earlier in dedupe
    assert dedupe.add("f", params()) is None
    assert dedupe.collisions == 1


def test_priming_uses_the_same_precision_the_deduplicator_was_built_with():
    dedupe = Deduplicator(precision=2)
    dedupe.prime([signature("f", {"x": 1.0}, precision=2)])
    assert dedupe.add("f", {"x": 1.001}) is None


def test_registered_family_name_is_what_the_driver_will_key_on():
    """Guards against the driver keying on a module path or a display name; the
    registry's name is the one that has to reach the signature."""
    assert registry.get(fam.NAME) is fam
