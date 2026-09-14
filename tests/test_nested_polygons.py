"""Stage 3 acceptance tests: hand-checkable closed forms, a healthy acceptance
rate, and the geometry_spec invariants that Stages 4 and 5 will depend on."""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest
import sympy as sp

from generator import registry
from generator.families import nested_polygons as fam
from generator.registry import rejection_reason


def params(n, m, side_outer=1, side_inner=1):
    return {"n": n, "m": m, "side_outer": side_outer, "side_inner": side_inner}


# --- the three hand-worked cases ------------------------------------------
# Each expected value is written in the form it is derived by hand, then compared
# symbolically, so the test checks the closed form rather than a decimal.


def test_hexagon_containing_triangle_unit_sides():
    """A6 - A3 = (6/4)cot(30) - (3/4)cot(60) = 3*sqrt(3)/2 - sqrt(3)/4 = 5*sqrt(3)/4."""
    _, answer, _ = fam.solve(params(6, 3))
    expected = 5 * sp.sqrt(3) / 4
    assert sp.simplify(answer - expected) == 0
    assert float(answer.evalf()) == pytest.approx(2.1650635094610964)


def test_octagon_containing_square_unit_sides():
    """A8 - A4 = 2cot(pi/8) - 1 = 2(1 + sqrt(2)) - 1 = 1 + 2*sqrt(2)."""
    _, answer, _ = fam.solve(params(8, 4))
    expected = 1 + 2 * sp.sqrt(2)
    assert sp.simplify(answer - expected) == 0
    assert float(answer.evalf()) == pytest.approx(3.8284271247461903)


def test_nonagon_containing_hexagon_unit_sides():
    """A9 - A6 = (9/4)cot(pi/9) - (3/2)sqrt(3). cot(pi/9) has no radical form,
    which is why the answer type is 'any exact expression' rather than radicals."""
    _, answer, _ = fam.solve(params(9, 6))
    expected = sp.Rational(9, 4) * sp.cot(sp.pi / 9) - sp.Rational(3, 2) * sp.sqrt(3)
    assert sp.simplify(answer - expected) == 0
    assert float(answer.evalf()) == pytest.approx(3.583747982419584)


def test_nonagon_answer_is_exact_but_not_radical():
    """Guards the answer-type decision: sympy must not have silently floated it."""
    _, answer, _ = fam.solve(params(9, 6))
    assert answer.free_symbols == set()
    assert answer.has(sp.cot)
    assert not answer.is_Float


# --- validity -------------------------------------------------------------


def test_rejects_inner_side_count_not_less_than_outer():
    assert "not less than" in fam.is_valid(params(6, 6))
    assert "not less than" in fam.is_valid(params(6, 8))


def test_rejects_inner_that_does_not_fit():
    reason = fam.is_valid(params(4, 3, side_outer=1, side_inner=20))
    assert "does not fit" in reason


def test_rejects_inner_too_small_to_read():
    reason = fam.is_valid(params(12, 3, side_outer=20, side_inner=1))
    assert "too small to read" in reason


def test_rejects_out_of_range_side_count():
    assert "outside" in fam.is_valid(params(40, 3))


def test_rejects_malformed_params():
    assert "malformed" in fam.is_valid({"n": 6})


def test_every_rejection_gives_a_reason_string():
    """A bare False would make the acceptance histogram useless."""
    for bad in [params(6, 6), params(4, 3, 1, 20), params(40, 3), {"n": 6}]:
        result = fam.is_valid(bad)
        assert isinstance(result, str) and result


# --- sampling -------------------------------------------------------------


def acceptance_report(n_draws: int = 1000, seed: int = 0):
    """Draw n_draws parameter sets and bucket them by outcome."""
    rng = np.random.default_rng(seed)
    reasons: collections.Counter = collections.Counter()
    accepted = 0
    for _ in range(n_draws):
        p = fam.sample(rng)
        result = fam.is_valid(p)
        if result is True:
            accepted += 1
        else:
            reasons[rejection_reason(result).split(" (")[0]] += 1
    return accepted / n_draws, reasons


def test_acceptance_rate_is_healthy():
    """Below 20% the sampler should draw from a constrained space instead of
    filtering afterwards, per the plan."""
    rate, _ = acceptance_report()
    assert rate >= 0.20, f"acceptance rate {rate:.1%} is too low; constrain sample() instead"


def test_sampling_is_reproducible_from_the_seed():
    a = [fam.sample(np.random.default_rng(7)) for _ in range(3)]
    b = [fam.sample(np.random.default_rng(7)) for _ in range(3)]
    assert a == b


def test_sample_never_constructs_its_own_randomness():
    """Families take rng as an argument; a family reaching for global numpy state
    would silently break run reproducibility."""
    source = (fam.__file__ or "")
    text = open(source).read()
    assert "np.random." not in text
    assert "random.seed" not in text


# --- geometry_spec invariants ---------------------------------------------


def test_family_does_not_import_matplotlib():
    """Rendering policy stays centralized in render.py, so families must not draw."""
    text = open(fam.__file__).read()
    assert "matplotlib" not in text
    assert "pyplot" not in text


def test_spec_has_two_polygons_and_one_region():
    _, _, spec = fam.solve(params(6, 3))
    polygons = [s for s in spec if s["kind"] == "polygon"]
    regions = [s for s in spec if s["kind"] == "region"]
    assert {p["id"] for p in polygons} == {"outer", "inner"}
    assert len(regions) == 1
    assert regions[0]["operation"] == "difference"
    assert regions[0]["of"] == ["outer", "inner"]


def test_region_references_only_declared_polygon_ids():
    """verify.py resolves these ids; a dangling reference must not reach it."""
    _, _, spec = fam.solve(params(8, 4))
    ids = {s["id"] for s in spec if s["kind"] == "polygon"}
    for region in (s for s in spec if s["kind"] == "region"):
        assert set(region["of"]) <= ids


def test_polygons_have_the_right_vertex_counts_and_side_lengths():
    _, _, spec = fam.solve(params(7, 5, side_outer=4, side_inner=3))
    by_id = {s["id"]: s for s in spec if s["kind"] == "polygon"}
    assert len(by_id["outer"]["points"]) == 7
    assert len(by_id["inner"]["points"]) == 5
    for pid, expected_side in (("outer", 4.0), ("inner", 3.0)):
        pts = by_id[pid]["points"]
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            assert math.dist(a, b) == pytest.approx(expected_side)


def test_side_counts_are_implicit_no_label_carries_them():
    """n and m must be readable only by counting sides in the figure."""
    _, _, spec = fam.solve(params(6, 3, side_outer=7, side_inner=3))
    labelled = {s["text"] for s in spec if s["kind"] == "length_label"}
    assert labelled == {"7", "3"}
    assert "6" not in labelled


def test_labels_every_side_per_the_chosen_policy():
    _, _, spec = fam.solve(params(6, 3))
    labels = [s for s in spec if s["kind"] == "length_label"]
    assert len(labels) == 6 + 3
    assert all(s["draw"] is True for s in labels)
    assert sum(1 for s in labels if s["owner"] == "outer") == 6
    assert sum(1 for s in labels if s["owner"] == "inner") == 3


def test_every_label_names_a_declared_polygon_as_its_owner():
    _, _, spec = fam.solve(params(7, 4, side_outer=5, side_inner=2))
    ids = {s["id"] for s in spec if s["kind"] == "polygon"}
    for label in (s for s in spec if s["kind"] == "length_label"):
        assert label["owner"] in ids


def test_label_segments_match_actual_polygon_edges():
    _, _, spec = fam.solve(params(5, 4, side_outer=6, side_inner=3))
    for label in (s for s in spec if s["kind"] == "length_label"):
        a, b = label["segment"]
        assert math.dist(a, b) == pytest.approx(label["value"])


def test_polygons_rest_on_a_flat_edge():
    """Orientation is fixed, not sampled, so rotation cannot create near-duplicates."""
    _, _, spec = fam.solve(params(6, 3))
    for poly in (s for s in spec if s["kind"] == "polygon"):
        ys = [y for _, y in poly["points"]]
        lowest = min(ys)
        assert sum(1 for y in ys if math.isclose(y, lowest, abs_tol=1e-9)) == 2


def test_inner_polygon_is_strictly_inside_outer():
    """Containment must hold for every valid configuration, not just typical ones."""
    rng = np.random.default_rng(3)
    checked = 0
    while checked < 50:
        p = fam.sample(rng)
        if fam.is_valid(p) is not True:
            continue
        _, _, spec = fam.solve(p)
        by_id = {s["id"]: s for s in spec if s["kind"] == "polygon"}
        outer_apothem = fam.apothem(int(p["n"]), float(p["side_outer"]))
        for x, y in by_id["inner"]["points"]:
            assert math.hypot(x, y) < outer_apothem
        checked += 1


def test_registered_and_reachable_through_the_registry():
    assert "nested_polygons" in registry.available()
    assert registry.get("nested_polygons") is fam
