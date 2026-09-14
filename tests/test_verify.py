"""Stage 5 acceptance tests: the verifier passes on everything the family
currently generates, and actually fires when the figure and the answer disagree.

The second half matters more than the first. A verifier that only ever returns
success is indistinguishable from no verifier at all, so most of what follows
constructs specs that are wrong in a specific way and asserts the wrongness is
caught -- including the one failure mode a params-based check could never see.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import sympy as sp

from generator import registry
from generator.families import nested_polygons as fam
from generator.registry import rejection_reason
from generator.verify import (
    ARC_RESOLUTION,
    ARC_TOLERANCE,
    TOLERANCE,
    VerificationError,
    measure,
    relative_difference,
    verify,
)


def params(n=6, m=3, side_outer=7, side_inner=3):
    return {"n": n, "m": m, "side_outer": side_outer, "side_inner": side_inner}


def solved(**kwargs):
    """(stem, answer, spec) for one hand-chosen, comfortably valid configuration."""
    return fam.solve(params(**kwargs))


# --- passes on what the family actually produces --------------------------


def test_verifies_every_valid_sample_in_a_seeded_run():
    """The Stage 5 acceptance criterion: no currently generating problem fails.

    Reported rather than merely asserted -- the worst relative difference across a
    run is the number that says whether the tolerance has real headroom or is
    quietly about to start failing.
    """
    rng = np.random.default_rng(0)
    verified = 0
    worst = 0.0
    for _ in range(200):
        candidate = fam.sample(rng)
        if rejection_reason(fam.is_valid(candidate)) is not None:
            continue
        _, answer, spec = fam.solve(candidate)
        worst = max(worst, verify(spec, answer, params=candidate))
        verified += 1

    assert verified > 0, "sampler produced nothing valid, so nothing was verified"
    assert worst < TOLERANCE / 1000, (
        f"verified {verified} problems but the worst relative difference was {worst:.3e}, "
        f"uncomfortably close to tolerance {TOLERANCE:.1e}"
    )


@pytest.mark.parametrize("n,m", [(6, 3), (8, 4), (9, 6), (12, 5), (5, 3), (4, 3)])
def test_verifies_the_hand_worked_and_awkward_side_counts(n, m):
    """Includes n=9, whose area keeps a cot(pi/9) that never reduces to a radical,
    so the float path is exercised on an answer sympy cannot simplify away."""
    _, answer, spec = solved(n=n, m=m)
    assert verify(spec, answer, params=params(n=n, m=m)) < TOLERANCE


def test_returns_the_relative_difference_rather_than_just_passing():
    _, answer, spec = solved()
    result = verify(spec, answer)
    assert isinstance(result, float)
    assert 0.0 <= result < TOLERANCE


# --- fires when the drawing disagrees with the answer ---------------------


def test_perturbing_one_coordinate_raises():
    """The plan's named case: corrupt the spec, not the params, and the verifier
    must notice that the figure no longer measures what the answer claims."""
    _, answer, spec = solved()
    corrupted = copy.deepcopy(spec)
    corrupted[0]["points"][0][0] += 0.5

    with pytest.raises(VerificationError):
        verify(corrupted, answer, params=params())


def test_the_answer_still_verifies_before_the_perturbation():
    """Guards the test above from passing for the wrong reason -- a spec that was
    already failing would make the corruption look effective when it was not."""
    _, answer, spec = solved()
    assert verify(spec, answer) < TOLERANCE
    corrupted = copy.deepcopy(spec)
    corrupted[0]["points"][0][0] += 0.5
    with pytest.raises(VerificationError):
        verify(corrupted, answer)


def test_a_figure_drawn_from_different_params_than_the_answer_raises():
    """The failure mode that motivates consuming geometry_spec instead of params.

    Here the params and the answer are mutually consistent, and a params-based
    rebuild would re-derive the same number and report success. Only measuring the
    spec catches that the inner polygon drawn is not the one the answer subtracts.
    """
    _, answer, _ = solved(n=6, m=3, side_outer=7, side_inner=3)
    _, _, wrong_spec = solved(n=6, m=4, side_outer=7, side_inner=3)

    with pytest.raises(VerificationError):
        verify(wrong_spec, answer, params=params(n=6, m=3))


def test_swapping_which_polygon_is_subtracted_raises():
    """Inner and outer exchanged: same two shapes, same params, wrong region."""
    _, answer, spec = solved()
    swapped = copy.deepcopy(spec)
    region = next(item for item in swapped if item["kind"] == "region")
    region["of"] = list(reversed(region["of"]))

    with pytest.raises(VerificationError):
        verify(swapped, answer, params=params())


def test_scaling_the_whole_figure_raises():
    """A uniform scale keeps the figure looking correct and every proportion intact
    while making the area wrong by the square of the factor."""
    _, answer, spec = solved()
    scaled = copy.deepcopy(spec)
    for item in scaled:
        if item["kind"] == "polygon":
            item["points"] = [[x * 1.01, y * 1.01] for x, y in item["points"]]

    with pytest.raises(VerificationError):
        verify(scaled, answer)


def test_tolerance_does_not_fire_on_floating_point_noise():
    """The counterpart to the corruption tests: a perturbation far below tolerance
    must pass, or the verifier would reject correct problems for rounding."""
    _, answer, spec = solved()
    nudged = copy.deepcopy(spec)
    nudged[0]["points"][0][0] += 1e-15

    assert verify(nudged, answer) < TOLERANCE


# --- the error message carries what a debugger needs ----------------------


def test_error_message_names_params_both_values_and_the_difference():
    _, answer, spec = solved()
    corrupted = copy.deepcopy(spec)
    corrupted[0]["points"][0][1] += 2.0

    with pytest.raises(VerificationError) as excinfo:
        verify(corrupted, answer, params=params())
    message = str(excinfo.value)

    assert "'n': 6" in message and "'side_outer': 7" in message
    assert "measured=" in message and "expected=" in message
    assert "relative difference=" in message and "tolerance=" in message


def test_error_message_is_explicit_when_params_were_not_supplied():
    """params is optional, so the message has to say the params are unknown rather
    than print a bare None that reads like the params were literally None."""
    _, answer, spec = solved()
    corrupted = copy.deepcopy(spec)
    corrupted[0]["points"][0][1] += 2.0

    with pytest.raises(VerificationError, match="<unknown>"):
        verify(corrupted, answer)


# --- malformed specs are rejected, not silently measured ------------------


def test_spec_with_no_region_raises():
    """Nothing to measure is a broken family, not an area of zero."""
    _, answer, spec = solved()
    without_region = [item for item in spec if item["kind"] != "region"]

    with pytest.raises(VerificationError, match="no region"):
        measure(without_region)


def test_region_referencing_an_undeclared_polygon_raises():
    _, _, spec = solved()
    broken = copy.deepcopy(spec)
    region = next(item for item in broken if item["kind"] == "region")
    region["of"] = ["outer", "nonexistent"]

    with pytest.raises(VerificationError, match="undeclared"):
        measure(broken)


def test_unsupported_region_operation_raises():
    _, _, spec = solved()
    broken = copy.deepcopy(spec)
    next(item for item in broken if item["kind"] == "region")["operation"] = "xor"

    with pytest.raises(VerificationError, match="unsupported region operation"):
        measure(broken)


def test_a_missing_operation_raises_rather_than_defaulting():
    """Defaulting to difference would let a family that forgot to state its
    operation get a plausible-looking answer for the wrong region."""
    _, _, spec = solved()
    broken = copy.deepcopy(spec)
    del next(item for item in broken if item["kind"] == "region")["operation"]

    with pytest.raises(VerificationError, match="unsupported region operation"):
        measure(broken)


# --- the boolean operations ------------------------------------------------


def two_squares():
    """Two 4x4 squares overlapping in a 2x2 corner: union 28, intersection 4."""
    return [
        {"kind": "polygon", "id": "a", "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]},
        {"kind": "polygon", "id": "b", "points": [[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]]},
    ]


def test_union_measures_the_combined_area():
    spec = two_squares() + [{"kind": "region", "operation": "union", "of": ["a", "b"]}]
    assert measure(spec) == pytest.approx(28.0)


def test_intersection_measures_the_overlap():
    spec = two_squares() + [{"kind": "region", "operation": "intersection", "of": ["a", "b"]}]
    assert measure(spec) == pytest.approx(4.0)


def test_union_is_not_the_sum_of_parts_when_shapes_overlap():
    """Inclusion-exclusion is the whole point of a composite figure: a family that
    added the two areas would be wrong by exactly the overlap."""
    spec = two_squares() + [{"kind": "region", "operation": "union", "of": ["a", "b"]}]
    assert measure(spec) != pytest.approx(32.0)


def test_duplicate_shape_ids_raise():
    """Two shapes sharing an id means a region's reference is ambiguous, and the
    later one would silently win."""
    spec = two_squares()
    spec[1]["id"] = "a"
    spec.append({"kind": "region", "operation": "union", "of": ["a", "a"]})
    with pytest.raises(VerificationError, match="duplicate shape id"):
        measure(spec)


def test_a_union_of_one_shape_is_that_shape():
    """How a family declares "the region is this single outline" -- a composite
    figure drawn as one path rather than as its construction pieces."""
    spec = [
        {"kind": "polygon", "id": "a", "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]},
        {"kind": "region", "operation": "union", "of": ["a"]},
    ]
    assert measure(spec) == pytest.approx(16.0)


def test_a_difference_of_one_shape_still_raises():
    """Difference is genuinely binary; one operand means the family forgot one."""
    spec = [
        {"kind": "polygon", "id": "a", "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]},
        {"kind": "region", "operation": "difference", "of": ["a"]},
    ]
    with pytest.raises(VerificationError, match="at least 2 shape"):
        measure(spec)


def test_a_polygon_with_too_few_points_raises():
    spec = [
        {"kind": "polygon", "id": "a", "points": [[0.0, 0.0], [1.0, 0.0]]},
        {"kind": "polygon", "id": "b", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]},
        {"kind": "region", "operation": "union", "of": ["a", "b"]},
    ]
    with pytest.raises(VerificationError, match="at least three points"):
        measure(spec)


def test_difference_of_a_single_polygon_raises():
    _, _, spec = solved()
    broken = copy.deepcopy(spec)
    next(item for item in broken if item["kind"] == "region")["of"] = ["outer"]

    with pytest.raises(VerificationError, match="at least 2 shape"):
        measure(broken)


def test_self_intersecting_polygon_raises():
    """A bowtie has no well-defined area, and shapely would quietly return one."""
    _, _, spec = solved()
    broken = copy.deepcopy(spec)
    outer = next(item for item in broken if item.get("id") == "outer")
    outer["points"][0], outer["points"][2] = outer["points"][2], outer["points"][0]

    with pytest.raises(VerificationError, match="not a valid simple polygon"):
        measure(broken)


def test_unknown_region_measure_raises():
    _, _, spec = solved()
    broken = copy.deepcopy(spec)
    next(item for item in broken if item["kind"] == "region")["measure"] = "volume"

    with pytest.raises(VerificationError, match="unknown region measure"):
        measure(broken)


# --- measure's own behaviour ----------------------------------------------


def test_measure_defaults_to_area_and_matches_the_closed_form():
    _, answer, spec = solved()
    assert measure(spec) == pytest.approx(float(answer.evalf()), rel=1e-12)


def test_measure_sums_multiple_regions():
    """A family that shades two disjoint pieces and asks for the total needs no
    special handling, per measure's docstring. This is what proves that."""
    square = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
    hole = [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
    far = [[10.0, 0.0], [12.0, 0.0], [12.0, 2.0], [10.0, 2.0]]
    far_hole = [[10.5, 0.5], [11.0, 0.5], [11.0, 1.0], [10.5, 1.0]]
    spec = [
        {"kind": "polygon", "id": "a", "points": square},
        {"kind": "polygon", "id": "a_hole", "points": hole},
        {"kind": "polygon", "id": "b", "points": far},
        {"kind": "polygon", "id": "b_hole", "points": far_hole},
        {"kind": "region", "operation": "difference", "of": ["a", "a_hole"]},
        {"kind": "region", "operation": "difference", "of": ["b", "b_hole"]},
    ]
    # (16 - 1) + (4 - 0.25)
    assert measure(spec) == pytest.approx(18.75)


def test_measure_supports_length():
    """Perimeter of a 4x4 square with a 1x1 hole: 16 outside plus 4 inside."""
    spec = [
        {"kind": "polygon", "id": "a", "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]},
        {"kind": "polygon", "id": "b", "points": [[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]},
        {"kind": "region", "operation": "difference", "of": ["a", "b"], "measure": "length"},
    ]
    assert measure(spec) == pytest.approx(20.0)


def test_difference_subtracts_every_listed_polygon():
    """``of`` beyond two entries unions the subtrahends rather than ignoring them."""
    spec = [
        {"kind": "polygon", "id": "a", "points": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]},
        {"kind": "polygon", "id": "b", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]},
        {"kind": "polygon", "id": "c", "points": [[3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 4.0]]},
        {"kind": "region", "operation": "difference", "of": ["a", "b", "c"]},
    ]
    assert measure(spec) == pytest.approx(14.0)


# --- answers arrive as sympy, floats, or ints -----------------------------


def test_accepts_an_exact_sympy_answer_without_pre_evaluating():
    _, answer, spec = solved()
    assert isinstance(answer, sp.Expr)
    assert verify(spec, answer) < TOLERANCE


def test_accepts_a_plain_float_answer():
    _, answer, spec = solved()
    assert verify(spec, float(answer.evalf())) < TOLERANCE


def test_relative_difference_floors_the_denominator_at_one():
    """Without the floor, a near-zero expected value turns any absolute error into
    an enormous relative one and the tolerance stops meaning anything."""
    assert relative_difference(0.5, 0.0) == pytest.approx(0.5)
    assert relative_difference(101.0, 100.0) == pytest.approx(0.01)


# --- the curved path -------------------------------------------------------
# No family emits a circle yet, so these specs are written by hand. They exist to
# keep ARC_TOLERANCE and ARC_RESOLUTION honest: constants describing a tolerance
# nothing can reach would be indistinguishable from constants describing a
# capability that does not exist.


def annulus(outer_r=5.0, inner_r=2.0):
    """Concentric circles, so the expected area is pi*(R^2 - r^2) exactly."""
    return [
        {"kind": "circle", "id": "outer", "center": [0.0, 0.0], "radius": outer_r},
        {"kind": "circle", "id": "inner", "center": [0.0, 0.0], "radius": inner_r},
        {"kind": "region", "operation": "difference", "of": ["outer", "inner"]},
    ]


def test_an_annulus_verifies_against_its_exact_area():
    expected = sp.pi * (sp.Integer(25) - sp.Integer(4))
    assert verify(annulus(), expected) < ARC_TOLERANCE


def test_a_circle_minus_an_inscribed_square_verifies():
    """Mixes a curved and a straight primitive in one region, which is the case
    that would break if circles were special-cased at the point of use."""
    half = 5.0 / 2 ** 0.5
    spec = [
        {"kind": "circle", "id": "c", "center": [0.0, 0.0], "radius": 5.0},
        {
            "kind": "polygon",
            "id": "s",
            "points": [[-half, -half], [half, -half], [half, half], [-half, half]],
        },
        {"kind": "region", "operation": "difference", "of": ["c", "s"]},
    ]
    assert verify(spec, 25 * sp.pi - 50) < ARC_TOLERANCE


def test_verify_selects_the_looser_tolerance_for_a_curved_spec():
    """The branch this exercises was previously unreachable. A buffered circle
    understates area by about 6.3e-6, which is far outside TOLERANCE and
    comfortably inside ARC_TOLERANCE -- so the auto-selection is what makes a
    correct curved spec pass, and forcing the strict value must fail it."""
    spec = annulus()
    expected = sp.pi * 21

    difference = verify(spec, expected)
    assert TOLERANCE < difference < ARC_TOLERANCE

    with pytest.raises(VerificationError):
        verify(spec, expected, tolerance=TOLERANCE)


def test_the_buffer_understates_area_by_the_documented_amount():
    """Pins the error model in the module docstring: the inscribed approximation
    is low by roughly 2*pi^2 / (3 * (4N)^2). If ARC_RESOLUTION is ever lowered,
    this is what says whether ARC_TOLERANCE still has its margin."""
    segments = 4 * ARC_RESOLUTION
    predicted = 2 * 3.141592653589793 ** 2 / (3 * segments ** 2)

    measured = measure(annulus(outer_r=5.0, inner_r=2.0))
    exact = 3.141592653589793 * 21
    shortfall = (exact - measured) / exact

    assert shortfall > 0, "buffer should inscribe, and so understate, the true area"
    assert shortfall == pytest.approx(predicted, rel=0.05)
    assert shortfall < ARC_TOLERANCE / 10


def test_a_wrong_annulus_still_raises():
    """The looser tolerance must not be so loose that it stops catching errors."""
    with pytest.raises(VerificationError):
        verify(annulus(), sp.pi * 20)


def test_circle_with_non_positive_radius_raises():
    spec = annulus()
    spec[0]["radius"] = 0.0
    with pytest.raises(VerificationError, match="non-positive radius"):
        measure(spec)


@pytest.mark.parametrize("kind", ["arc", "sector"])
def test_curved_kinds_without_a_defined_format_still_raise(kind):
    """A circle is center plus radius and nothing else, so implementing it guessed
    at very little. An arc needs start and end angles and a sense of direction,
    which is a real format decision -- so it still belongs to the first family
    that needs one."""
    spec = [
        {"kind": "polygon", "id": "a", "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]},
        {"kind": kind, "id": "c", "center": [0.0, 0.0], "radius": 1.0},
        {"kind": "region", "operation": "difference", "of": ["a", "c"]},
    ]
    with pytest.raises(VerificationError, match="no defined format yet"):
        measure(spec)


def test_arc_tolerance_is_looser_than_the_straight_edged_one():
    """The two constants exist to be different; equal values would mean the curved
    path silently inherits a tolerance a buffered circle cannot meet."""
    assert ARC_TOLERANCE > TOLERANCE
    assert ARC_RESOLUTION >= 64


def test_an_explicit_tolerance_overrides_the_default():
    """A caller can loosen the check, which is how the arc path will eventually be
    exercised before the automatic branch has anything to detect."""
    _, answer, spec = solved()
    loosened = copy.deepcopy(spec)
    loosened[0]["points"][0][0] += 1e-6

    with pytest.raises(VerificationError):
        verify(loosened, answer)
    assert verify(loosened, answer, tolerance=ARC_TOLERANCE) < ARC_TOLERANCE


# --- the module boundary --------------------------------------------------


def test_verify_does_not_import_the_family_it_checks():
    """verify must stay family-agnostic: it reads the spec format, not the module
    that produced it. An import here would be the first step back toward a
    params-based check.

    Inspects the import statements rather than the source text, so the module
    docstring stays free to name the family that defined the spec format.
    """
    import ast
    import generator.verify as verify_module

    tree = ast.parse(open(verify_module.__file__).read())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    offenders = [name for name in imported if "families" in name]
    assert not offenders, f"verify imports a family module: {offenders}"


def test_verify_works_on_a_spec_it_was_handed_with_no_params_at_all():
    """params is for the error message only. Verification itself must not need it."""
    _, answer, spec = solved()
    assert verify(spec, answer, params=None) < TOLERANCE
