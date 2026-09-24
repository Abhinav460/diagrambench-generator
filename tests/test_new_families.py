"""The three families added to cover the paper's remaining Category 1 configurations.

Paper section 3.3 requires problems spanning composite figures, circular regions,
shaded areas, nested polygons, and coordinate-based layouts. ``nested_polygons``
covered the middle two; these three cover the rest.

Each family gets the same treatment ``nested_polygons`` had: hand-checkable closed
forms, a healthy acceptance rate, and the spec invariants that make the problem
diagram-dependent rather than merely illustrated.
"""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest
import sympy as sp

from generator import registry
from generator.families import composite_rectilinear as comp
from generator.families import coordinate_polygon as coord
from generator.families import inscribed_circle as circ
from generator.registry import rejection_reason
from generator.schema import stem_leaks_geometry
from generator.verify import ARC_TOLERANCE, TOLERANCE, verify

ALL_NEW = [circ, comp, coord]

#: The worst relative difference each family's end-to-end run may report. Polygon
#: families measure exactly (0.0 in practice); inscribed_circle is measured against a
#: 4096-segment polygon standing in for its circle, so it gets the same ARC_TOLERANCE
#: that ``verify`` applies to any curved spec. Worst observed: 7.2e-6.
RUN_TOLERANCE = {
    circ.NAME: ARC_TOLERANCE,
    comp.NAME: TOLERANCE,
    coord.NAME: TOLERANCE,
}


# --- inscribed_circle: hand-worked closed forms ----------------------------


def test_square_with_inscribed_circle():
    """Side 2 gives a circle of radius 1: area 4 - pi."""
    _, answer, _ = circ.solve({"k": 4, "side": 2})
    assert sp.simplify(answer - (4 - sp.pi)) == 0


def test_hexagon_with_inscribed_circle_unit_side():
    """A6 = 3*sqrt(3)/2; apothem = sqrt(3)/2 so the circle is 3*pi/4."""
    _, answer, _ = circ.solve({"k": 6, "side": 1})
    assert sp.simplify(answer - (3 * sp.sqrt(3) / 2 - 3 * sp.pi / 4)) == 0


def test_triangle_with_inscribed_circle_unit_side():
    """A3 = sqrt(3)/4; apothem = 1/(2*sqrt(3)) so the circle is pi/12."""
    _, answer, _ = circ.solve({"k": 3, "side": 1})
    assert sp.simplify(answer - (sp.sqrt(3) / 4 - sp.pi / 12)) == 0


def test_the_radius_is_the_apothem_not_the_circumradius():
    """The tangency is the whole point: a circle at the circumradius would pass
    through the vertices and spill outside the polygon, and the answer would be
    negative rather than merely wrong."""
    _, _, spec = circ.solve({"k": 6, "side": 4})
    circle = next(item for item in spec if item["kind"] == "circle")
    assert circle["radius"] == pytest.approx(circ.apothem(6, 4))
    assert circle["radius"] < circ.circumradius(6, 4)


def test_inscribed_circle_answers_are_positive():
    for k in range(circ.MIN_SIDES, circ.MAX_SIDES + 1):
        _, answer, _ = circ.solve({"k": k, "side": 3})
        assert float(answer.evalf()) > 0, f"k={k} gave a non-positive area"


def test_ring_fraction_matches_a_directly_computed_ratio():
    """Guards the closed form. An earlier version had the reciprocal of the
    tangent, which agreed only at k=4 where tan(pi/4) is 1 -- so a spot check on a
    square would have passed it."""
    for k in range(3, 9):
        _, answer, _ = circ.solve({"k": k, "side": 1})
        polygon_area = float(circ.exact_area(k, 1).evalf())
        assert circ.ring_fraction({"k": k}) == pytest.approx(
            float(answer.evalf()) / polygon_area
        )


def test_thin_rings_are_rejected():
    assert circ.ring_fraction({"k": 8}) >= circ.MIN_RING_FRACTION
    assert circ.ring_fraction({"k": 20}) < circ.MIN_RING_FRACTION


# --- composite_rectilinear: hand-worked and structural ---------------------


def test_composite_area_is_the_rectangle_minus_the_notch():
    _, answer, _ = comp.solve({"width": 10, "height": 8, "notch_w": 4, "notch_h": 3})
    assert answer == 80 - 12


def test_composite_outline_has_six_vertices():
    _, _, spec = comp.solve({"width": 10, "height": 8, "notch_w": 4, "notch_h": 3})
    polygon = next(item for item in spec if item["kind"] == "polygon")
    assert len(polygon["points"]) == 6


def test_composite_is_drawn_as_one_outline_not_two_pieces():
    """Emitting the outer rectangle and the notch separately would draw the
    decomposition, which is the thing the solver is supposed to work out."""
    _, _, spec = comp.solve({"width": 10, "height": 8, "notch_w": 4, "notch_h": 3})
    assert len([item for item in spec if item["kind"] == "polygon"]) == 1
    region = next(item for item in spec if item["kind"] == "region")
    assert region["operation"] == "union" and len(region["of"]) == 1


def test_exactly_two_composite_edges_are_withheld():
    """The unlabeled_inferrable trap: each withheld edge is the difference of two
    labelled ones, sitting adjacent to them."""
    _, _, spec = comp.solve({"width": 10, "height": 8, "notch_w": 4, "notch_h": 3})
    labels = [item for item in spec if item["kind"] == "length_label"]
    withheld = [item for item in labels if not item["draw"]]

    assert len(labels) == 6
    assert len(withheld) == 2
    assert sorted(item["value"] for item in withheld) == [5.0, 6.0]  # h-b and w-a


def test_composite_label_values_match_their_segment_lengths():
    """A label whose text disagrees with the edge it sits on is a figure that lies,
    and no amount of area checking would catch it."""
    _, _, spec = comp.solve({"width": 12, "height": 9, "notch_w": 5, "notch_h": 4})
    for item in spec:
        if item["kind"] != "length_label":
            continue
        (x1, y1), (x2, y2) = item["segment"]
        assert math.hypot(x2 - x1, y2 - y1) == pytest.approx(item["value"])


def test_composite_outline_winds_counter_clockwise():
    _, _, spec = comp.solve({"width": 10, "height": 8, "notch_w": 4, "notch_h": 3})
    points = next(item for item in spec if item["kind"] == "polygon")["points"]
    total = sum(
        points[i][0] * points[(i + 1) % 6][1] - points[(i + 1) % 6][0] * points[i][1]
        for i in range(6)
    )
    assert total > 0


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"width": 10, "height": 8, "notch_w": 10, "notch_h": 3}, "not strictly inside"),
        ({"width": 10, "height": 8, "notch_w": 1, "notch_h": 3}, "notch width"),
        ({"width": 10, "height": 8, "notch_w": 4, "notch_h": 7}, "notch height"),
        ({"width": 2, "height": 8, "notch_w": 1, "notch_h": 3}, "outer dimensions"),
    ],
)
def test_composite_rejections_name_their_cause(params, expected):
    assert expected in comp.is_valid(params)


# --- coordinate_polygon: hand-worked and structural ------------------------


def test_coordinate_area_of_a_right_triangle():
    """(0,0), (4,0), (4,3) is half of a 4x3 rectangle."""
    _, answer, _ = coord.solve({"points": [[0, 0], [4, 0], [4, 3]]})
    assert answer == 6


def test_coordinate_area_of_a_unit_offset_square():
    _, answer, _ = coord.solve({"points": [[1, 1], [4, 1], [4, 4], [1, 4]]})
    assert answer == 9


def test_coordinate_area_can_be_a_half_integer():
    """Lattice polygons have half-integer areas, which is a third answer shape
    alongside the radicals and the integers the other families produce."""
    _, answer, _ = coord.solve({"points": [[0, 0], [3, 0], [1, 3]]})
    assert answer == sp.Rational(9, 2)


def test_coordinate_answer_is_exact_not_a_float():
    _, answer, _ = coord.solve({"points": [[0, 0], [3, 0], [1, 3]]})
    assert isinstance(answer, sp.Expr)
    assert answer.is_rational


def test_some_vertices_are_labelled_and_some_are_not():
    """The withheld coordinates are what make the grid load-bearing."""
    _, _, spec = coord.solve({"points": [[0, 0], [5, 0], [5, 4], [0, 4]]})
    points = [item for item in spec if item["kind"] == "point_label"]
    assert len(points) == 4
    assert sum(1 for item in points if item["draw"]) == coord.LABELLED_VERTICES
    assert sum(1 for item in points if not item["draw"]) > 0


def test_every_vertex_is_marked_even_when_unlabelled():
    """An unmarked vertex is invisible where two edges meet at a shallow angle, and
    the task becomes guessing where the polygon turns rather than reading it."""
    _, _, spec = coord.solve({"points": [[0, 0], [5, 0], [5, 4]]})
    polygon = next(item for item in spec if item["kind"] == "polygon")
    marked = {tuple(item["point"]) for item in spec if item["kind"] == "point_label"}
    assert marked == {tuple(p) for p in polygon["points"]}


def test_the_grid_is_emitted():
    _, _, spec = coord.solve({"points": [[0, 0], [5, 0], [5, 4]]})
    axes = [item for item in spec if item["kind"] == "axes"]
    assert len(axes) == 1 and axes[0]["step"] == 1.0


def test_coordinate_rejects_collinear_vertices():
    assert "collinear" in coord.is_valid({"points": [[0, 0], [2, 0], [4, 0]]})


def test_coordinate_rejects_a_self_intersecting_quadrilateral():
    """A bowtie has no well-defined area, and the shoelace formula returns one
    anyway -- the difference of the two lobes."""
    assert coord.is_valid({"points": [[0, 0], [4, 4], [4, 0], [0, 4]]}) is not True


def test_coordinate_rejects_duplicate_vertices():
    assert "duplicate" in coord.is_valid({"points": [[0, 0], [4, 0], [4, 0]]})


def test_coordinate_rejects_slivers():
    assert "below the readable minimum" in coord.is_valid({"points": [[0, 0], [5, 0], [5, 1]]})


def test_coordinate_rejects_out_of_bounds_vertices():
    assert "outside the lattice bounds" in coord.is_valid({"points": [[0, 0], [99, 0], [4, 3]]})


def test_shoelace_agrees_with_the_exact_form():
    points = [[1, 1], [6, 2], [4, 5]]
    assert coord.shoelace([[float(x), float(y)] for x, y in points]) == pytest.approx(
        float(coord.exact_area(points).evalf())
    )


# --- every family, uniformly ------------------------------------------------


@pytest.mark.parametrize("family", ALL_NEW, ids=lambda f: f.NAME)
def test_registered_and_reachable_through_the_registry(family):
    assert registry.get(family.NAME) is family


@pytest.mark.parametrize("family", ALL_NEW, ids=lambda f: f.NAME)
def test_family_does_not_import_matplotlib(family):
    """Rendering belongs to render.py. A family that imports matplotlib has started
    deciding how it looks, and the separation stops holding."""
    import ast

    tree = ast.parse(open(family.__file__).read())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported if "matplotlib" in name]


@pytest.mark.parametrize("family", ALL_NEW, ids=lambda f: f.NAME)
def test_acceptance_rate_is_healthy(family):
    """Below 20% means the sampler should draw from a constrained space rather than
    filtering afterwards."""
    rng = np.random.default_rng(0)
    reasons: collections.Counter = collections.Counter()
    accepted = 0
    for _ in range(1000):
        reason = rejection_reason(family.is_valid(family.sample(rng)))
        if reason is None:
            accepted += 1
        else:
            reasons[reason.split("(")[0].strip()] += 1

    rate = accepted / 1000
    assert rate >= 0.20, f"{family.NAME} accepted {rate:.1%}; rejections: {reasons.most_common()}"


@pytest.mark.parametrize("family", ALL_NEW, ids=lambda f: f.NAME)
def test_every_valid_sample_verifies(family):
    """The spec-based cross-check, run across each family's own parameter space."""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(150):
        params = family.sample(rng)
        if rejection_reason(family.is_valid(params)) is not None:
            continue
        _, answer, spec = family.solve(params)
        verify(spec, answer, params=params)
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("family", ALL_NEW, ids=lambda f: f.NAME)
def test_stems_never_leak_geometry(family):
    """Category 1 requires the diagram to be the exclusive carrier. A stem naming a
    shape or stating a number breaks that."""
    rng = np.random.default_rng(0)
    for _ in range(60):
        params = family.sample(rng)
        if rejection_reason(family.is_valid(params)) is not None:
            continue
        stem, _, _ = family.solve(params)
        assert stem_leaks_geometry(stem) is None, f"{family.NAME}: {stem!r}"


@pytest.mark.parametrize("family", ALL_NEW, ids=lambda f: f.NAME)
def test_sampling_is_reproducible_from_the_seed(family):
    assert family.sample(np.random.default_rng(7)) == family.sample(np.random.default_rng(7))


@pytest.mark.parametrize("family", ALL_NEW, ids=lambda f: f.NAME)
def test_every_rejection_gives_a_reason_string(family):
    """A bare False makes a low acceptance rate undiagnosable."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        result = family.is_valid(family.sample(rng))
        assert result is True or (isinstance(result, str) and result)


@pytest.mark.parametrize("family", ALL_NEW, ids=lambda f: f.NAME)
def test_solve_declares_exactly_one_region(family):
    rng = np.random.default_rng(0)
    params = next(
        p for p in (family.sample(rng) for _ in range(200))
        if rejection_reason(family.is_valid(p)) is None
    )
    _, _, spec = family.solve(params)
    assert len([item for item in spec if item["kind"] == "region"]) == 1


# --- the coverage claim -----------------------------------------------------


def test_all_five_paper_configurations_are_covered():
    """Paper section 3.3: composite figures, circular regions, shaded areas, nested
    polygons, and coordinate-based layouts. This is the test that fails if a family
    is deleted without its configuration being picked up elsewhere."""
    registered = set(registry.available())
    assert {
        "composite_rectilinear",   # composite figures
        "inscribed_circle",        # circular regions
        "nested_polygons",         # nested polygons, and shaded areas
        "coordinate_polygon",      # coordinate-based layouts
    } <= registered


def test_the_families_produce_different_answer_shapes():
    """A benchmark whose answers all look alike lets a model pattern-match the form
    of an answer rather than derive it."""
    _, circle_answer, _ = circ.solve({"k": 6, "side": 2})
    _, composite_answer, _ = comp.solve({"width": 10, "height": 8, "notch_w": 4, "notch_h": 3})
    _, coordinate_answer, _ = coord.solve({"points": [[0, 0], [3, 0], [1, 3]]})

    assert circle_answer.has(sp.pi)
    assert composite_answer.is_Integer
    assert coordinate_answer.is_Rational and not coordinate_answer.is_Integer


@pytest.mark.parametrize("family", ALL_NEW, ids=lambda f: f.NAME)
def test_fifty_problem_end_to_end_run_renders_and_verifies(family, tmp_path):
    """A bounded production-sized run catches generator hangs missed by unit tests."""
    from generator.cli import generate
    from generator.emit import read_manifest

    out = tmp_path / family.NAME
    stats = generate(family.NAME, 50, 20260921, out)
    records = list(read_manifest(out))

    assert stats.complete
    assert stats.emitted == 50
    assert len(records) == 50
    assert all((out / record.image_path).is_file() for record in records)
    assert stats.worst_relative_difference < RUN_TOLERANCE[family.NAME]
