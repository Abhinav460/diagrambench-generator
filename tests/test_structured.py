"""Structured problem input: typed params, split validation, schema, loader, CLI."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from generator import GENERATOR_VERSION, registry
from generator.cli import generate_from_file, main
from generator.emit import MANIFEST_NAME, read_manifest, relative_image_path
from generator.families import composite_rectilinear as comp
from generator.families import coordinate_polygon as coord
from generator.families import inscribed_circle as circ
from generator.families import nested_polygons as nested
from generator.params import ParamTypeError, length_text, parse_value
from generator.schema import Datapoint, SchemaValidationError
from generator.structured import (
    RowError,
    StructuredInputError,
    StructuredProblem,
    load_rows,
    parse_expected_answer,
)
from generator.verify import verify

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples" / "structured_problems.jsonl"
ALL = [nested, comp, coord, circ]


def write_rows(path: Path, rows: list) -> Path:
    path.write_text(
        "\n".join(row if isinstance(row, str) else json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


# --- typed params ------------------------------------------------------------


@pytest.mark.parametrize("bad", [6.5, 6.0, True, "6", None])
def test_integer_params_are_not_coerced(bad):
    with pytest.raises(ParamTypeError):
        parse_value("integer", "k", bad)


def test_lengths_are_exact_fractions_read_from_the_decimal_written():
    assert parse_value("length", "side", 2.5) == Fraction(5, 2)
    assert parse_value("length", "side", 0.1) == Fraction(1, 10)
    assert parse_value("length", "side", 10) == Fraction(10)


@pytest.mark.parametrize("bad", [True, "2.5", float("inf"), float("nan"), None])
def test_lengths_reject_non_numbers(bad):
    with pytest.raises(ParamTypeError):
        parse_value("length", "side", bad)


@pytest.mark.parametrize("bad", [[[0, 0], [4.5, 0], [4, 3]], [[0, 0, 1]], "points", [[0, "1"]]])
def test_lattice_points_must_be_integer_pairs(bad):
    with pytest.raises(ParamTypeError):
        parse_value("lattice_points", "points", bad)


@pytest.mark.parametrize("value,text", [(10, "10"), (Fraction(10), "10"), (Fraction(5, 2), "2.5"),
                                        (Fraction(1, 8), "0.125"), (Fraction(1, 3), "1/3")])
def test_length_text(value, text):
    assert length_text(value) == text


# --- the truncation bug, per family --------------------------------------------


def test_inscribed_circle_solves_a_fractional_side_exactly():
    """Regression: side=2.5 was solved, and verified, as side=2."""
    _, answer, spec = circ.solve({"k": 4, "side": 2.5})
    assert sp.simplify(answer - (sp.Rational(25, 4) - sp.Rational(25, 16) * sp.pi)) == 0
    verify(spec, answer)
    labels = [item["text"] for item in spec if item["kind"] == "length_label"]
    assert labels == ["2.5"] * 4


def test_nested_polygons_solves_fractional_sides_exactly():
    _, answer, spec = nested.solve({"n": 4, "m": 3, "side_outer": 7.5, "side_inner": 2.5})
    expected = sp.Rational(225, 4) - sp.sqrt(3) / 4 * sp.Rational(25, 4)
    assert sp.simplify(answer - expected) == 0
    verify(spec, answer)


def test_composite_solves_fractional_sides_exactly():
    _, answer, spec = comp.solve({"width": 12.5, "height": 6, "notch_w": 5, "notch_h": 2.5})
    assert answer == sp.Rational(125, 2)
    verify(spec, answer)


@pytest.mark.parametrize(
    "family,params",
    [
        (circ, {"k": 6.5, "side": 10}),
        (nested, {"n": 4.5, "m": 3, "side_outer": 10, "side_inner": 2}),
        (nested, {"n": 6, "m": 3.0, "side_outer": 10, "side_inner": 2}),
        (coord, {"points": [[0, 0], [4.5, 0], [4, 3]]}),
    ],
    ids=["circle_k", "nested_n", "nested_m_float_int", "coord_point"],
)
def test_solve_refuses_non_integer_counts_and_coordinates(family, params):
    with pytest.raises(ParamTypeError):
        family.solve(params)
    assert "malformed params" in family.is_valid(params)


# --- geometry vs readability -----------------------------------------------------


@pytest.mark.parametrize("family", ALL, ids=lambda f: f.NAME)
def test_every_family_supports_structured_input(family):
    assert set(family.PARAM_TYPES)
    assert callable(family.validation_issues)


READABILITY_ONLY = [
    (nested, {"n": 4, "m": 6, "side_outer": 8, "side_inner": 2}),
    (nested, {"n": 4, "m": 8, "side_outer": 10, "side_inner": 4}),
    (comp, {"width": 10, "height": 10, "notch_w": 9, "notch_h": 5}),
    (comp, {"width": 30, "height": 10, "notch_w": 10, "notch_h": 5}),
    (coord, {"points": [[0, 0], [6, 0], [6, 6], [3, 3], [0, 6]]}),
    (coord, {"points": [[0, 0], [5, 0], [5, 1]]}),
    (circ, {"k": 12, "side": 5}),
]


@pytest.mark.parametrize("family,params", READABILITY_ONLY)
def test_readability_issues_warn_for_structured_input_but_still_reject_draws(family, params):
    blocking, warnings = registry.classify_issues(family, params)
    assert blocking is None
    assert warnings
    # Random draws keep every check they had.
    assert family.is_valid(params) == warnings[0]
    stem, answer, spec = family.solve(params)
    verify(spec, answer)


GEOMETRY_INVALID = [
    (nested, {"n": 4, "m": 3, "side_outer": 4, "side_inner": 10}, "does not fit"),
    (nested, {"n": 2, "m": 3, "side_outer": 4, "side_inner": 1}, "side count"),
    (nested, {"n": 6, "m": 3, "side_outer": 0, "side_inner": 1}, "positive"),
    (comp, {"width": 10, "height": 8, "notch_w": 10, "notch_h": 3}, "not strictly inside"),
    (comp, {"width": 10, "height": 8, "notch_w": 0, "notch_h": 3}, "positive"),
    (coord, {"points": [[0, 0], [4, 4], [4, 0], [0, 4]]}, "self-intersecting"),
    (coord, {"points": [[0, 0], [2, 0], [4, 0]]}, "collinear"),
    (coord, {"points": [[0, 0], [4, 0], [4, 0]]}, "duplicate"),
    (coord, {"points": [[0, 0], [4, 0], [2, 0], [2, 3]]}, "collinear"),
    (circ, {"k": 2, "side": 5}, "side count"),
    (circ, {"k": 4, "side": -1}, "positive"),
]


@pytest.mark.parametrize("family,params,reason", GEOMETRY_INVALID)
def test_geometry_issues_block_structured_input(family, params, reason):
    blocking, _ = registry.classify_issues(family, params)
    assert blocking is not None and reason in blocking
    assert family.is_valid(params) is not True


def test_an_octagon_can_fit_a_square_with_a_fill_ratio_above_one():
    """The containment check is exact, not the conservative ratio bound."""
    params = {"n": 4, "m": 8, "side_outer": 10, "side_inner": 4}
    assert nested.fill_ratio(params) > 1
    assert nested.inner_fits(4, 8, 10, 4)


# --- schema ------------------------------------------------------------------------


def structured_record(**overrides):
    base = dict(
        problem_id="circle_in_square_10",
        family="inscribed_circle",
        category=1,
        params={"k": 4, "side": 10},
        stem="Find the area of the shaded region.",
        image_path=relative_image_path("circle_in_square_10"),
        answer_exact="100 - 25*pi",
        answer_decimal=21.46018366025517,
        signature="deadbeef",
        generator_version=GENERATOR_VERSION,
        origin="structured",
        expected_answer="100 - 25*pi",
    )
    base.update(overrides)
    return Datapoint(**base)


def test_a_structured_record_validates_and_round_trips():
    record = structured_record(source_site="textbook", original_answer="100 − 25π")
    record.validate()
    assert Datapoint.from_json(record.to_json()) == record


@pytest.mark.parametrize("missing", ["expected_answer", "params", "signature", "generator_version"])
def test_a_structured_record_requires_its_provenance(missing):
    with pytest.raises(SchemaValidationError, match=missing):
        structured_record(**{missing: None}).validate()


def test_a_structured_record_rejects_a_blank_expected_answer():
    with pytest.raises(SchemaValidationError, match="expected_answer"):
        structured_record(expected_answer="  ").validate()


def test_a_structured_record_has_no_seed():
    with pytest.raises(SchemaValidationError, match="seed"):
        structured_record(seed=0).validate()


@pytest.mark.parametrize(
    "params", [{"k": 4}, {"k": 4, "side": 10, "radius": 5}, {"sides": 4, "side": 10}]
)
def test_a_structured_record_needs_exactly_the_family_keys(params):
    with pytest.raises(SchemaValidationError, match="exactly the keys"):
        structured_record(params=params).validate()


def test_a_structured_record_needs_a_registered_family():
    with pytest.raises(SchemaValidationError, match="registered family"):
        structured_record(family="trapezoids").validate()


def test_expected_answer_stays_optional_and_unserialized_for_other_origins():
    """Generated manifests must stay byte-identical to those already on disk."""
    generated = structured_record(origin="generated", seed=0, expected_answer=None)
    generated.validate()
    assert "expected_answer" not in json.loads(generated.to_json())
    assert Datapoint.from_json(generated.to_json()) == generated


# --- loader ------------------------------------------------------------------------


GOOD_ROW = {
    "input_id": "sq",
    "family": "inscribed_circle",
    "params": {"k": 4, "side": 10},
    "expected_answer": "100 - 25*pi",
}


@pytest.mark.parametrize(
    "row,reason",
    [
        ({k: v for k, v in GOOD_ROW.items() if k != "expected_answer"}, "missing required keys"),
        ({**GOOD_ROW, "expected_answer": ""}, "non-empty"),
        ({**GOOD_ROW, "expected_answer": "100 - 25*x"}, "unknown symbols"),
        ({**GOOD_ROW, "expected_answer": "sqrt(-1)"}, "finite real"),
        ({**GOOD_ROW, "expected_answer": "100 -"}, "does not parse"),
        ({**GOOD_ROW, "params": {"k": 6.5, "side": 10}}, "k must be an integer"),
        ({**GOOD_ROW, "params": {"k": 4, "side": "10"}}, "side must be a number"),
        ({**GOOD_ROW, "params": {"k": 4}}, "exactly the keys"),
        ({**GOOD_ROW, "family": "trapezoids"}, "no family named"),
        ({**GOOD_ROW, "input_id": "has space"}, "input_id"),
        ({**GOOD_ROW, "input_id": "../escape"}, "input_id"),
        ({**GOOD_ROW, "colour": "red"}, "unknown keys"),
        ({**GOOD_ROW, "source": {"site": "x", "page": "3"}}, "unknown source keys"),
        ({**GOOD_ROW, "source": {"site": 3}}, "source.site"),
        ("not json", "invalid JSON"),
        ("[1, 2]", "JSON object"),
    ],
)
def test_bad_rows_are_reported_individually(tmp_path, row, reason):
    rows = load_rows(write_rows(tmp_path / "p.jsonl", [row, {**GOOD_ROW, "input_id": "ok"}]))
    assert isinstance(rows[0], RowError) and reason in rows[0].reason, rows[0]
    assert isinstance(rows[1], StructuredProblem)


def test_expected_answer_parses_exactly_as_sympify_for_the_examples():
    for line in EXAMPLES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            text = json.loads(line)["expected_answer"]
            assert parse_expected_answer(text) == sp.sympify(text), text


@pytest.mark.parametrize("text", ["2^3", "2**3", "Rational(25, 4)", "E", "cbrt(27)", ".5 + 1.", "-sqrt(2)"])
def test_expected_answer_accepts_the_documented_forms(text):
    assert parse_expected_answer(text) == sp.sympify(text)


@pytest.mark.parametrize(
    "template",
    [
        "Integer(__import__('pathlib').Path({marker!r}).write_text('x'))",
        "__import__('pathlib').Path({marker!r}).write_text('x') or 1",
        "().__class__.__base__.__subclasses__()",
        "(1).real",
        "[1][0]",
        "1 if 1 else 2",
        "(lambda: 1)()",
    ],
)
def test_expected_answer_is_never_run_as_python(tmp_path, template):
    marker = tmp_path / "executed"
    with pytest.raises(ValueError, match="does not parse"):
        parse_expected_answer(template.format(marker=str(marker)))
    assert not marker.exists()


@pytest.mark.parametrize("text", ["foo(2)", "sqrt(2) + y", "max(1, 2)", "len(pi)"])
def test_expected_answer_names_are_allowlisted(text):
    with pytest.raises(ValueError, match="unknown symbols"):
        parse_expected_answer(text)


def test_duplicate_input_ids_are_reported_on_the_later_row(tmp_path):
    rows = load_rows(write_rows(tmp_path / "p.jsonl", [GOOD_ROW, "", GOOD_ROW]))
    assert isinstance(rows[0], StructuredProblem)
    assert isinstance(rows[1], RowError) and rows[1].line == 3 and "duplicate" in rows[1].reason


def test_source_maps_onto_record_fields(tmp_path):
    row = {**GOOD_ROW, "source": {"site": "book", "url": "https://example.org/12", "problem_id": "12"}}
    (problem,) = load_rows(write_rows(tmp_path / "p.jsonl", [row]))
    assert problem.provenance() == {
        "source_site": "book",
        "source_url": "https://example.org/12",
        "source_problem_id": "12",
    }


@pytest.mark.parametrize("content", ["", "\n\n"])
def test_an_empty_file_is_an_error(tmp_path, content):
    path = tmp_path / "p.jsonl"
    path.write_text(content)
    with pytest.raises(StructuredInputError):
        load_rows(path)


def test_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(StructuredInputError):
        load_rows(tmp_path / "absent.jsonl")


# --- end to end --------------------------------------------------------------------


def test_the_example_file_emits_every_row(tmp_path):
    out = tmp_path / "run"
    stats = generate_from_file(EXAMPLES, out)

    failed = [(row.input_id, row.status, row.detail) for row in stats.rows if not row.passed]
    assert not failed
    assert stats.complete

    records = list(read_manifest(out))
    input_ids = [json.loads(line)["input_id"] for line in EXAMPLES.read_text().splitlines()]
    assert [r.problem_id for r in records] == input_ids
    for r in records:
        r.validate()
        assert r.origin == "structured" and r.seed is None
        assert (out / r.image_path).is_file()
        assert sp.simplify(sp.sympify(r.answer_exact) - sp.sympify(r.expected_answer)) == 0
    assert {r.family for r in records} == set(registry.available())

    warned = {row.input_id for row in stats.rows if row.warnings}
    assert {"nested_square_hexagon", "composite_deep_notch", "coord_arrow_pentagon"} <= warned


def test_a_wrong_expected_answer_is_not_emitted_and_frees_its_signature(tmp_path):
    rows = [
        {**GOOD_ROW, "input_id": "wrong", "expected_answer": "100 - 20*pi"},
        {**GOOD_ROW, "input_id": "right"},
        {**GOOD_ROW, "input_id": "right_again", "params": {"k": 4, "side": 10.0}},
    ]
    out = tmp_path / "run"
    stats = generate_from_file(write_rows(tmp_path / "p.jsonl", rows), out)

    assert [row.status for row in stats.rows] == ["answer_mismatch", "emitted", "duplicate"]
    assert "expected 100 - 20*pi" in stats.rows[0].detail
    assert [r.problem_id for r in read_manifest(out)] == ["right"]
    assert sorted(p.name for p in (out / "images").iterdir()) == ["right.png"]


def test_geometry_rejects_and_input_errors_do_not_stop_the_run(tmp_path):
    rows = [
        {**GOOD_ROW, "input_id": "bad_k", "params": {"k": 2, "side": 10}},
        "{broken",
        GOOD_ROW,
    ]
    stats = generate_from_file(write_rows(tmp_path / "p.jsonl", rows), tmp_path / "run")
    assert [row.status for row in stats.rows] == ["geometry_rejected", "input_error", "emitted"]
    assert stats.invalid == 2
    assert "geometry rejected     : 1" in stats.report()


def test_cli_from_file_exits_zero_when_every_row_is_emitted(tmp_path, capsys):
    code = main(["--from-file", str(EXAMPLES), "--out", str(tmp_path / "run"), "--report"])
    assert code == 0
    report = capsys.readouterr().out
    assert "emitted               : 13/13" in report
    assert "FAIL" not in report
    assert (tmp_path / "run" / MANIFEST_NAME).is_file()


def test_cli_from_file_exits_one_and_names_failing_rows(tmp_path, capsys):
    rows = [GOOD_ROW, {**GOOD_ROW, "input_id": "wrong", "params": {"k": 3, "side": 6}}]
    code = main(["--from-file", str(write_rows(tmp_path / "p.jsonl", rows)), "--out", str(tmp_path / "run")])
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "wrong: answer_mismatch" in captured.err


def test_cli_from_file_reports_an_unreadable_file(tmp_path, capsys):
    code = main(["--from-file", str(tmp_path / "absent.jsonl"), "--out", str(tmp_path / "run")])
    assert code == 1
    assert "StructuredInputError" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra", [["--n", "10"], ["--seed", "0"], ["--family", "nested_polygons"], ["--max-draws", "5"]]
)
def test_cli_from_file_refuses_draw_only_options_even_at_their_defaults(tmp_path, extra):
    with pytest.raises(SystemExit):
        main(["--from-file", str(EXAMPLES), "--out", str(tmp_path / "run"), *extra])


def test_cli_from_file_requires_out():
    with pytest.raises(SystemExit):
        main(["--from-file", str(EXAMPLES)])
