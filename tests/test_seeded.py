"""Seeded structured input: pinned sampling, pinned ceilings, loader, schema, driver.

A seeded row (one with ``variants``) pins some of a family's params and draws the
rest through the family's own ``sample``. The first block below is the guarantee
that makes this safe to add: with nothing pinned, every draw -- and so every
existing ``--family`` run -- is byte-identical to what it was before pinning existed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from generator import GENERATOR_VERSION, registry
from generator.cli import generate, generate_from_file, generate_mixed, main
from generator.emit import MANIFEST_NAME, read_manifest
from generator.families import composite_rectilinear as comp
from generator.families import coordinate_polygon as coord
from generator.families import inscribed_circle as circ
from generator.families import nested_polygons as nested
from generator.params import parse_params
from generator.schema import Datapoint, SchemaValidationError
from generator.structured import RowError, StructuredProblem, load_rows
from generator.verify import ARC_TOLERANCE

PINNABLE = [circ, nested, comp]


def write_rows(path: Path, rows: list) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- empty pins change nothing ------------------------------------------------------

#: Captured on main at 18334f9, before pinning existed. ``stream:`` hashes the first
#: 300 ``sample(default_rng(7))`` draws; ``manifest:`` hashes the manifest of
#: ``generate(name, 4, 7, ...)`` (``all``: ``generate_mixed`` over every family, n=8).
#: A manifest records ``generator_version``, so bumping it changes these on purpose:
#: recapture them then, from the commit before the bump, and never to make a
#: sampling change pass.
GOLDEN = {
    "stream:inscribed_circle": "ae3a304dccd2c7f1e78a058aedf7e7dc4436d52a84cc8592f04f2e7da66661f3",
    "stream:nested_polygons": "31b3202c16040e777d269cb0179b092bb4e2a9ba12c5ff611f31f4c8c849d181",
    "stream:composite_rectilinear": "fbfbe228945892a182a7a78ff74bb2fa3a5bec66ccbec7add527d33a9ebef773",
    "stream:coordinate_polygon": "128f9353b72022ff7826331126ed6d603789ad6ee247f8dfed480aa35ffcca01",
    "manifest:inscribed_circle": "d7afdd6ed6b803b5180863af5ad79648007a389ea1d51680b23554cdca746901",
    "manifest:nested_polygons": "ffc59162f2a87f4e309e6aad6d4dd07a8a63b9f97e56fcdb7d441fbc3c2099fc",
    "manifest:composite_rectilinear": "43a2e7122534fb6732f964e22310535c842ebd808e9bcad60900d33f3f5f9e64",
    "manifest:coordinate_polygon": "102de8c90d280b1877c48a495d8a6ca55fd3cf65372c301c4d1df9bbccdf30d3",
    "manifest:all": "9a7802e3221cbfd9553b4f72739e09b3590e548bbc0df417d5ace69f5f8d3a5a",
}
GOLDEN_VERSION = "0.2.0"


def _stream(family, *args) -> str:
    rng = np.random.default_rng(7)
    draws = [family.sample(rng, *args) for _ in range(300)]
    return sha256(json.dumps(draws, sort_keys=True).encode())


@pytest.mark.parametrize("family", [circ, nested, comp, coord], ids=lambda f: f.NAME)
def test_unpinned_draws_match_main(family):
    assert _stream(family) == GOLDEN[f"stream:{family.NAME}"]


@pytest.mark.parametrize("family", PINNABLE, ids=lambda f: f.NAME)
@pytest.mark.parametrize("empty", [None, {}])
def test_empty_pins_draw_exactly_what_a_plain_draw_does(family, empty):
    assert _stream(family, empty) == GOLDEN[f"stream:{family.NAME}"]


@pytest.mark.parametrize("family", PINNABLE, ids=lambda f: f.NAME)
def test_empty_pins_leave_the_rng_where_a_plain_draw_does(family):
    plain, pinned = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(50):
        family.sample(plain)
        family.sample(pinned, {})
    assert plain.bit_generator.state == pinned.bit_generator.state


@pytest.mark.skipif(GENERATOR_VERSION != GOLDEN_VERSION, reason="golden manifests predate a version bump")
@pytest.mark.parametrize("name", [f.NAME for f in [circ, nested, comp, coord]])
def test_fixed_seed_family_runs_are_byte_identical_to_main(name, tmp_path):
    generate(name, 4, 7, tmp_path)
    assert sha256((tmp_path / MANIFEST_NAME).read_bytes()) == GOLDEN[f"manifest:{name}"]


@pytest.mark.skipif(GENERATOR_VERSION != GOLDEN_VERSION, reason="golden manifests predate a version bump")
def test_fixed_seed_mixed_run_is_byte_identical_to_main(tmp_path):
    generate_mixed(registry.available(), 8, 7, tmp_path)
    assert sha256((tmp_path / MANIFEST_NAME).read_bytes()) == GOLDEN["manifest:all"]


@pytest.mark.parametrize("family", PINNABLE, ids=lambda f: f.NAME)
def test_unpinned_parameter_space_is_unchanged(family):
    assert list(family.parameter_space()) == list(family.parameter_space({}))


# --- pinned sampling ------------------------------------------------------------------


@pytest.mark.parametrize(
    "family,pinned",
    [
        (circ, {"k": 6}),
        (circ, {"side": 2.5}),
        (nested, {"n": 5}),
        (nested, {"m": 8}),
        (nested, {"n": 6, "m": 4, "side_outer": 10}),
        (comp, {"width": 10}),
        (comp, {"width": 12.5, "notch_h": 3}),
        (comp, {"notch_w": 9}),
    ],
)
def test_pinned_values_are_held_and_the_rest_drawn_from_the_same_space(family, pinned):
    rng = np.random.default_rng(11)
    space = [json.dumps(p, sort_keys=True) for p in family.parameter_space(pinned)]
    for _ in range(300):
        params = family.sample(rng, pinned)
        assert set(params) == set(family.PARAM_TYPES)
        assert {key: params[key] for key in pinned} == pinned
        assert json.dumps(params, sort_keys=True) in space


def test_nested_draws_its_inner_count_below_the_pinned_outer_count():
    rng = np.random.default_rng(0)
    assert {nested.sample(rng, {"n": 5})["m"] for _ in range(200)} == {3, 4}
    # A triangle leaves no smaller count: m is drawn as 3 and left to validation.
    assert {nested.sample(rng, {"n": 3})["m"] for _ in range(20)} == {3}


def test_composite_draws_its_notch_inside_a_pinned_width():
    rng = np.random.default_rng(0)
    assert {comp.sample(rng, {"width": 12.5})["notch_w"] for _ in range(2000)} == set(range(1, 13))
    assert {comp.sample(rng, {"width": 5})["notch_w"] for _ in range(500)} == {1, 2, 3, 4}
    # No integer notch fits under 1: drawn as 1, and rejected as geometry.
    params = comp.sample(rng, {"width": 1})
    assert params["notch_w"] == 1 and comp.is_valid(params) is not True


def test_supports_pinning():
    assert all(registry.supports_pinning(family) for family in PINNABLE)
    assert not registry.supports_pinning(coord)


# --- pinned ceilings -------------------------------------------------------------------


def test_a_pinned_ceiling_counts_only_the_pinned_space():
    assert registry.ceiling(circ, pinned={"k": 6}) == 20
    assert registry.ceiling(circ, pinned={"k": 6, "side": 10}) == 1
    assert registry.ceiling(circ, pinned={}) == registry.ceiling(circ)


def test_warn_admits_what_block_rejects():
    square_around_hexagon = {"n": 4, "m": 6}
    assert registry.ceiling(nested, pinned=square_around_hexagon) == 0
    assert registry.ceiling(nested, pinned=square_around_hexagon, readability="warn") > 0


def test_draw_verdict_blocks_exactly_as_is_valid():
    rng = np.random.default_rng(5)
    for _ in range(200):
        params = nested.sample(rng)
        reason, warnings = registry.draw_verdict(nested, params)
        assert reason == registry.rejection_reason(nested.is_valid(params)) and warnings == []


# --- loader ---------------------------------------------------------------------------

SEEDED_ROW = {"input_id": "hex", "family": "inscribed_circle", "params": {"k": 6}, "variants": 3}
ORDINARY_ROW = {
    "input_id": "sq",
    "family": "inscribed_circle",
    "params": {"k": 4, "side": 10},
    "expected_answer": "100 - 25*pi",
}


def test_a_seeded_row_loads_with_partial_params(tmp_path):
    (row,) = load_rows(write_rows(tmp_path / "p.jsonl", [SEEDED_ROW]))
    assert isinstance(row, StructuredProblem)
    assert row.seeded and row.variants == 3 and row.readability == "block"
    assert row.expected_answer is None
    assert row.variant_ids() == ["hex_000", "hex_001", "hex_002"]


def test_variant_ids_widen_past_a_thousand():
    row = StructuredProblem(1, "big", "inscribed_circle", {}, None, variants=1500)
    assert row.variant_ids()[0] == "big_0000" and row.variant_ids()[-1] == "big_1499"


@pytest.mark.parametrize("params", [{}, {"k": 6, "side": 2.5}])
def test_a_seeded_row_may_pin_nothing_or_everything(tmp_path, params):
    (row,) = load_rows(write_rows(tmp_path / "p.jsonl", [{**SEEDED_ROW, "params": params}]))
    assert isinstance(row, StructuredProblem) and row.params == params


@pytest.mark.parametrize(
    "row,reason",
    [
        ({**SEEDED_ROW, "expected_answer": "1"}, "cannot carry expected_answer"),
        ({**ORDINARY_ROW, "readability": "warn"}, "readability applies only to seeded rows"),
        ({**SEEDED_ROW, "variants": 0}, "positive integer"),
        ({**SEEDED_ROW, "variants": True}, "positive integer"),
        ({**SEEDED_ROW, "variants": "3"}, "positive integer"),
        ({**SEEDED_ROW, "readability": "maybe"}, "readability must be one of"),
        ({**SEEDED_ROW, "params": {"n": 6}}, "must be a subset"),
        ({**SEEDED_ROW, "params": {"k": 6.5}}, "k must be an integer"),
        ({**SEEDED_ROW, "family": "coordinate_polygon", "params": {}}, "does not support seeded rows"),
    ],
)
def test_bad_seeded_rows_are_reported_individually(tmp_path, row, reason):
    rows = load_rows(write_rows(tmp_path / "p.jsonl", [row, ORDINARY_ROW]))
    assert isinstance(rows[0], RowError) and reason in rows[0].reason, rows[0]
    assert isinstance(rows[1], StructuredProblem)


@pytest.mark.parametrize("seeded_first", [True, False])
def test_an_output_id_clash_is_reported_on_the_later_row(tmp_path, seeded_first):
    clashing = {**ORDINARY_ROW, "input_id": "hex_001"}
    rows = [SEEDED_ROW, clashing] if seeded_first else [clashing, SEEDED_ROW]
    loaded = load_rows(write_rows(tmp_path / "p.jsonl", rows))
    assert isinstance(loaded[0], StructuredProblem)
    assert isinstance(loaded[1], RowError) and "'hex_001' is already used" in loaded[1].reason


# --- schema ---------------------------------------------------------------------------


def seeded_record(**overrides) -> Datapoint:
    fields = dict(
        problem_id="hex_000",
        family="inscribed_circle",
        category=1,
        stem="Find the area of the shaded region.",
        image_path="images/hex_000.png",
        answer_exact="150*sqrt(3) - 75*pi",
        answer_decimal=24.2,
        origin="seeded",
        seed=3,
        params={"k": 6, "side": 10},
        signature="abc",
        generator_version=GENERATOR_VERSION,
        parent_input_id="hex",
        pinned={"k": 6},
    )
    fields.update(overrides)
    return Datapoint(**fields)


def test_a_seeded_record_validates_and_round_trips():
    record = seeded_record()
    record.validate()
    assert json.loads(record.to_json())["pinned"] == {"k": 6}
    assert Datapoint.from_json(record.to_json()) == record


def test_a_seeded_record_may_pin_nothing():
    record = seeded_record(pinned={})
    record.validate()
    assert json.loads(record.to_json())["pinned"] == {}


@pytest.mark.parametrize("missing", ["seed", "parent_input_id", "pinned", "signature"])
def test_a_seeded_record_requires_its_provenance(missing):
    with pytest.raises(SchemaValidationError, match="origin='seeded' requires"):
        seeded_record(**{missing: None}).validate()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"expected_answer": "1"}, "no known answer"),
        ({"pinned": {"k": 4}}, "does not match params"),
        ({"pinned": {"n": 6}}, "does not match params"),
        ({"params": {"k": 6}}, "exactly the keys"),
        ({"seed": True}, "seed must be an int"),
    ],
)
def test_a_seeded_record_rejects_inconsistent_fields(overrides, match):
    with pytest.raises(SchemaValidationError, match=match):
        seeded_record(**overrides).validate()


def test_seeded_fields_stay_unserialized_for_other_origins():
    record = seeded_record(origin="generated", parent_input_id=None, pinned=None)
    record.validate()
    assert "pinned" not in record.to_dict() and "parent_input_id" not in record.to_dict()


# --- driver ---------------------------------------------------------------------------


def test_a_seeded_row_emits_verified_variants_with_its_pins(tmp_path):
    path = write_rows(
        tmp_path / "p.jsonl",
        [
            {**SEEDED_ROW, "variants": 4, "source": {"site": "example"}},
            {"input_id": "L", "family": "composite_rectilinear", "params": {"width": 12.5}, "variants": 3},
        ],
    )
    stats = generate_from_file(path, tmp_path / "run", seed=3)
    assert stats.complete and stats.emitted == stats.requested == 7
    assert all(row.passed for row in stats.rows)
    assert stats.worst_relative_difference < ARC_TOLERANCE

    records = list(read_manifest(tmp_path / "run"))
    assert [r.problem_id for r in records] == ["hex_000", "hex_001", "hex_002", "hex_003", "L_000", "L_001", "L_002"]
    for record in records:
        record.validate()
        assert record.origin == "seeded" and record.seed == 3 and record.expected_answer is None
        assert (tmp_path / "run" / record.image_path).is_file()
    hexes, ells = records[:4], records[4:]
    assert all(r.params_dict["k"] == 6 and r.parent_input_id == "hex" for r in hexes)
    assert all(r.source_site == "example" for r in hexes)
    assert all(r.params_dict["width"] == 12.5 and dict(r.pinned) == {"width": 12.5} for r in ells)
    assert len({r.signature for r in records}) == 7


def _manifest_of(rows: list, tmp_path: Path, name: str, seed: int) -> list[dict]:
    out = tmp_path / name
    generate_from_file(write_rows(tmp_path / f"{name}.jsonl", rows), out, seed=seed)
    return [json.loads(line) for line in (out / MANIFEST_NAME).read_text().splitlines()]


def test_seeded_rows_are_reproducible_from_the_seed_and_independent_of_other_rows(tmp_path):
    row = {"input_id": "nest", "family": "nested_polygons", "params": {"n": 6}, "variants": 4}
    other = {"input_id": "L", "family": "composite_rectilinear", "params": {}, "variants": 2}

    first = _manifest_of([row], tmp_path, "a", seed=1)
    assert _manifest_of([row], tmp_path, "b", seed=1) == first
    assert _manifest_of([row], tmp_path, "c", seed=2) != first
    # Another family's row ahead of it neither shifts its stream nor takes its problems.
    with_other = _manifest_of([other, row], tmp_path, "d", seed=1)
    assert [r for r in with_other if r["parent_input_id"] == "nest"] == first


def test_a_seeded_row_never_repeats_an_earlier_rows_problem(tmp_path):
    rows = [
        {**ORDINARY_ROW, "params": {"k": 6, "side": 10}, "expected_answer": "150*sqrt(3) - 75*pi"},
        {**SEEDED_ROW, "params": {"k": 6}, "variants": 25},
    ]
    stats = generate_from_file(write_rows(tmp_path / "p.jsonl", rows), tmp_path / "run", seed=0)
    seeded = stats.rows[1]
    # 20 sides with k=6, one already taken by the ordinary row: stops at 19, not at max draws.
    assert seeded.status == "short" and seeded.variants.emitted == 19
    assert seeded.variants.draws < 25 * 10
    assert "only 20 unique valid variants with these pins, 1 already emitted" in seeded.detail
    signatures = [r.signature for r in read_manifest(tmp_path / "run")]
    assert len(signatures) == len(set(signatures)) == 20


def test_readability_warn_admits_pins_a_random_draw_never_would(tmp_path):
    square_around_hexagon = {"input_id": "sqhex", "family": "nested_polygons", "params": {"n": 4, "m": 6}, "variants": 3}
    blocked = generate_from_file(write_rows(tmp_path / "b.jsonl", [square_around_hexagon]), tmp_path / "b")
    assert blocked.rows[0].status == "short" and blocked.emitted == 0
    assert 'try "readability": "warn"' in blocked.rows[0].detail

    warned = generate_from_file(
        write_rows(tmp_path / "w.jsonl", [{**square_around_hexagon, "readability": "warn"}]), tmp_path / "w"
    )
    assert warned.complete and warned.emitted == 3
    assert warned.rows[0].variants.readability_warnings
    for record in read_manifest(tmp_path / "w"):
        values = parse_params(nested.PARAM_TYPES, record.params_dict)
        assert nested.inner_fits(values["n"], values["m"], values["side_outer"], values["side_inner"])


def test_cli_seeded_rows_take_seed_and_exit_two_when_only_short(tmp_path, capsys):
    short = {**SEEDED_ROW, "params": {"k": 6, "side": 10}, "variants": 2}
    path = write_rows(tmp_path / "p.jsonl", [short])
    code = main(["--from-file", str(path), "--seed", "5", "--out", str(tmp_path / "run"), "--report"])
    assert code == 2
    captured = capsys.readouterr()
    assert "SHORT  line 1  hex [inscribed_circle]  1/2 variants" in captured.out
    assert "hex: short: only 1 unique valid variants" in captured.err
    (record,) = read_manifest(tmp_path / "run")
    assert record.seed == 5


def test_cli_a_failed_row_alongside_a_short_one_exits_one(tmp_path):
    rows = [
        {**SEEDED_ROW, "params": {"k": 6, "side": 10}, "variants": 2},
        {**ORDINARY_ROW, "input_id": "wrong", "expected_answer": "1"},
    ]
    assert main(["--from-file", str(write_rows(tmp_path / "p.jsonl", rows)), "--out", str(tmp_path / "run")]) == 1


def test_a_file_without_seeded_rows_reports_as_before(tmp_path):
    stats = generate_from_file(write_rows(tmp_path / "p.jsonl", [ORDINARY_ROW]), tmp_path / "run")
    assert "short seeded rows" not in stats.report()
