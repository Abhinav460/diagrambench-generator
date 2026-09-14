"""Stage 7 acceptance tests: byte-identical manifests from the same seed, a valid
partial manifest after an interrupted run, and a 200-problem run that reports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from generator import GENERATOR_VERSION
from generator.cli import RunStats, build_parser, generate, main
from generator.emit import (
    IMAGES_DIRNAME,
    MANIFEST_NAME,
    EmitError,
    ManifestWriter,
    image_path_for,
    problem_id_for,
    read_manifest,
    relative_image_path,
    write_manifest,
)
from generator.schema import Datapoint, SchemaValidationError

REPO = Path(__file__).resolve().parent.parent


def record(problem_id="nested_polygons_00000", **overrides):
    base = dict(
        problem_id=problem_id,
        family="nested_polygons",
        category=1,
        seed=0,
        params={"n": 6, "m": 3, "side_outer": 7, "side_inner": 3},
        stem="Find the area of the shaded region.",
        image_path=relative_image_path(problem_id),
        answer_exact="147*sqrt(3)/2 - 9*sqrt(3)/4",
        answer_decimal=119.4,
        signature="deadbeef",
        generator_version=GENERATOR_VERSION,
        origin="generated",
    )
    base.update(overrides)
    return Datapoint(**base)


# --- reproducibility -------------------------------------------------------


def test_same_seed_produces_a_byte_identical_manifest(tmp_path):
    """The Stage 7 acceptance criterion, and the claim the whole package rests on:
    seed plus generator_version reproduces a run."""
    a, b = tmp_path / "a", tmp_path / "b"
    generate("nested_polygons", 12, 0, a)
    generate("nested_polygons", 12, 0, b)

    assert (a / MANIFEST_NAME).read_bytes() == (b / MANIFEST_NAME).read_bytes()


def test_same_seed_produces_byte_identical_images(tmp_path):
    """A reproducible manifest pointing at differing images would be a reproducible
    run in name only."""
    a, b = tmp_path / "a", tmp_path / "b"
    generate("nested_polygons", 6, 3, a)
    generate("nested_polygons", 6, 3, b)

    for image in sorted((a / IMAGES_DIRNAME).iterdir()):
        assert image.read_bytes() == (b / IMAGES_DIRNAME / image.name).read_bytes()


def test_reproducible_across_processes(tmp_path):
    """Same-process equality would survive a hidden dependence on interpreter state;
    a fresh process is what rules that out."""
    local = tmp_path / "local"
    generate("nested_polygons", 8, 5, local)

    remote = tmp_path / "remote"
    subprocess.run(
        [sys.executable, "-m", "generator.cli", "--n", "8", "--seed", "5", "--out", str(remote)],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    assert (local / MANIFEST_NAME).read_bytes() == (remote / MANIFEST_NAME).read_bytes()


def test_different_seeds_produce_different_runs(tmp_path):
    """Guards the reproducibility tests from passing because the seed is ignored."""
    a, b = tmp_path / "a", tmp_path / "b"
    generate("nested_polygons", 8, 0, a)
    generate("nested_polygons", 8, 1, b)
    assert (a / MANIFEST_NAME).read_bytes() != (b / MANIFEST_NAME).read_bytes()


# --- the partial-manifest guarantee ---------------------------------------


def test_an_interrupted_run_leaves_a_valid_readable_manifest(tmp_path):
    """The reason the format is JSONL and the reason writes are flushed per line.

    A verification failure is injected partway through, standing in for a kill: the
    lines already written must be complete records, not a truncated JSON array.
    """
    out = tmp_path / "run"
    import generator.cli as cli_module

    real_verify = cli_module.verify
    calls = {"n": 0}

    def exploding_verify(spec, answer, **kwargs):
        calls["n"] += 1
        if calls["n"] > 4:
            raise cli_module.VerificationError("injected failure")
        return real_verify(spec, answer, **kwargs)

    cli_module.verify = exploding_verify
    try:
        with pytest.raises(cli_module.VerificationError):
            generate("nested_polygons", 50, 0, out)
    finally:
        cli_module.verify = real_verify

    records = list(read_manifest(out))
    assert len(records) == 4
    for r in records:
        r.validate()


def test_a_truncated_final_line_is_readable_in_non_strict_mode(tmp_path):
    """A process killed mid-write can leave half a line. Strict reading must refuse
    it, and non-strict reading must recover everything before it."""
    out = tmp_path / "run"
    generate("nested_polygons", 4, 0, out)

    path = out / MANIFEST_NAME
    path.write_text(path.read_text() + '{"problem_id": "nested_pol')

    with pytest.raises(EmitError, match="not a valid record"):
        list(read_manifest(out))
    assert len(list(read_manifest(out, strict=False))) == 4


def test_corruption_that_is_not_the_last_line_always_raises(tmp_path):
    """Non-strict mode tolerates an interrupted write, not a damaged file."""
    out = tmp_path / "run"
    generate("nested_polygons", 4, 0, out)

    path = out / MANIFEST_NAME
    lines = path.read_text().splitlines()
    lines[1] = "{not json"
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(EmitError):
        list(read_manifest(out, strict=False))


def test_every_line_is_independently_parseable(tmp_path):
    out = tmp_path / "run"
    generate("nested_polygons", 6, 0, out)
    for line in (out / MANIFEST_NAME).read_text().splitlines():
        assert json.loads(line)["origin"] == "generated"


# --- the emitted directory -------------------------------------------------


def test_each_record_has_an_image_that_exists_where_it_says(tmp_path):
    out = tmp_path / "run"
    generate("nested_polygons", 6, 0, out)

    records = list(read_manifest(out))
    for r in records:
        assert (out / r.image_path).exists(), f"{r.problem_id} has no image at {r.image_path}"
    assert len(list((out / IMAGES_DIRNAME).iterdir())) == len(records)


def test_image_paths_are_relative_and_slash_separated(tmp_path):
    """A manifest has to stay valid when the directory is zipped and moved."""
    out = tmp_path / "run"
    generate("nested_polygons", 3, 0, out)
    for r in read_manifest(out):
        assert not Path(r.image_path).is_absolute()
        assert "\\" not in r.image_path
        assert r.image_path.startswith(f"{IMAGES_DIRNAME}/")


def test_problem_ids_are_unique_zero_padded_and_sort_in_generation_order(tmp_path):
    out = tmp_path / "run"
    generate("nested_polygons", 12, 0, out)

    ids = [r.problem_id for r in read_manifest(out)]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids), "lexical order should match generation order"
    assert ids[0] == "nested_polygons_00000"


def test_signatures_in_a_run_are_all_distinct(tmp_path):
    """Dedupe is wired into the driver, not merely importable by it."""
    out = tmp_path / "run"
    generate("nested_polygons", 25, 0, out)
    signatures = [r.signature for r in read_manifest(out)]
    assert len(set(signatures)) == len(signatures)


def test_records_carry_the_generator_version(tmp_path):
    out = tmp_path / "run"
    generate("nested_polygons", 3, 0, out)
    assert all(r.generator_version == GENERATOR_VERSION for r in read_manifest(out))


def test_answer_exact_round_trips_through_sympy(tmp_path):
    """answer_exact is stored as a string; if it does not parse back it is prose,
    not an answer."""
    import sympy as sp

    out = tmp_path / "run"
    generate("nested_polygons", 5, 0, out)
    for r in read_manifest(out):
        parsed = sp.sympify(r.answer_exact)
        assert float(parsed.evalf()) == pytest.approx(r.answer_decimal, rel=1e-9)


# --- refusing to clobber ---------------------------------------------------


def test_writing_over_an_existing_manifest_is_refused(tmp_path):
    out = tmp_path / "run"
    generate("nested_polygons", 2, 0, out)
    with pytest.raises(EmitError, match="already exists"):
        generate("nested_polygons", 2, 0, out)


def test_overwrite_is_allowed_when_asked_for(tmp_path):
    out = tmp_path / "run"
    generate("nested_polygons", 2, 0, out)
    stats = generate("nested_polygons", 2, 1, out, overwrite=True)
    assert stats.emitted == 2


def test_an_invalid_record_never_reaches_the_manifest(tmp_path):
    """Validation lives in the writer so a driver cannot bypass it by forgetting."""
    with pytest.raises(SchemaValidationError):
        with ManifestWriter(tmp_path) as writer:
            writer.write(record(answer_decimal=0.0))
    assert (tmp_path / MANIFEST_NAME).read_text() == ""


def test_writer_outside_its_context_manager_raises(tmp_path):
    with pytest.raises(EmitError, match="outside its context manager"):
        ManifestWriter(tmp_path).write(record())


def test_write_manifest_batch_form_matches_the_streaming_form(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    records = [record(problem_id_for("f", i)) for i in range(3)]

    write_manifest(records, a)
    with ManifestWriter(b) as writer:
        for r in records:
            writer.write(r)

    assert (a / MANIFEST_NAME).read_bytes() == (b / MANIFEST_NAME).read_bytes()


def test_reading_a_missing_manifest_raises(tmp_path):
    with pytest.raises(EmitError, match="no manifest"):
        list(read_manifest(tmp_path))


def test_negative_problem_index_raises():
    with pytest.raises(EmitError, match="non-negative"):
        problem_id_for("f", -1)


# --- the reported statistics ----------------------------------------------


def test_stats_arithmetic_is_consistent(tmp_path):
    stats = generate("nested_polygons", 30, 0, tmp_path / "run")
    assert stats.draws == stats.invalid + stats.duplicates + stats.emitted
    assert stats.emitted == 30
    assert stats.complete


def test_report_names_every_number_the_plan_asks_for(tmp_path):
    stats = generate("nested_polygons", 20, 0, tmp_path / "run")
    text = stats.report()
    assert "acceptance rate" in text
    assert "collision rate" in text
    assert "verified" in text
    assert "rejections by cause" in text


def test_rejection_reasons_are_aggregated_by_cause_not_by_instance(tmp_path):
    """is_valid embeds actual values in its messages. Left ungrouped, a thousand
    rejections would produce close to a thousand distinct 'causes'."""
    stats = generate("nested_polygons", 40, 0, tmp_path / "run")
    assert stats.invalid > 0
    assert len(stats.rejection_reasons) <= 5
    for reason in stats.rejection_reasons:
        assert not any(ch.isdigit() for ch in reason), reason


def test_empty_stats_do_not_divide_by_zero():
    stats = RunStats(family="f", seed=0, requested=1)
    assert stats.acceptance_rate == 0.0
    assert stats.collision_rate == 0.0


def test_worst_relative_difference_is_tracked_and_tiny(tmp_path):
    from generator.verify import TOLERANCE

    stats = generate("nested_polygons", 20, 0, tmp_path / "run")
    assert 0 <= stats.worst_relative_difference < TOLERANCE


# --- exhausting the parameter space ---------------------------------------


def test_a_run_that_cannot_reach_n_stops_and_reports_short(tmp_path):
    """A family with a finite space must not spin forever when asked for more
    problems than it has."""
    stats = generate("nested_polygons", 100, 0, tmp_path / "run", max_draws=150)
    assert not stats.complete
    assert stats.draws == 150
    assert stats.emitted < 100


def test_a_short_run_still_leaves_a_valid_manifest(tmp_path):
    out = tmp_path / "run"
    stats = generate("nested_polygons", 100, 0, out, max_draws=150)
    assert len(list(read_manifest(out))) == stats.emitted


# --- the command line ------------------------------------------------------


def test_cli_run_exits_zero_and_reports(tmp_path, capsys):
    code = main(["--n", "5", "--seed", "0", "--out", str(tmp_path / "run"), "--report"])
    assert code == 0
    assert "acceptance rate" in capsys.readouterr().out


def test_cli_without_report_prints_nothing_on_stdout(tmp_path, capsys):
    main(["--n", "3", "--seed", "0", "--out", str(tmp_path / "run")])
    assert capsys.readouterr().out == ""


def test_cli_short_run_exits_two(tmp_path):
    code = main(
        ["--n", "100", "--seed", "0", "--out", str(tmp_path / "run"), "--max-draws", "120"]
    )
    assert code == 2, "a short run is a distinct outcome from a failed one"


def test_cli_unknown_family_exits_one(tmp_path, capsys):
    code = main(["--family", "no_such_family", "--n", "2", "--out", str(tmp_path / "run")])
    assert code == 1
    assert "unknown family" in capsys.readouterr().err


def test_cli_refusing_to_clobber_exits_one(tmp_path, capsys):
    out = str(tmp_path / "run")
    assert main(["--n", "2", "--out", out]) == 0
    assert main(["--n", "2", "--out", out]) == 1
    assert "already exists" in capsys.readouterr().err


def test_cli_overwrite_flag_succeeds(tmp_path):
    out = str(tmp_path / "run")
    assert main(["--n", "2", "--out", out]) == 0
    assert main(["--n", "2", "--out", out, "--overwrite"]) == 0


def test_cli_list_families_needs_no_out_directory(capsys):
    assert main(["--list-families"]) == 0
    assert "nested_polygons" in capsys.readouterr().out


def test_cli_rejects_a_non_positive_n(tmp_path):
    with pytest.raises(SystemExit):
        main(["--n", "0", "--out", str(tmp_path / "run")])


def test_cli_requires_out_when_generating():
    with pytest.raises(SystemExit):
        main(["--n", "2"])


def test_cli_defaults_are_the_documented_ones():
    args = build_parser().parse_args(["--out", "x"])
    assert args.family == "nested_polygons"
    assert args.seed == 0
    assert args.n == 10


@pytest.mark.slow
def test_the_headline_run_of_200_completes(tmp_path):
    """The plan's acceptance criterion: --n 200 --seed 0 completes and reports."""
    stats = generate("nested_polygons", 200, 0, tmp_path / "run")
    assert stats.complete
    assert stats.emitted == 200
    assert len(list(read_manifest(tmp_path / "run"))) == 200


# --- generating across every family -----------------------------------------


def test_family_all_emits_from_every_registered_family(tmp_path):
    from generator import registry
    from generator.cli import generate_mixed

    out = tmp_path / "run"
    stats = generate_mixed(registry.available(), 40, 0, out)

    assert stats.emitted == 40
    assert set(stats.per_family) == set(registry.available()), (
        f"some family emitted nothing: {dict(stats.per_family)}"
    )


def test_mixed_problem_ids_are_unique_and_dense_within_each_family(tmp_path):
    """Ids are numbered per family rather than globally, so a family's images are
    contiguous in a directory listing instead of scattered through the run."""
    from generator import registry
    from generator.cli import generate_mixed

    out = tmp_path / "run"
    generate_mixed(registry.available(), 40, 0, out)

    by_family: dict[str, list[str]] = {}
    for record in read_manifest(out):
        by_family.setdefault(record.family, []).append(record.problem_id)

    all_ids = [pid for ids in by_family.values() for pid in ids]
    assert len(set(all_ids)) == len(all_ids)
    for family, ids in by_family.items():
        expected = [problem_id_for(family, i) for i in range(len(ids))]
        assert sorted(ids) == expected


def test_mixed_runs_are_reproducible(tmp_path):
    from generator import registry
    from generator.cli import generate_mixed

    a, b = tmp_path / "a", tmp_path / "b"
    generate_mixed(registry.available(), 24, 0, a)
    generate_mixed(registry.available(), 24, 0, b)
    assert (a / MANIFEST_NAME).read_bytes() == (b / MANIFEST_NAME).read_bytes()


def test_each_family_draws_from_its_own_stream(tmp_path):
    """Per-family RNGs mean adding a family does not renumber the others' output.

    Checked by generating a subset and confirming those families produce exactly
    what they produced in the full run -- which a single shared stream would break.
    """
    from generator.cli import generate_mixed

    full = tmp_path / "full"
    generate_mixed(["nested_polygons", "inscribed_circle"], 30, 0, full)
    full_nested = [
        r.params_dict for r in read_manifest(full) if r.family == "nested_polygons"
    ]

    alone = tmp_path / "alone"
    generate_mixed(["nested_polygons"], len(full_nested), 0, alone)
    alone_nested = [r.params_dict for r in read_manifest(alone)]

    assert full_nested == alone_nested


def test_every_record_in_a_mixed_run_verifies_and_is_category_one(tmp_path):
    from generator import registry
    from generator.cli import generate_mixed
    from generator.schema import stem_leaks_geometry

    out = tmp_path / "run"
    generate_mixed(registry.available(), 30, 0, out)
    for record in read_manifest(out):
        record.validate()
        assert record.category == 1
        assert stem_leaks_geometry(record.stem) is None


def test_cli_family_all_runs(tmp_path, capsys):
    code = main(["--family", "all", "--n", "16", "--seed", "0", "--out", str(tmp_path / "r"), "--report"])
    assert code == 0
    assert "emitted by family" in capsys.readouterr().out
