"""A run cannot ask a family for more problems than it has: enumerable families
expose a ceiling, the rest are guarded by a duplicate-streak limit, and a run past
the ceiling is warned, stopped, and reported rather than left burning draws."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from generator import cli, registry
from generator.cli import generate, generate_mixed, main
from generator.emit import read_manifest

REPO = Path(__file__).resolve().parent.parent

# Facts about the current MIN_/MAX_ constants; a change to them should change these.
CEILINGS = {"inscribed_circle": 120, "nested_polygons": 9334, "composite_rectilinear": 11449}


@pytest.fixture
def no_render(monkeypatch):
    """Skip matplotlib so a 120-emit run takes a second rather than half a minute."""
    monkeypatch.setattr(cli, "render", lambda spec, path, stem: None)


# --- the ceiling itself -------------------------------------------------------


@pytest.mark.parametrize("name,expected", sorted(CEILINGS.items()))
def test_enumerable_families_expose_their_ceiling(name, expected):
    assert registry.ceiling(registry.get(name)) == expected


def test_coordinate_polygon_has_no_enumerable_ceiling():
    assert registry.ceiling(registry.get("coordinate_polygon")) is None


def test_ceiling_counts_only_valid_distinct_parameter_sets():
    family = registry.get("nested_polygons")
    raw = list(family.parameter_space())
    valid = [p for p in raw if registry.rejection_reason(family.is_valid(p)) is None]
    assert len(raw) == 18000 and len(valid) == CEILINGS["nested_polygons"] < len(raw)
    assert len({tuple(sorted(p.items())) for p in valid}) == len(valid)


# --- a single family past its ceiling -----------------------------------------


def test_run_past_the_ceiling_warns_stops_at_the_ceiling_and_leaves_a_valid_manifest(
    tmp_path, capsys, no_render
):
    stats = generate("inscribed_circle", 200, 0, tmp_path / "run")
    assert stats.emitted == 120 and not stats.complete
    assert stats.draws < 1000, "must stop at the ceiling, not burn draws to max_draws"
    assert stats.ceilings == {"inscribed_circle": 120}
    assert stats.stopped_early == (
        "inscribed_circle has emitted all 120 of its unique valid combinations; "
        "stopping at 120/200 emitted"
    )
    assert capsys.readouterr().err.splitlines() == [
        "warning: inscribed_circle has 120 unique valid combinations; n=200 requested "
        "-- will emit at most 120 and then stop"
    ]
    assert len(list(read_manifest(tmp_path / "run"))) == 120
    assert "stopped early         : inscribed_circle has emitted all 120" in stats.report()


def test_run_at_the_ceiling_completes_without_warning(tmp_path, capsys, no_render):
    stats = generate("inscribed_circle", 120, 0, tmp_path / "run")
    assert stats.complete and stats.emitted == 120 and stats.stopped_early is None
    assert capsys.readouterr().err == ""


#: Seed and params of the n=10 runs in output/final, written 2026-09-10, before
#: ceilings existed (9b834c5). Only what the test compares is kept: output/ is
#: gitignored, so reading the manifests directly only ever passed on one machine.
GOLDEN = json.loads((REPO / "tests" / "data" / "ceiling_golden_params.json").read_text())


@pytest.mark.parametrize("name", sorted(registry.available()))
def test_run_under_the_ceiling_is_unchanged(name, tmp_path, capsys, no_render):
    """Same seed, same params as before ceilings existed: the saved n=10 runs."""
    golden = GOLDEN[name]
    stats = generate(name, len(golden["params"]), golden["seed"], tmp_path / "run")
    assert stats.complete and stats.stopped_early is None
    assert capsys.readouterr().err == ""
    assert [r.params_dict for r in read_manifest(tmp_path / "run")] == golden["params"]


# --- an unknown ceiling: the duplicate-streak guard --------------------------


def test_unknown_ceiling_stops_after_a_duplicate_streak(tmp_path, monkeypatch, no_render):
    family = registry.get("coordinate_polygon")
    rng = np.random.default_rng(0)
    fixed = family.sample(rng)
    while registry.rejection_reason(family.is_valid(fixed)) is not None:
        fixed = family.sample(rng)
    monkeypatch.setattr(family, "sample", lambda rng: fixed)
    monkeypatch.setattr(cli, "DUPLICATE_STREAK_LIMIT", 50)

    stats = generate("coordinate_polygon", 10, 0, tmp_path / "run")
    assert stats.emitted == 1 and stats.draws == 51 and not stats.complete
    assert stats.stopped_early == (
        "no new unique problems found after 50 consecutive duplicate or rejected draws; "
        "stopping early at 1/10 emitted"
    )


def test_known_ceiling_is_not_second_guessed_by_the_streak_guard(tmp_path, monkeypatch, no_render):
    """Near a known ceiling the last problems legitimately take long streaks to find."""
    family = registry.get("inscribed_circle")
    fixed = {"k": 4, "side": 10}
    monkeypatch.setattr(family, "sample", lambda rng: fixed)
    monkeypatch.setattr(cli, "DUPLICATE_STREAK_LIMIT", 50)
    stats = generate("inscribed_circle", 2, 0, tmp_path / "run", max_draws=200)
    assert stats.draws == 200 and stats.stopped_early is None


# --- mixed runs ---------------------------------------------------------------


def _fake_ceilings(monkeypatch, fake: dict[str, int]):
    real = registry.ceiling
    monkeypatch.setattr(
        registry, "ceiling", lambda fam, **kw: fake.get(fam.NAME, real(fam, **kw))
    )


def test_mixed_run_moves_on_when_one_family_hits_its_ceiling(tmp_path, monkeypatch, capsys, no_render):
    _fake_ceilings(monkeypatch, {"inscribed_circle": 3})
    stats = generate_mixed(["inscribed_circle", "composite_rectilinear"], 10, 0, tmp_path / "run")
    assert stats.complete and stats.stopped_early is None
    assert stats.per_family == {"inscribed_circle": 3, "composite_rectilinear": 7}
    assert capsys.readouterr().err.splitlines() == [
        "warning: inscribed_circle has 3 unique valid combinations; its even share of n=10 "
        "across 2 families is 5 -- will emit at most 3 from it and let the other families "
        "fill the rest"
    ]


def test_mixed_run_stops_when_every_family_is_exhausted(tmp_path, monkeypatch, no_render):
    _fake_ceilings(monkeypatch, {"inscribed_circle": 2, "composite_rectilinear": 2})
    stats = generate_mixed(["inscribed_circle", "composite_rectilinear"], 10, 0, tmp_path / "run")
    assert stats.emitted == 4 and not stats.complete
    assert stats.stopped_early == (
        "every family is exhausted (inscribed_circle has emitted all 2 of its unique valid "
        "combinations; composite_rectilinear has emitted all 2 of its unique valid "
        "combinations); stopping at 4/10 emitted"
    )


# --- the command line ---------------------------------------------------------


def test_cli_past_the_ceiling_exits_two_with_warning_and_stop_message(tmp_path, capsys, no_render):
    code = main(
        ["--family", "inscribed_circle", "--n", "200", "--seed", "0",
         "--out", str(tmp_path / "run"), "--report"]
    )
    assert code == 2
    out, err = capsys.readouterr()
    assert err.splitlines() == [
        "warning: inscribed_circle has 120 unique valid combinations; n=200 requested "
        "-- will emit at most 120 and then stop",
        "stopped early: inscribed_circle has emitted all 120 of its unique valid "
        "combinations; stopping at 120/200 emitted",
    ]
    assert "ceiling               : inscribed_circle 120" in out
    assert "emitted               : 120/200   *** SHORT ***" in out
