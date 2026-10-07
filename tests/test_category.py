"""The category field and the Category 1 prompt-leak check.

Both close gaps between the two spec drafts: ``docs/generator_spec_draft_v1.md`` carries a
``category`` field and a ``_leaks()`` guard that ``GENERATOR_SPEC.md`` -- the file
the implementation plan names as the source of truth -- omits. The paper needs
both, since its central result is the *gap* between the two categories and its
first selection criterion is that a Category 1 diagram be the exclusive carrier of
geometric information.
"""

from __future__ import annotations

import warnings

import pytest

from generator import GENERATOR_VERSION
from generator.cli import generate
from generator.emit import read_manifest
from generator.schema import Datapoint, SchemaValidationError, StemLeakWarning, stem_leaks_geometry


def make(**overrides) -> Datapoint:
    base = dict(
        problem_id="gen-nested-0001",
        family="nested_polygons",
        category=1,
        seed=0,
        params={"n": 6, "m": 3},
        stem="Find the area of the shaded region.",
        image_path="images/gen-nested-0001.png",
        answer_exact="3*sqrt(3)/2",
        answer_decimal=2.598076211353316,
        signature="a3f1c0deadbeef",
        generator_version=GENERATOR_VERSION,
        origin="generated",
    )
    base.update(overrides)
    return Datapoint(**base)


# --- the field --------------------------------------------------------------


def test_a_category_one_record_validates():
    make().validate()


def test_a_category_two_record_validates_without_a_diagram():
    """Category 2 is textual. The schema has to be able to express it even though
    nothing generates it yet, or the field is decorative."""
    make(category=2, image_path=None, stem="Find the area of triangle ABC.").validate()


@pytest.mark.parametrize("bad", [0, 3, -1, "1", None, 1.0])
def test_an_invalid_category_is_rejected(bad):
    with pytest.raises(SchemaValidationError, match="category must be"):
        make(category=bad).validate()


def test_true_does_not_pass_as_category_one():
    """bool is a subclass of int, so True == 1 without an explicit check."""
    with pytest.raises(SchemaValidationError, match="category must be"):
        make(category=True).validate()


def test_category_survives_a_json_round_trip():
    record = make(category=2, image_path=None, stem="A textual problem.")
    assert Datapoint.from_json(record.to_json()) == record


def test_category_appears_in_the_emitted_dict():
    assert make().to_dict()["category"] == 1


# --- the coupling between category and diagram ------------------------------


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_category_one_without_a_diagram_is_rejected(missing):
    with pytest.raises(SchemaValidationError, match="requires an image_path"):
        make(image_path=missing).validate()


def test_category_two_with_a_diagram_is_rejected():
    """A Category 2 problem carrying a figure has silently become Category 1, which
    would corrupt the between-category gap the paper reports."""
    with pytest.raises(SchemaValidationError, match="must not carry a diagram"):
        make(category=2, stem="A textual problem.").validate()


# --- the leak check ---------------------------------------------------------


def test_a_clean_stem_does_not_leak():
    assert stem_leaks_geometry("Find the area of the shaded region.") is None


@pytest.mark.parametrize(
    "stem",
    [
        "Find the area of the shaded region if the side is 6.",
        "A regular polygon has 12 sides. Find the shaded area.",
    ],
)
def test_digits_in_a_stem_are_a_leak(stem):
    assert "digits" in stem_leaks_geometry(stem)


@pytest.mark.parametrize(
    "stem",
    [
        "Find the area between the hexagon and the triangle.",
        "The shaded region lies inside the octagon.",
        "Find the area of the shaded square.",
    ],
)
def test_polygon_names_are_a_leak(stem):
    """The side count is the implicit quantity. Naming the polygon states it in
    words, which is the same leak as printing the number."""
    assert "names shapes" in stem_leaks_geometry(stem)


@pytest.mark.parametrize("stem", ["Find the area of the two shaded regions.", "Six sides are shown."])
def test_number_words_are_a_leak(stem):
    assert "number words" in stem_leaks_geometry(stem)


def test_the_check_is_case_insensitive_and_ignores_punctuation():
    assert stem_leaks_geometry("Find the area of the HEXAGON.") is not None
    assert stem_leaks_geometry("Consider the triangle, then shade it.") is not None


def test_a_leaking_stem_is_rejected_on_a_category_one_record():
    with pytest.raises(SchemaValidationError, match="must not carry geometric information"):
        make(stem="Find the area between the hexagon and the triangle.").validate()


def test_the_leak_check_does_not_apply_to_category_two():
    """Category 2 conveys all geometric information through natural language. The
    check would reject every valid textual problem."""
    make(
        category=2,
        image_path=None,
        stem="A triangle has sides 3, 4, and 5. Find its area.",
    ).validate()


def test_the_error_message_names_the_offending_stem():
    with pytest.raises(SchemaValidationError, match="hexagon"):
        make(stem="Find the area of the hexagon.").validate()


# --- given records: the paper's own stems warn instead of rejecting ------------


def make_given(**overrides) -> Datapoint:
    base = dict(
        problem_id="paper_042",
        family="given",
        category=1,
        stem="Find the area of the shaded region.",
        image_path="images/paper_042.png",
        answer_exact="3*sqrt(3)/2",
        answer_decimal=2.598076211353316,
        origin="given",
        paper_index=42,
        answer_type="exact",
    )
    base.update(overrides)
    return Datapoint(**base)


@pytest.mark.parametrize(
    "stem",
    [
        "Find the area between the hexagon and the triangle.",
        "The square has side 4. Find the shaded area.",
        "Find the area of the two shaded regions.",
    ],
)
def test_a_leaking_given_stem_warns_instead_of_rejecting(stem):
    with pytest.warns(StemLeakWarning, match="must not carry geometric information"):
        make_given(stem=stem).validate()


def test_the_warning_names_the_record_and_the_stem():
    with pytest.warns(StemLeakWarning, match=r"paper_042: .*hexagon"):
        make_given(stem="Find the area of the hexagon.").validate()


def test_a_clean_given_stem_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_given().validate()


def test_a_given_category_one_record_still_needs_its_diagram():
    """Only the stem check softens; a Category 1 problem without a figure is still wrong."""
    with pytest.raises(SchemaValidationError, match="requires an image_path"):
        make_given(image_path=None).validate()


@pytest.mark.parametrize("origin", ["generated", "harvested"])
def test_other_origins_still_reject_a_leaking_stem(origin):
    with pytest.raises(SchemaValidationError, match="must not carry geometric information"):
        make(origin=origin, stem="Find the area of the hexagon.").validate()


# --- what the generator actually emits --------------------------------------


def test_every_generated_record_is_category_one_with_a_clean_stem(tmp_path):
    out = tmp_path / "run"
    generate("nested_polygons", 5, 0, out)
    for record in read_manifest(out):
        assert record.category == 1
        assert record.image_path
        assert stem_leaks_geometry(record.stem) is None


def test_the_version_was_bumped_for_the_format_change():
    """A manifest written by 0.1.0 has no category field, so 0.2.0 cannot claim to
    reproduce it. The version is the claim, and it had to move."""
    assert GENERATOR_VERSION != "0.1.0"
