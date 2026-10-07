"""Stage 1 acceptance tests: the record round-trips, and validate() rejects each
invariant violation individually."""

from __future__ import annotations

import json
import math

import pytest

from generator.schema import (
    PAPER_PROBLEM_COUNT,
    Datapoint,
    SchemaValidationError,
    canonical_params,
)


def make(**overrides) -> Datapoint:
    """A valid record, with fields overridable so each test perturbs exactly one."""
    base = dict(
        problem_id="gen-nested-0001",
        family="nested_polygons",
        category=1,
        seed=0,
        params={"n": 6, "m": 3, "side": 1},
        stem="Find the area of the shaded region.",
        image_path="images/gen-nested-0001.png",
        answer_exact="3*sqrt(3)/2",
        answer_decimal=2.598076211353316,
        signature="a3f1c0deadbeef",
        generator_version="0.2.0",
        origin="generated",
    )
    base.update(overrides)
    return Datapoint(**base)


# --- round-trip -----------------------------------------------------------


def test_round_trips_to_json_and_back_unchanged():
    dp = make()
    assert Datapoint.from_json(dp.to_json()) == dp


def test_round_trip_survives_harvest_provenance_fields():
    dp = make(
        origin="harvested",
        source_site="andymath",
        source_url="https://andymath.com/home/geometry/geometry-challenges/",
        source_problem_id="27",
        retrieved_at="2026-09-02T16:00:00Z",
        answer_source="inline_html",
        original_answer=r"The area is \( 8+4\sqrt{3}\) cm\(^2 \)",
    )
    dp.validate()
    assert Datapoint.from_json(dp.to_json()) == dp


def test_params_serialize_as_a_json_object_not_a_string():
    """The manifest must be readable with one parse, not two."""
    payload = json.loads(make().to_json())
    assert payload["params"] == {"m": 3, "n": 6, "side": 1}


def test_params_are_hashable_and_order_independent():
    a = make(params={"n": 6, "m": 3, "side": 1})
    b = make(params={"side": 1, "m": 3, "n": 6})
    assert a.params == b.params
    assert hash(a.params) == hash(b.params)


def test_params_dict_gives_back_a_plain_dict():
    assert make().params_dict == {"m": 3, "n": 6, "side": 1}


def test_nested_sequence_params_round_trip():
    dp = make(params={"vertices": [[0, 0], [1, 0], [0, 1]]})
    dp.validate()
    assert Datapoint.from_json(dp.to_json()) == dp
    assert dp.params_dict["vertices"] == [[0, 0], [1, 0], [0, 1]]


def test_a_valid_record_validates():
    make().validate()


# --- rejection cases, one per test ----------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite_answer(bad):
    with pytest.raises(SchemaValidationError, match="finite"):
        make(answer_decimal=bad).validate()


def test_rejects_zero_answer():
    with pytest.raises(SchemaValidationError, match="degenerate"):
        make(answer_decimal=0.0).validate()


@pytest.mark.parametrize("bad", ["", "   "])
def test_rejects_empty_stem(bad):
    with pytest.raises(SchemaValidationError, match="stem"):
        make(stem=bad).validate()


def test_rejects_params_that_fail_to_serialize():
    """A complex value is hashable, so it reaches the serialization check rather
    than being caught earlier by the hashability check."""
    with pytest.raises(SchemaValidationError, match="not JSON-serializable"):
        make(params={"weird": complex(1, 2)}).validate()


def test_rejects_unhashable_param_values():
    with pytest.raises(SchemaValidationError, match="not hashable"):
        make(params={"weird": {1, 2}}).validate()


def test_rejects_nan_inside_params():
    """allow_nan=False: NaN is not valid JSON and must not reach the manifest."""
    with pytest.raises(SchemaValidationError, match="not JSON-serializable"):
        make(params={"ratio": math.nan}).validate()


def test_rejects_bad_origin():
    with pytest.raises(SchemaValidationError, match="origin"):
        make(origin="synthesized").validate()


def test_rejects_empty_image_path():
    with pytest.raises(SchemaValidationError, match="image_path"):
        make(image_path="").validate()


def test_rejects_non_integer_seed():
    with pytest.raises(SchemaValidationError, match="seed"):
        make(seed="0").validate()


def test_rejects_boolean_seed():
    """bool subclasses int; a True seed is a bug that would otherwise pass."""
    with pytest.raises(SchemaValidationError, match="seed"):
        make(seed=True).validate()


def test_rejects_non_string_param_keys():
    """Int keys would come back from JSON as strings and break the round-trip."""
    with pytest.raises(SchemaValidationError, match="param keys must be str"):
        make(params={1: "x"})


# --- construction and reload guards ---------------------------------------


def test_record_is_frozen():
    with pytest.raises(Exception):
        make().stem = "something else"  # type: ignore[misc]


def test_from_dict_rejects_unknown_fields():
    payload = make().to_dict()
    payload["extra_field"] = 1
    with pytest.raises(SchemaValidationError, match="unknown fields"):
        Datapoint.from_dict(payload)


def test_from_dict_rejects_missing_required_fields():
    payload = make().to_dict()
    del payload["answer_exact"]
    with pytest.raises(SchemaValidationError, match="missing required fields"):
        Datapoint.from_dict(payload)


def test_canonical_params_is_idempotent():
    once = canonical_params({"n": 6, "m": 3})
    assert canonical_params(once) == once


def test_canonical_params_rejects_duplicate_keys():
    with pytest.raises(SchemaValidationError, match="duplicate param keys"):
        canonical_params((("n", 1), ("n", 2)))


# --- origin="given": one of the paper's problems, packaged as published ----------


def make_given(**overrides) -> Datapoint:
    """A valid given record: no generator provenance, identified by paper_index."""
    base = dict(
        problem_id="paper_017",
        family="given",
        category=1,
        stem="Find the area of the shaded region.",
        image_path="images/paper_017.png",
        answer_exact="3*sqrt(3)/2",
        answer_decimal=2.598076211353316,
        origin="given",
        paper_index=17,
        answer_type="exact",
    )
    base.update(overrides)
    return Datapoint(**base)


def test_a_given_record_validates_and_round_trips():
    dp = make_given(source_image="IMG_0417.png")
    dp.validate()
    assert Datapoint.from_json(dp.to_json()) == dp


def test_a_given_record_serializes_its_fields_and_other_origins_do_not():
    assert {"paper_index", "answer_type"} <= set(make_given().to_dict())
    assert "source_image" not in make_given().to_dict()
    assert not {"paper_index", "source_image", "answer_type"} & set(make().to_dict())


@pytest.mark.parametrize(
    "field,value",
    [("seed", 0), ("params", {"n": 6}), ("signature", "abc"), ("generator_version", "0.2.0")],
)
def test_a_given_record_rejects_generator_provenance(field, value):
    with pytest.raises(SchemaValidationError, match="origin='given' has no generator provenance"):
        make_given(**{field: value}).validate()


@pytest.mark.parametrize("bad", [None, 0, PAPER_PROBLEM_COUNT + 1, -1, True, 17.0, "17"])
def test_a_given_record_requires_paper_index_in_range(bad):
    with pytest.raises(SchemaValidationError, match="paper_index"):
        make_given(paper_index=bad).validate()


@pytest.mark.parametrize("index", [1, PAPER_PROBLEM_COUNT])
def test_paper_index_bounds_are_inclusive(index):
    make_given(paper_index=index).validate()


@pytest.mark.parametrize("bad", [None, "proof", ""])
def test_a_given_record_requires_a_known_answer_type(bad):
    with pytest.raises(SchemaValidationError, match="answer_type"):
        make_given(answer_type=bad).validate()


@pytest.mark.parametrize("bad", ["", "   ", 7])
def test_source_image_must_be_a_non_empty_string_when_set(bad):
    with pytest.raises(SchemaValidationError, match="source_image"):
        make_given(source_image=bad).validate()


def test_an_unavailable_answer_leaves_both_answer_fields_empty():
    dp = make_given(answer_type="unavailable", answer_exact=None, answer_decimal=None)
    dp.validate()
    assert Datapoint.from_json(dp.to_json()) == dp


@pytest.mark.parametrize(
    "overrides",
    [{"answer_exact": "3", "answer_decimal": None}, {"answer_exact": None, "answer_decimal": 3.0}],
)
def test_an_unavailable_answer_rejects_a_stray_value(overrides):
    with pytest.raises(SchemaValidationError, match="answer_type='unavailable' means there is no answer"):
        make_given(answer_type="unavailable", **overrides).validate()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"answer_exact": None}, "answer_exact must be a non-empty string"),
        ({"answer_decimal": None}, "answer_decimal must be a real number"),
        ({"answer_decimal": 0}, "answer_decimal is zero"),
    ],
)
def test_an_exact_given_answer_is_checked_like_any_other(overrides, match):
    with pytest.raises(SchemaValidationError, match=match):
        make_given(**overrides).validate()


@pytest.mark.parametrize("origin", ["generated", "harvested"])
def test_only_given_records_may_omit_the_answer(origin):
    with pytest.raises(SchemaValidationError, match="answer_exact must be a non-empty string"):
        make(origin=origin, answer_exact=None, answer_decimal=None, answer_type="unavailable").validate()


def test_answer_fields_stay_required_keys_even_when_empty():
    """Nullable is not omittable: a manifest line without answer_exact is malformed."""
    payload = make_given(answer_type="unavailable", answer_exact=None, answer_decimal=None).to_dict()
    del payload["answer_exact"]
    with pytest.raises(SchemaValidationError, match="missing required fields"):
        Datapoint.from_dict(payload)
