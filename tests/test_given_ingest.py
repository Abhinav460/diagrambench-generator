"""Packaging the paper's own problems as origin='given' records."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from generator.emit import EmitError, read_manifest
from generator.given import GivenInputError, package_given

PNG = b"\x89PNG\r\n\x1a\nfake"

GOOD_1 = {
    "paper_index": 1, "category": 1, "stem": "Find the shaded area.",
    "answer_exact": "150*sqrt(3) - 75*pi", "answer_type": "exact",
    "source_image": "image1.png",
}
GOOD_2 = {
    "paper_index": 2, "category": 2,
    "stem": "Find the remainder when the sum is divided by seven.",
    "answer_exact": None, "answer_type": "unavailable",
}
BAD_ANSWER = {
    "paper_index": 3, "category": 1, "stem": "Find the shaded area.",
    "answer_exact": "foo(", "answer_type": "exact", "source_image": "image3.png",
}


def setup(tmp_path: Path, rows: list):
    images = tmp_path / "images_in"
    images.mkdir()
    for name in ("image1.png", "image3.png", "image4.png"):
        (images / name).write_bytes(PNG)
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows_path, images, tmp_path / "out"


def test_packages_good_rows_and_reports_bad_one(tmp_path):
    rows_path, images, out = setup(tmp_path, [GOOD_1, GOOD_2, BAD_ANSWER])
    report = package_given(rows_path, images, out)
    assert report.written == 2
    assert len(report.errors) == 1 and report.errors[0][0] == 3

    records = list(read_manifest(out))
    assert [r.paper_index for r in records] == [1, 2]
    first, second = records
    assert first.origin == "given" and first.image_path == "images/given_00001.png"
    assert first.answer_decimal == pytest.approx(150 * math.sqrt(3) - 75 * math.pi)
    assert (out / "images" / "given_00001.png").is_file()
    assert second.image_path is None and second.answer_exact is None
    assert second.answer_decimal is None and second.answer_type == "unavailable"


def test_duplicate_paper_index_is_an_error(tmp_path):
    dup = dict(GOOD_1, source_image="image4.png")
    rows_path, images, out = setup(tmp_path, [GOOD_1, dup])
    report = package_given(rows_path, images, out)
    assert report.written == 1
    assert report.errors[0][0] == 2 and "duplicate" in report.errors[0][1]


def test_leaky_stem_is_written_and_counted(tmp_path):
    leaky = dict(GOOD_1, stem="Find the area of the hexagon.")
    rows_path, images, out = setup(tmp_path, [leaky])
    report = package_given(rows_path, images, out)
    assert report.written == 1 and report.stem_leaks == 1 and not report.errors


def test_refuses_existing_manifest_and_empty_file(tmp_path):
    rows_path, images, out = setup(tmp_path, [GOOD_1])
    package_given(rows_path, images, out)
    with pytest.raises(EmitError):
        package_given(rows_path, images, out)
    package_given(rows_path, images, out, overwrite=True)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(GivenInputError):
        package_given(empty, images, tmp_path / "out2")


@pytest.mark.parametrize("filename", ["../image1.png", r"..\image1.png", r"C:\image1.png", "sub/image1.png", "image1:.png"])
def test_rejects_source_image_paths_instead_of_filenames(tmp_path, filename):
    rows_path, images, out = setup(tmp_path, [dict(GOOD_1, source_image=filename)])
    report = package_given(rows_path, images, out)
    assert report.written == 0
    assert report.errors[0][0] == 1 and "bare .png filename" in report.errors[0][1]
    assert list(read_manifest(out)) == []


def test_rejects_image_symlink_outside_images_dir(tmp_path):
    rows_path, images, out = setup(tmp_path, [dict(GOOD_1, source_image="link.png")])
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    try:
        (images / "link.png").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    report = package_given(rows_path, images, out)
    assert report.written == 0
    assert "inside images_dir" in report.errors[0][1]


def test_given_exact_answer_may_be_zero(tmp_path):
    row = dict(GOOD_2, answer_type="exact", answer_exact="0")
    rows_path, images, out = setup(tmp_path, [row])
    report = package_given(rows_path, images, out)
    assert report.written == 1 and not report.errors
    record, = read_manifest(out)
    assert record.answer_exact == "0" and record.answer_decimal == 0.0


def test_rejects_untrusted_answer_expression(tmp_path):
    row = dict(GOOD_2, answer_type="exact", answer_exact="__import__('os').getcwd()")
    rows_path, images, out = setup(tmp_path, [row])
    report = package_given(rows_path, images, out)
    assert report.written == 0
    assert report.errors[0][0] == 1 and "could not be parsed" in report.errors[0][1]
