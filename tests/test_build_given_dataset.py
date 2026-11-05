"""End-to-end checks for the reusable original-image dataset command."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from generator.emit import read_manifest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_given_dataset.py"


@pytest.fixture
def source_dir(tmp_path):
    source = tmp_path / "original PNGs"
    source.mkdir()
    for index in range(1, 201):
        Image.new("RGB", (1, 1), (index, index % 127, index % 67)).save(source / f"image{index}.png")
    return source


def command(source: Path, output: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--source-dir", str(source), "--out-dir", str(output), *flags],
        capture_output=True,
        text=True,
        cwd=source.parent,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_valid_self_contained_and_deterministic_dataset(source_dir, tmp_path):
    first, second = tmp_path / "first output", tmp_path / "second output"
    for output in (first, second):
        result = command(source_dir, output)
        assert result.returncode == 0, result.stderr
        assert (output / "validation_report.md").read_text(encoding="utf-8") == result.stdout
        records = list(read_manifest(output))
        assert len(records) == 200
        assert [record.problem_id for record in records] == [f"given_{index:05d}" for index in range(1, 201)]
        assert {path.name for path in (output / "images").iterdir()} == {
            f"given_{index:05d}.png" for index in range(1, 201)
        }
        for index, record in enumerate(records, start=1):
            assert record.origin == "given" and record.family == "given"
            assert record.category is None and record.stem is None
            assert record.answer_type == "unavailable"
            assert record.answer_exact is None and record.answer_decimal is None
            assert record.source_image == f"image{index}.png"
            assert record.image_path == f"images/given_{index:05d}.png"
            record.validate()
            assert sha256(source_dir / record.source_image) == sha256(output / record.image_path)
        assert command(source_dir, output, "--validate-only").returncode == 0
    assert (first / "manifest.jsonl").read_bytes() == (second / "manifest.jsonl").read_bytes()
    assert (first / "validation_report.md").read_bytes() == (second / "validation_report.md").read_bytes()


@pytest.mark.parametrize("bad_name", ["image201.png", "other.png", "IMAGE1.PNG"])
def test_rejects_unexpected_png_names(source_dir, tmp_path, bad_name):
    (source_dir / bad_name).write_bytes((source_dir / "image1.png").read_bytes())
    output = tmp_path / "output"
    result = command(source_dir, output)
    assert result.returncode != 0 and "unexpected" in result.stderr
    assert not output.exists()


def test_rejects_missing_source_image(source_dir, tmp_path):
    (source_dir / "image100.png").unlink()
    output = tmp_path / "output"
    result = command(source_dir, output)
    assert result.returncode != 0 and "image100.png" in result.stderr
    assert not output.exists()


def test_rejects_invalid_png_contents(source_dir, tmp_path):
    (source_dir / "image100.png").write_bytes(b"not actually a PNG")
    output = tmp_path / "output"
    result = command(source_dir, output)
    assert result.returncode != 0 and "dataset validation failed" in result.stderr
    assert not output.exists()


def test_refuses_overwrite_and_allows_explicit_rebuild(source_dir, tmp_path):
    output = tmp_path / "output"
    assert command(source_dir, output).returncode == 0
    manifest_before = (output / "manifest.jsonl").read_bytes()
    image_before = (output / "images" / "given_00001.png").read_bytes()
    result = command(source_dir, output)
    assert result.returncode != 0 and "--overwrite" in result.stderr
    assert (output / "manifest.jsonl").read_bytes() == manifest_before
    assert (output / "images" / "given_00001.png").read_bytes() == image_before
    assert command(source_dir, output, "--overwrite").returncode == 0
    assert (output / "manifest.jsonl").read_bytes() == manifest_before


def test_validation_rejects_duplicate_ids_and_changed_image_bytes(source_dir, tmp_path):
    output = tmp_path / "output"
    assert command(source_dir, output).returncode == 0
    manifest = output / "manifest.jsonl"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    duplicate = json.loads(lines[1])
    duplicate["problem_id"] = "given_00001"
    lines[1] = json.dumps(duplicate)
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = command(source_dir, output, "--validate-only")
    assert result.returncode != 0 and "duplicate problem ID" in result.stderr
    assert command(source_dir, output, "--overwrite").returncode == 0
    image = output / "images" / "given_00001.png"
    Image.new("RGB", (1, 1), (0, 0, 0)).save(image)
    result = command(source_dir, output, "--validate-only")
    assert result.returncode != 0 and "image bytes differ" in result.stderr


def test_refuses_unsafe_output_directory(source_dir, tmp_path):
    result = command(source_dir, source_dir)
    assert result.returncode != 0 and "source directory" in result.stderr
    extra = tmp_path / "output"
    extra.mkdir()
    (extra / "unrelated.txt").write_text("keep me", encoding="utf-8")
    result = command(source_dir, extra, "--overwrite")
    assert result.returncode != 0 and "unrelated files" in result.stderr
    assert (extra / "unrelated.txt").read_text(encoding="utf-8") == "keep me"
