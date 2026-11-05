"""Build and validate the paper's 200 image-only given problems.

Run from any working directory with --source-dir and --out-dir. The source is
read-only; the output is a self-contained manifest, images, and report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

# Direct execution puts scripts/, rather than the repository root, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.emit import read_manifest
from generator.given import package_given
from generator.schema import PAPER_PROBLEM_COUNT


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_png(path: Path) -> None:
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError(f"not a PNG: {path}")
        image.verify()


def expected_names(prefix: str) -> list[str]:
    return [f"{prefix}{index}.png" for index in range(1, PAPER_PROBLEM_COUNT + 1)]


def check_sources(source_dir: Path) -> None:
    if not source_dir.is_dir():
        raise ValueError(f"source directory does not exist: {source_dir}")
    expected = set(expected_names("image"))
    actual = {path.name for path in source_dir.iterdir() if path.suffix.lower() == ".png"}
    if actual != expected:
        raise ValueError(
            f"source PNG names differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    for name in expected_names("image"):
        path = source_dir / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != source_dir.resolve():
            raise ValueError(f"source image must be a regular file inside source directory: {path}")
        verify_png(path)


def check_output_target(source_dir: Path, out_dir: Path, overwrite: bool) -> None:
    source = source_dir.resolve()
    output = out_dir.resolve()
    if output == source or output in source.parents:
        raise ValueError("output directory cannot be the source directory or its parent")
    if out_dir.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {out_dir}")
    if not out_dir.exists():
        return
    if not out_dir.is_dir():
        raise ValueError(f"output path is not a directory: {out_dir}")
    contents = list(out_dir.iterdir())
    if contents and not overwrite:
        raise ValueError(f"output directory is not empty: {out_dir}; pass --overwrite to replace it")
    if overwrite:
        allowed = {"manifest.jsonl", "validation_report.md", "images"}
        unexpected = {path.name for path in contents} - allowed
        if unexpected:
            raise ValueError(f"output contains unrelated files: {sorted(unexpected)}")
        images = out_dir / "images"
        if images.exists():
            if images.is_symlink() or not images.is_dir():
                raise ValueError(f"output images path is not a regular directory: {images}")
            unexpected_images = {path.name for path in images.iterdir()} - {
                f"given_{index:05d}.png" for index in range(1, PAPER_PROBLEM_COUNT + 1)
            }
            if unexpected_images or any(path.is_symlink() for path in images.iterdir()):
                raise ValueError(f"output images directory contains unexpected files or symlinks: {images}")
        for name in ("manifest.jsonl", "validation_report.md"):
            path = out_dir / name
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError(f"output file is not a regular file: {path}")


def validate_dataset(source_dir: Path, out_dir: Path) -> str:
    """Check records, PNGs, and exact bytes; return a deterministic report."""
    check_sources(source_dir)
    manifest = out_dir / "manifest.jsonl"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if len(lines) != PAPER_PROBLEM_COUNT or any(not line.strip() for line in lines):
        raise ValueError(f"manifest must contain exactly {PAPER_PROBLEM_COUNT} nonempty records")
    records = list(read_manifest(out_dir))
    images_dir = out_dir / "images"
    expected_images = {f"given_{index:05d}.png" for index in range(1, PAPER_PROBLEM_COUNT + 1)}
    actual_images = {path.name for path in images_dir.iterdir()}
    if actual_images != expected_images:
        raise ValueError(
            f"output images differ: missing={sorted(expected_images - actual_images)}, "
            f"unexpected={sorted(actual_images - expected_images)}"
        )
    seen: set[str] = set()
    source_bytes = 0
    for index, record in enumerate(records, start=1):
        record.validate()
        problem_id = f"given_{index:05d}"
        source_name = f"image{index}.png"
        image_path = f"images/{problem_id}.png"
        if record.problem_id in seen:
            raise ValueError(f"duplicate problem ID: {record.problem_id}")
        seen.add(record.problem_id)
        if record.problem_id != problem_id or record.paper_index != index:
            raise ValueError(f"record {index} has incorrect ID, index, or order")
        if record.origin != "given" or record.family != "given":
            raise ValueError(f"record {index} has incorrect origin or family")
        if record.source_image != source_name or record.image_path != image_path:
            raise ValueError(f"record {index} has incorrect or unsafe image paths")
        if record.category is not None or record.stem is not None:
            raise ValueError(f"record {index} contains unverified metadata")
        if record.answer_type != "unavailable" or record.answer_exact is not None or record.answer_decimal is not None:
            raise ValueError(f"record {index} contains an unverified answer")
        source = source_dir / source_name
        target = out_dir / image_path
        if target.is_symlink() or not target.is_file() or target.resolve().parent != images_dir.resolve():
            raise ValueError(f"output image is not a regular file inside images directory: {target}")
        verify_png(target)
        if digest(source) != digest(target):
            raise ValueError(f"image bytes differ: {source_name}")
        source_bytes += source.stat().st_size
    if len(records) != PAPER_PROBLEM_COUNT or len(seen) != PAPER_PROBLEM_COUNT:
        raise ValueError(f"expected {PAPER_PROBLEM_COUNT} unique records, got {len(records)}")
    return "\n".join(
        [
            "# DiagramBench original image dataset validation",
            "",
            f"- Source PNGs: {PAPER_PROBLEM_COUNT}; image1.png through image200.png; all PNG-verified.",
            f"- Manifest records: {len(records)}; all load and pass schema validation.",
            f"- Output PNGs: {len(actual_images)}; given_00001.png through given_00200.png; all PNG-verified.",
            f"- Problem IDs: {len(seen)} unique, ordered by paper index.",
            f"- Image bytes: SHA-256 equals the corresponding source for all {PAPER_PROBLEM_COUNT} images.",
            "- Origin/family: given for every record.",
            "- Metadata: category=null, stem=null, answer_type=unavailable; answer values null.",
            f"- Source image bytes copied: {source_bytes}.",
            f"- Manifest SHA-256: `{digest(manifest)}`.",
            "",
            "Source filenames are recorded in `source_image`. The PNG is the authoritative problem statement.",
            "No categories, transcriptions, or answers were inferred. Source image redistribution rights are unverified.",
            "",
        ]
    )


def build(source_dir: Path, out_dir: Path, overwrite: bool = False) -> str:
    check_sources(source_dir)
    check_output_target(source_dir, out_dir, overwrite)
    with tempfile.TemporaryDirectory(prefix="diagrambench-given-") as temporary:
        rows = Path(temporary) / "rows.jsonl"
        rows.write_text(
            "".join(
                json.dumps({"paper_index": index, "source_image": f"image{index}.png"}, sort_keys=True) + "\n"
                for index in range(1, PAPER_PROBLEM_COUNT + 1)
            ),
            encoding="utf-8",
            newline="\n",
        )
        result = package_given(rows, source_dir, out_dir, overwrite=overwrite)
    if result.written != PAPER_PROBLEM_COUNT or result.errors:
        raise ValueError(result.report())
    report = validate_dataset(source_dir, out_dir)
    (out_dir / "validation_report.md").write_text(report, encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="directory with image1.png through image200.png")
    parser.add_argument("--out-dir", type=Path, required=True, help="dataset output directory")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset in the output directory")
    parser.add_argument("--validate-only", action="store_true", help="validate an existing dataset without writing files")
    args = parser.parse_args()
    try:
        report = (
            validate_dataset(args.source_dir, args.out_dir)
            if args.validate_only
            else build(args.source_dir, args.out_dir, overwrite=args.overwrite)
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"dataset validation failed: {exc}\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
