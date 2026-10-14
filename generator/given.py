"""Package the paper's own problems as origin="given" records.

Input is a JSONL file, one row per paper problem. A row can contain only the
source image and its verified index:

    {"paper_index": 1, "source_image": "image1.png"}

- ``paper_index`` (required): int, 1..PAPER_PROBLEM_COUNT. Repeated indices
  among accepted rows are reported and not emitted twice.
- ``category``: 1 (diagram-dependent), 2 (textual), or null/absent (unknown).
- ``stem``: optional verified transcription; null/absent means the original image
  remains the sole problem statement.
- ``answer_type``: "exact" or "unavailable" (the default).
- ``answer_exact``: sympy-parseable string; required when answer_type is "exact",
  must be null/absent when "unavailable".
- ``source_image``: bare .png filename inside ``images_dir``. A supplied image is
  copied for every category; a legacy category 2 row may omit it if it has a stem.

Each row is checked independently: a bad row is reported with its line number and
does not stop the run. Images are copied through emit's own naming helpers, and
records are written with ManifestWriter, which validates them again.
"""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from generator.answers import parse_expected_answer
from generator.emit import (
    IMAGES_DIRNAME,
    EmitError,
    ManifestWriter,
    image_path_for,
    problem_id_for,
    relative_image_path,
)
from generator.schema import Datapoint, StemLeakWarning

__all__ = ["GivenInputError", "GivenReport", "package_given"]

FAMILY = "given"
REQUIRED_KEYS = ("paper_index",)
OPTIONAL_KEYS = ("category", "stem", "answer_type", "answer_exact", "source_image")


class GivenInputError(ValueError):
    """The rows file as a whole cannot be read: missing, unreadable, or empty."""


@dataclass
class GivenReport:
    written: int = 0
    stem_leaks: int = 0
    errors: list = field(default_factory=list)  # (line_number, reason)

    def report(self) -> str:
        lines = [
            f"written: {self.written}",
            f"errors: {len(self.errors)}",
            f"category 1 stems flagged as leaking geometry (written anyway): {self.stem_leaks}",
        ]
        for lineno, reason in self.errors:
            lines.append(f"  line {lineno}: {reason}")
        return "\n".join(lines)


def _record_from_row(row: Any) -> Datapoint:
    """Build a Datapoint from one parsed row, or raise ValueError saying why not."""
    if not isinstance(row, dict):
        raise ValueError(f"row must be a JSON object; got {type(row).__name__}")
    unknown = sorted(set(row) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    if unknown:
        raise ValueError(f"unknown keys: {unknown}")
    missing = [key for key in REQUIRED_KEYS if key not in row]
    if missing:
        raise ValueError(f"missing keys: {missing}")

    index = row["paper_index"]
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"paper_index must be an int; got {index!r}")
    problem_id = problem_id_for(FAMILY, index)

    category = row.get("category")
    source_image = row.get("source_image")
    image_path: Optional[str] = None
    if category == 1:
        if not isinstance(source_image, str) or not source_image.strip():
            raise ValueError("category 1 requires source_image (a .png filename)")
    if source_image is not None:
        if not isinstance(source_image, str) or not source_image.strip():
            raise ValueError(f"source_image must be a bare .png filename; got {source_image!r}")
        if (
            Path(source_image).name != source_image
            or any(char in source_image for char in ("/", "\\", ":"))
            or not source_image.lower().endswith(".png")
        ):
            raise ValueError(f"source_image must be a bare .png filename; got {source_image!r}")
        image_path = relative_image_path(problem_id)

    answer_type = row.get("answer_type", "unavailable")
    answer_exact = row.get("answer_exact")
    answer_decimal: Optional[float] = None
    if answer_type == "exact":
        if not isinstance(answer_exact, str) or not answer_exact.strip():
            raise ValueError("answer_type 'exact' requires a non-empty answer_exact string")
        try:
            answer_decimal = float(parse_expected_answer(answer_exact).evalf())
        except Exception as exc:  # parse errors come in several exception types
            raise ValueError(f"answer_exact {answer_exact!r} could not be parsed: {exc}") from None

    return Datapoint(
        problem_id=problem_id,
        family=FAMILY,
        category=category,
        stem=row.get("stem"),
        image_path=image_path,
        answer_exact=answer_exact,
        answer_decimal=answer_decimal,
        origin="given",
        paper_index=index,
        source_image=source_image,
        answer_type=answer_type,
    )


def package_given(
    rows_path: Union[str, Path],
    images_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    overwrite: bool = False,
) -> GivenReport:
    """Emit one ``given`` record per row of ``rows_path`` into ``out_dir``."""
    rows_path, images_dir, out_dir = Path(rows_path), Path(images_dir), Path(out_dir)
    try:
        text = rows_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise GivenInputError(f"cannot read {rows_path}: {exc}") from exc
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        raise GivenInputError(f"{rows_path} has no rows")

    images_out = out_dir / IMAGES_DIRNAME
    if not overwrite and images_out.is_dir() and any(images_out.iterdir()):
        raise EmitError(f"{images_out} already contains files; pass overwrite=True to replace them")

    report = GivenReport()
    seen: set = set()
    with ManifestWriter(out_dir, overwrite=overwrite) as writer:
        for lineno, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            try:
                record = _record_from_row(json.loads(raw))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    record.validate()
                leaked = any(issubclass(w.category, StemLeakWarning) for w in caught)
                if record.paper_index in seen:
                    raise ValueError(f"duplicate paper_index {record.paper_index}")
                source = None
                if record.image_path is not None:
                    source = images_dir / record.source_image
                    if not source.is_file():
                        raise ValueError(f"image not found: {source}")
                    if source.resolve().parent != images_dir.resolve():
                        raise ValueError(f"image must remain inside images_dir: {source}")
            except (ValueError, TypeError) as exc:
                report.errors.append((lineno, str(exc)))
                continue

            if source is not None:
                destination = image_path_for(out_dir, record.problem_id)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                writer.write(record)
            seen.add(record.paper_index)
            report.written += 1
            report.stem_leaks += int(leaked)
    return report
