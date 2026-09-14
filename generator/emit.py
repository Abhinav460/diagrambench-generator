"""Write a run to disk: one image per problem, one manifest line per problem.

The output directory is the deliverable. Everything else in this package exists to
produce it, so the invariants here are about what a *reader* of that directory can
rely on:

- ``manifest.jsonl`` is one JSON object per line, and every complete line is a
  valid record even if the run that wrote it died.
- ``images/`` holds one PNG per manifest line, named by ``problem_id``, so a
  record and its figure can be paired without consulting anything else.
- Two runs with the same seed and ``generator_version`` produce byte-identical
  manifests.

Why records stream rather than accumulate
-----------------------------------------
The spec's driver sketch collects records in a list and writes them at the end.
That is simpler, and it is why the format is JSONL -- but a list written at the
end makes the format's main advantage unreachable: a run killed at problem 900 of
1000 would leave no manifest at all, having held every record in memory. So
``ManifestWriter`` appends and flushes each line as it is produced. Generation is
slow (sympy, then matplotlib) and a long run is exactly the kind that gets
interrupted.

Flushing every line rather than relying on buffering is a deliberate cost. It is
a few microseconds against tens of milliseconds of rendering, and it is what makes
the partial-file guarantee true rather than probable.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Iterator, Optional

from generator.schema import Datapoint, SchemaValidationError

__all__ = [
    "IMAGES_DIRNAME",
    "MANIFEST_NAME",
    "EmitError",
    "ManifestWriter",
    "image_path_for",
    "problem_id_for",
    "read_manifest",
    "write_manifest",
]

#: Layout of an output directory. Constants because a reader -- a notebook, a
#: collaborator's loader -- needs to hardcode them somewhere, and it should be here.
MANIFEST_NAME = "manifest.jsonl"
IMAGES_DIRNAME = "images"


class EmitError(OSError):
    """Raised when a run cannot be written, or would overwrite an existing one."""


def problem_id_for(family: str, index: int, *, width: int = 5) -> str:
    """A stable, sortable id: family name plus a zero-padded index within the run.

    Zero-padded so that lexical order matches generation order in a file listing;
    ``image10`` sorting before ``image2`` is a small thing that becomes annoying
    across a few hundred files. Derived from the index rather than the signature
    because an id is for humans referring to a problem, while the signature is for
    machines comparing two.
    """
    if index < 0:
        raise EmitError(f"problem index must be non-negative; got {index}")
    return f"{family}_{index:0{width}d}"


def image_path_for(out_dir: Path, problem_id: str) -> Path:
    """Absolute path of a problem's PNG within a run directory."""
    return Path(out_dir) / IMAGES_DIRNAME / f"{problem_id}.png"


def relative_image_path(problem_id: str) -> str:
    """What goes in the record's ``image_path`` field.

    Relative to the run directory, and with a forward slash regardless of platform,
    so a manifest stays valid when the directory is moved, zipped, or uploaded --
    which for a dataset published alongside a paper is the normal case, not the
    exception.
    """
    return f"{IMAGES_DIRNAME}/{problem_id}.png"


class ManifestWriter:
    """Appends validated records to ``manifest.jsonl``, one flushed line each.

    A context manager because the file handle must be closed even when the driver
    raises -- a ``VerificationError`` mid-run should still leave the lines already
    written intact and readable.
    """

    def __init__(self, out_dir: Path | str, *, overwrite: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / MANIFEST_NAME
        self.overwrite = overwrite
        self.count = 0
        self._handle: Optional[Any] = None

    def __enter__(self) -> "ManifestWriter":
        if self.path.exists() and not self.overwrite:
            raise EmitError(
                f"{self.path} already exists; pass overwrite=True (or --overwrite) to "
                f"replace it. Refusing silently to append would mix two runs into one "
                f"manifest, and refusing to overwrite protects a completed run."
            )
        (self.out_dir / IMAGES_DIRNAME).mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "w", encoding="utf-8", newline="\n")
        return self

    def write(self, record: Datapoint) -> None:
        """Validate, serialize, append, flush.

        Validation happens here rather than at the call site so that no unvalidated
        record can reach the manifest by way of a driver that forgot to check.
        """
        if self._handle is None:
            raise EmitError("ManifestWriter used outside its context manager")
        record.validate()
        self._handle.write(record.to_json() + "\n")
        self._handle.flush()
        self.count += 1

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def write_manifest(
    records: Iterable[Datapoint], out_dir: Path | str, *, overwrite: bool = False
) -> Path:
    """Write a complete set of records at once. The batch form of ``ManifestWriter``.

    Kept for callers that already hold every record -- a test, or a rewrite of an
    existing manifest -- but the driver uses the streaming form.
    """
    with ManifestWriter(out_dir, overwrite=overwrite) as writer:
        for record in records:
            writer.write(record)
        return writer.path


def read_manifest(out_dir: Path | str, *, strict: bool = True) -> Iterator[Datapoint]:
    """Read records back, skipping blank lines.

    ``strict=False`` tolerates a truncated final line, which is what a run killed
    mid-write leaves behind. That tolerance is opt-in: silently ignoring a corrupt
    line by default would let a damaged manifest look complete.
    """
    path = Path(out_dir) / MANIFEST_NAME
    if not path.exists():
        raise EmitError(f"no manifest at {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            yield Datapoint.from_json(line)
        except (json.JSONDecodeError, SchemaValidationError):
            is_last = number == len(lines)
            if strict or not is_last:
                raise EmitError(
                    f"{path}:{number} is not a valid record. If this is the last line "
                    f"of an interrupted run, read with strict=False to skip it."
                )
            return
