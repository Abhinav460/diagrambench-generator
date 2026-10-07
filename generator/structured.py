"""Structured problem input: real problems entered as a family's params.

A random draw invents a parameter set; a structured row names one. The row is
fed to the same ``solve`` -> ``verify`` -> ``render`` path a draw is (see
``cli.generate_from_file``), and its known answer is checked against the one the
family computes. That check is the point of the path: ``verify`` only confirms the
figure agrees with ``solve``, so it cannot notice ``solve`` answering a different
problem than the source posed. ``expected_answer`` can.

File format
-----------
JSONL, one problem per line; blank lines are skipped::

    {"input_id": "hexagon_ring_10", "family": "inscribed_circle",
     "params": {"k": 6, "side": 10}, "expected_answer": "150*sqrt(3) - 75*pi",
     "source": {"site": "...", "url": "...", "problem_id": "12",
                "original_answer": "150√3 − 75π"},
     "notes": "optional free text"}

- ``input_id`` (required): unique within the file, and used as the record's
  ``problem_id`` and image name, so limited to letters, digits, ``_``, ``.``, ``-``.
- ``family`` (required): a registered family that declares ``PARAM_TYPES``.
- ``params`` (required): exactly that family's keys, each of its declared kind
  (see ``generator.params``). No coercion: ``2.5`` for a side count is an error,
  not a 2.
- ``expected_answer`` (required): the source's answer as a sympy expression
  (``sqrt``, ``pi``, ``^`` or ``**``; ``ANSWER_NAMES`` is the full list of names).
  Must evaluate to a finite real number.
- ``source`` (optional): any of ``site``, ``url``, ``problem_id``,
  ``original_answer``, ``retrieved_at``, copied into the record's provenance fields.
- ``notes`` (optional): free text, not emitted.

``expected_answer`` is checked token by token before sympy sees it: only numbers,
the operators ``+ - * / ^ ** ( ) ,`` and the names in ``ANSWER_NAMES`` are allowed,
and the parse then runs with no builtins in scope. An input file cannot run code.
The parser lives in ``generator.answers``; it is re-exported here.

Each row is checked independently, so one bad row is reported rather than
aborting the file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from generator import registry
from generator.answers import ANSWER_NAMES, parse_expected_answer
from generator.params import parse_params

__all__ = [
    "ANSWER_NAMES",
    "RowError",
    "StructuredInputError",
    "StructuredProblem",
    "load_rows",
    "parse_expected_answer",
]

REQUIRED_KEYS = ("input_id", "family", "params", "expected_answer")
OPTIONAL_KEYS = ("source", "notes")

#: ``source`` key -> ``Datapoint`` field.
SOURCE_FIELDS = {
    "site": "source_site",
    "url": "source_url",
    "problem_id": "source_problem_id",
    "original_answer": "original_answer",
    "retrieved_at": "retrieved_at",
}

_INPUT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class StructuredInputError(ValueError):
    """The file as a whole cannot be read: missing, unreadable, or empty."""


@dataclass(frozen=True)
class StructuredProblem:
    """One row that passed loading. Its geometry has not been checked yet."""

    line: int
    input_id: str
    family: str
    params: dict[str, Any]
    expected_answer: str
    source: dict[str, str] = field(default_factory=dict)
    notes: Optional[str] = None

    def provenance(self) -> dict[str, str]:
        """The ``source`` block as ``Datapoint`` keyword arguments."""
        return {SOURCE_FIELDS[key]: value for key, value in self.source.items()}


@dataclass(frozen=True)
class RowError:
    """One row that could not be loaded, and why."""

    line: int
    input_id: Optional[str]
    family: Optional[str]
    reason: str


def _check_row(data: Any, seen_ids: set[str]) -> Union[tuple[str, str, dict, str, dict, Optional[str]], str]:
    """The row's fields, or the first reason it cannot be loaded."""
    if not isinstance(data, dict):
        return f"row must be a JSON object; got {type(data).__name__}"

    unknown = sorted(set(data) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    if unknown:
        return f"unknown keys {unknown}; allowed: {list(REQUIRED_KEYS + OPTIONAL_KEYS)}"
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        return f"missing required keys {missing}"

    input_id = data["input_id"]
    if not isinstance(input_id, str) or not _INPUT_ID.match(input_id):
        return (
            f"input_id must be 1-128 letters, digits, '_', '.' or '-', starting with a "
            f"letter or digit; got {input_id!r}"
        )
    if input_id in seen_ids:
        return f"duplicate input_id {input_id!r}"

    family_name = data["family"]
    if not isinstance(family_name, str):
        return f"family must be a string; got {family_name!r}"
    try:
        family = registry.get(family_name)
    except KeyError as exc:
        return str(exc.args[0])
    param_types = getattr(family, "PARAM_TYPES", None)
    if param_types is None or not hasattr(family, "validation_issues"):
        return f"family {family_name!r} does not support structured input"

    params = data["params"]
    if not isinstance(params, dict):
        return f"params must be a JSON object; got {type(params).__name__}"
    expected_keys, actual_keys = set(param_types), set(params)
    if expected_keys != actual_keys:
        return (
            f"params for {family_name!r} must have exactly the keys {sorted(expected_keys)}; "
            f"missing {sorted(expected_keys - actual_keys)}, "
            f"unexpected {sorted(actual_keys - expected_keys)}"
        )
    try:
        parse_params(param_types, params)
    except ValueError as exc:
        return f"params: {exc}"

    expected = data["expected_answer"]
    if not isinstance(expected, str) or not expected.strip():
        return f"expected_answer must be a non-empty string; got {expected!r}"
    try:
        parse_expected_answer(expected)
    except ValueError as exc:
        return str(exc)

    source = data.get("source", {})
    if not isinstance(source, dict):
        return f"source must be a JSON object; got {type(source).__name__}"
    bad_source = sorted(set(source) - set(SOURCE_FIELDS))
    if bad_source:
        return f"unknown source keys {bad_source}; allowed: {list(SOURCE_FIELDS)}"
    for key, value in source.items():
        if not isinstance(value, str):
            return f"source.{key} must be a string; got {value!r}"

    notes = data.get("notes")
    if notes is not None and not isinstance(notes, str):
        return f"notes must be a string; got {notes!r}"

    return input_id, family_name, params, expected, source, notes


def load_rows(path: Path | str) -> list[Union[StructuredProblem, RowError]]:
    """Read a structured-input file into loaded problems and per-row errors, in order."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StructuredInputError(f"cannot read structured input {path}: {exc}") from exc

    rows: list[Union[StructuredProblem, RowError]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            rows.append(RowError(line_number, None, None, f"invalid JSON: {exc}"))
            continue

        checked = _check_row(data, seen_ids)
        if isinstance(checked, str):
            input_id = data.get("input_id") if isinstance(data, dict) else None
            family = data.get("family") if isinstance(data, dict) else None
            rows.append(
                RowError(
                    line_number,
                    input_id if isinstance(input_id, str) else None,
                    family if isinstance(family, str) else None,
                    checked,
                )
            )
            continue

        input_id, family_name, params, expected, source, notes = checked
        seen_ids.add(input_id)
        rows.append(
            StructuredProblem(line_number, input_id, family_name, params, expected, source, notes)
        )

    if not rows:
        raise StructuredInputError(f"structured input {path} contains no rows")
    return rows
