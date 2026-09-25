"""Command-line entry point: draw, validate, solve, render, verify, dedupe, emit.

Run with::

    python -m generator.cli --family nested_polygons --n 200 --seed 0 --out runs/dev
    python -m generator.cli --from-file problems.jsonl --out runs/structured

The second form skips the random draw: each row of the file names a real problem as
a family's params, and goes through the same solve/verify/render/emit step a draw
does (``_emit_problem``), plus a check against the row's known answer. See
``generator.structured`` for the file format. A row with ``variants`` is seeded instead:
its params are pinned and the rest drawn from ``--seed``, through the same loop as a
random run (``_draw_into``).

The loop is the spec's driver, with one structural change: every rejection is
counted by cause. A run that produces 40 problems from 1000 draws has failed at
something, and the difference between "the sampler draws invalid configurations"
and "the sampler redraws the same twenty problems" is the difference between
loosening a constraint and rewriting ``sample``. Without per-cause counts, the two
look identical from outside.

Reproducibility
---------------
``--seed`` plus ``generator_version`` is meant to be sufficient to reproduce a run
byte for byte. That holds because the only randomness is the seeded ``Generator``,
records are written in generation order, and ``render`` already strips the
matplotlib version from PNG metadata.

Note what this does *not* survive: changing a family's sampling code without
bumping ``GENERATOR_VERSION``. The version is the claim being made, and nothing
here can verify it. That is a discipline, not a guarantee.

Exit codes
----------
``0`` the run reached ``--n``; ``1`` a genuine failure (verification, or an
unwritable output directory); ``2`` the run exhausted ``--max-draws`` short of
``--n``. A short run is separated from a failed one because it is a normal outcome
for a family whose parameter space is smaller than the request, and a caller
scripting a sweep needs to tell the two apart.

With ``--from-file``: ``0`` every row was emitted; ``2`` the only shortfall is
seeded rows that stopped short of their ``variants`` (a pinned space smaller than
the request, as with ``--n``); ``1`` anything else -- a row that failed to load, was
geometrically invalid, disagreed with its expected answer, or duplicated an earlier
row, as well as the failures above.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

from generator import GENERATOR_VERSION, registry
from generator.dedupe import DEFAULT_PRECISION, Deduplicator
from generator.emit import (
    EmitError,
    ManifestWriter,
    image_path_for,
    problem_id_for,
    relative_image_path,
)
from generator.render import render
from generator.schema import Datapoint
from generator.structured import (
    RowError,
    StructuredInputError,
    StructuredProblem,
    load_rows,
    parse_expected_answer,
)
from generator.verify import TOLERANCE, VerificationError, relative_difference, verify

__all__ = [
    "AnswerMismatch",
    "RowOutcome",
    "RunStats",
    "generate",
    "generate_from_file",
    "generate_mixed",
    "main",
]

#: Multiple of ``--n`` used as the default draw ceiling. Without a ceiling, a
#: family whose space is smaller than the request spins forever; 200x is loose
#: enough that a healthy family never reaches it.
DEFAULT_DRAW_MULTIPLIER = 200

#: For a family whose ceiling is unknown (no ``parameter_space``): stop once this many
#: consecutive draws have produced no new problem, rather than burning draws silently
#: until ``max_draws``. Families with a known ceiling are stopped by the ceiling
#: instead; a streak guard would misfire on them, since the last few problems of a
#: near-exhausted space routinely take more than this many draws to turn up.
DUPLICATE_STREAK_LIMIT = 5000


class AnswerMismatch(ValueError):
    """The family's answer disagrees with a structured row's known answer.

    Not a ``VerificationError``: the figure and the answer agree with each other,
    so the pipeline is sound. What is wrong is the row -- its params, its expected
    answer, or the choice of family -- and the run continues past it.
    """


@dataclass
class RowOutcome:
    """What happened to one row of a structured-input file."""

    line: int
    input_id: Optional[str]
    family: Optional[str]
    #: ``emitted``, ``input_error``, ``geometry_rejected``, ``answer_mismatch``,
    #: ``duplicate``, or ``error``; for a seeded row, ``emitted`` or ``short``.
    status: str
    detail: str = ""
    warnings: list[str] = field(default_factory=list)
    answer: Optional[str] = None
    #: A seeded row's own draw loop: what it asked for and what it got.
    variants: Optional[RunStats] = None

    @property
    def passed(self) -> bool:
        return self.status == "emitted"


@dataclass
class RunStats:
    """What a run did, in the terms needed to diagnose a disappointing one.

    For a structured-input run, ``draws`` counts rows, ``invalid`` counts rows that
    failed to load or were geometrically invalid, and ``rows`` holds each row's
    outcome. ``requested`` counts problems: one per ordinary row, ``variants`` per
    seeded row, whose draws and rejections are kept on its own ``RowOutcome``.
    """

    family: str
    seed: Optional[int]
    requested: int
    draws: int = 0
    invalid: int = 0
    duplicates: int = 0
    emitted: int = 0
    worst_relative_difference: float = 0.0
    rejection_reasons: Counter = field(default_factory=Counter)
    per_family: Counter = field(default_factory=Counter)
    rows: list[RowOutcome] = field(default_factory=list)
    readability_warnings: Counter = field(default_factory=Counter)
    structured: bool = False
    #: Each drawn family's number of distinct valid problems, or ``None`` if unknown.
    ceilings: dict[str, Optional[int]] = field(default_factory=dict)
    #: Why the loop stopped before ``requested``, when it stopped for a known reason.
    stopped_early: Optional[str] = None

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draws that passed ``is_valid``. Duplicates count as accepted
        here: they were valid configurations, just ones already seen."""
        return (self.draws - self.invalid) / self.draws if self.draws else 0.0

    @property
    def collision_rate(self) -> float:
        valid = self.draws - self.invalid
        return self.duplicates / valid if valid else 0.0

    @property
    def complete(self) -> bool:
        return self.emitted >= self.requested

    def report(self) -> str:
        """A human-readable summary, printed at the end of every run."""
        if self.structured:
            return self._structured_report()
        lines = [
            f"family                : {self.family}",
            f"seed                  : {self.seed}",
            f"emitted               : {self.emitted}/{self.requested}"
            + ("" if self.complete else "   *** SHORT ***"),
            f"draws                 : {self.draws}",
        ]
        if self.ceilings:
            lines.append(
                "ceiling               : "
                + ", ".join(
                    f"{name} {'unknown' if limit is None else limit}"
                    for name, limit in sorted(self.ceilings.items())
                )
            )
        if self.stopped_early:
            lines.append(f"stopped early         : {self.stopped_early}")
        lines += [
            f"acceptance rate       : {self.acceptance_rate:.1%} "
            f"({self.draws - self.invalid} valid, {self.invalid} rejected)",
            f"collision rate        : {self.collision_rate:.1%} "
            f"({self.duplicates} duplicates of {self.draws - self.invalid} valid)",
            f"verified              : {self.emitted} "
            f"(worst relative difference {self.worst_relative_difference:.2e})",
        ]
        if len(self.per_family) > 1:
            lines.append("emitted by family  :")
            for name, count in sorted(self.per_family.items()):
                lines.append(f"    {count:6d}  {name}")
        if self.rejection_reasons:
            lines.append("rejections by cause   :")
            for reason, count in self.rejection_reasons.most_common():
                lines.append(f"    {count:6d}  {reason}")
        return "\n".join(lines)


    def _structured_report(self) -> str:
        statuses = Counter(row.status for row in self.rows)
        lines = [
            f"input                 : {self.family}",
            f"emitted               : {self.emitted}/{self.requested}"
            + ("" if self.complete else "   *** INCOMPLETE ***"),
            f"input errors          : {statuses['input_error']}",
            f"geometry rejected     : {statuses['geometry_rejected']}",
            f"answer mismatches     : {statuses['answer_mismatch']}",
            f"duplicates            : {statuses['duplicate']}",
            f"other errors          : {statuses['error']}",
        ]
        if any(row.variants is not None for row in self.rows):
            lines.append(f"short seeded rows     : {statuses['short']}")
        lines += [
            f"verified              : {self.emitted} "
            f"(worst relative difference {self.worst_relative_difference:.2e})",
            f"readability warnings  : "
            f"{sum(1 for row in self.rows if row.warnings or (row.variants and row.variants.readability_warnings))} rows",
        ]
        if len(self.per_family) > 1:
            lines.append("emitted by family  :")
            for name, count in sorted(self.per_family.items()):
                lines.append(f"    {count:6d}  {name}")
        lines.append("rows                  :")
        for row in self.rows:
            mark = "PASS" if row.passed else ("SHORT" if row.status == "short" else "FAIL")
            where = f"line {row.line}  {row.input_id or '<no input_id>'} [{row.family or '?'}]"
            if row.variants is not None:
                run = row.variants
                ceiling = run.ceilings.get(row.family or "")
                detail = (
                    f"{run.emitted}/{run.requested} variants ({run.draws} draws, "
                    f"{run.invalid} rejected, {run.duplicates} duplicates, ceiling "
                    f"{'unknown' if ceiling is None else ceiling})"
                )
                if not row.passed:
                    detail += f": {row.detail}"
            elif row.passed:
                detail = f"answer={row.answer}"
            else:
                detail = f"{row.status}: {row.detail}"
            lines.append(f"    {mark}  {where}  {detail}")
            if row.variants is not None:
                for reason, count in row.variants.rejection_reasons.most_common():
                    lines.append(f"            {count:6d} rejected: {reason}")
                for warning, count in row.variants.readability_warnings.most_common():
                    lines.append(f"            {count:6d} emitted with readability warning: {warning}")
            for warning in row.warnings:
                lines.append(f"            readability warning (not blocking): {warning}")
        return "\n".join(lines)


def _reason_key(reason: str) -> str:
    """Collapse a rejection reason to its cause, dropping the specific numbers.

    ``is_valid`` embeds actual values in its messages, which is what makes a single
    rejection debuggable and what makes a thousand of them unaggregatable. Cutting
    at the first parenthesis or digit run keeps the cause and discards the instance.
    """
    head = reason.split("(")[0].strip()
    words = [word for word in head.split() if not any(ch.isdigit() for ch in word)]
    return " ".join(words) or head


def _emit_problem(
    family: Any,
    params: Mapping[str, Any],
    *,
    out_dir: Path,
    writer: ManifestWriter,
    dedupe: Deduplicator,
    stats: RunStats,
    next_problem_id: Callable[[], str],
    seed: Optional[int],
    origin: str = "generated",
    check_answer: Optional[Callable[[Any], None]] = None,
    record_fields: Optional[Mapping[str, Any]] = None,
) -> Optional[Datapoint]:
    """Dedupe, solve, verify, render and record one validated parameter set.

    The step every driver shares, whatever chose the params. Returns the written
    record, or ``None`` for a duplicate. Raises ``VerificationError`` when the figure
    and the answer disagree, and whatever ``check_answer`` raises when the answer is
    rejected; either way nothing is rendered or written.

    ``next_problem_id`` is called only once the problem is certain to be emitted, so
    a duplicate or a rejected answer does not consume an id. Likewise, a problem
    that raises before it is written releases its dedupe signature.
    """
    name = family.NAME
    signature = dedupe.add(name, params)
    if signature is None:
        stats.duplicates += 1
        return None

    try:
        return _solve_and_write(
            family,
            params,
            signature,
            out_dir=out_dir,
            writer=writer,
            stats=stats,
            next_problem_id=next_problem_id,
            seed=seed,
            origin=origin,
            check_answer=check_answer,
            record_fields=record_fields,
        )
    except BaseException:
        dedupe.discard(signature)
        raise


def _solve_and_write(
    family: Any,
    params: Mapping[str, Any],
    signature: str,
    *,
    out_dir: Path,
    writer: ManifestWriter,
    stats: RunStats,
    next_problem_id: Callable[[], str],
    seed: Optional[int],
    origin: str,
    check_answer: Optional[Callable[[Any], None]],
    record_fields: Optional[Mapping[str, Any]],
) -> Datapoint:
    """The part of ``_emit_problem`` after dedupe has accepted the problem."""
    name = family.NAME
    stem, answer, spec = family.solve(params)

    # Verify before rendering: a figure that disagrees with its answer
    # should not leave a PNG behind for a later run to mistake for output.
    difference = verify(spec, answer, params=params)
    stats.worst_relative_difference = max(stats.worst_relative_difference, difference)

    if check_answer is not None:
        check_answer(answer)

    problem_id = next_problem_id()
    render(spec, image_path_for(out_dir, problem_id), stem=stem)

    record = Datapoint(
        problem_id=problem_id,
        family=name,
        category=1,
        seed=seed,
        params=params,
        stem=stem,
        image_path=relative_image_path(problem_id),
        answer_exact=str(answer),
        answer_decimal=float(answer.evalf()) if hasattr(answer, "evalf") else float(answer),
        signature=signature,
        generator_version=GENERATOR_VERSION,
        origin=origin,
        **(record_fields or {}),
    )
    writer.write(record)
    stats.emitted += 1
    stats.per_family[name] += 1
    return record


def generate(
    family_name: str,
    n: int,
    seed: int,
    out_dir: Path | str,
    *,
    precision: int = DEFAULT_PRECISION,
    max_draws: Optional[int] = None,
    overwrite: bool = False,
    progress_every: int = 0,
) -> RunStats:
    """Generate ``n`` verified, deduplicated problems into ``out_dir``.

    Returns the stats rather than printing them, so the same loop serves the CLI
    and a notebook. Raises ``VerificationError`` on ground-truth disagreement --
    the one failure that must never be counted and continued past.
    """
    family = registry.get(family_name)
    rng = np.random.default_rng(seed)
    dedupe = Deduplicator(precision=precision)
    stats = RunStats(family=family_name, seed=seed, requested=n)
    ceiling = max_draws if max_draws is not None else n * DEFAULT_DRAW_MULTIPLIER
    out_dir = Path(out_dir)

    limit = registry.ceiling(family, precision=precision)
    stats.ceilings[family_name] = limit
    if limit is not None and n > limit:
        _warn(
            f"{family_name} has {limit} unique valid combinations; n={n} requested "
            f"-- will emit at most {limit} and then stop"
        )

    with ManifestWriter(out_dir, overwrite=overwrite) as writer:
        _draw_into(
            family,
            rng,
            n=n,
            limit=limit,
            max_draws=ceiling,
            out_dir=out_dir,
            writer=writer,
            dedupe=dedupe,
            stats=stats,
            next_problem_id=lambda: problem_id_for(family_name, stats.emitted),
            seed=seed,
            progress_every=progress_every,
        )

    
    return stats


def _draw_into(
    family: Any,
    rng: np.random.Generator,
    *,
    n: int,
    limit: Optional[int],
    max_draws: int,
    out_dir: Path,
    writer: ManifestWriter,
    dedupe: Deduplicator,
    stats: RunStats,
    next_problem_id: Callable[[], str],
    seed: int,
    pinned: Optional[Mapping[str, Any]] = None,
    readability: str = "block",
    origin: str = "generated",
    record_fields: Optional[Mapping[str, Any]] = None,
    progress_every: int = 0,
) -> None:
    """Draw from one family until ``n`` are emitted into ``stats``, or it stops early.

    The loop of ``generate``, shared with seeded rows of ``generate_from_file``. With
    ``pinned`` unset the draw is ``family.sample(rng)`` and the check ``is_valid``,
    exactly as for a plain run; with it, the pins are passed to ``sample`` and
    ``readability`` picks the check (``registry.draw_verdict``).
    """
    name = family.NAME
    streak = 0  # consecutive draws that emitted nothing; only consulted when limit is None
    while stats.emitted < n and stats.draws < max_draws:
        if limit is not None and stats.emitted >= limit:
            stats.stopped_early = (
                f"{name} has emitted all {limit} of its unique valid "
                f"combinations; stopping at {stats.emitted}/{n} emitted"
            )
            break
        if limit is None and streak >= DUPLICATE_STREAK_LIMIT:
            stats.stopped_early = (
                f"no new unique problems found after {streak} consecutive duplicate "
                f"or rejected draws; stopping early at {stats.emitted}/{n} emitted"
            )
            break

        stats.draws += 1
        params = family.sample(rng) if pinned is None else family.sample(rng, pinned)

        reason, warnings = registry.draw_verdict(family, params, readability)
        if reason is not None:
            stats.invalid += 1
            stats.rejection_reasons[_reason_key(reason)] += 1
            streak += 1
            continue

        record = _emit_problem(
            family,
            params,
            out_dir=out_dir,
            writer=writer,
            dedupe=dedupe,
            stats=stats,
            next_problem_id=next_problem_id,
            seed=seed,
            origin=origin,
            record_fields=record_fields,
        )
        if record is None:
            streak += 1
            continue
        streak = 0
        for warning in warnings:
            stats.readability_warnings[_reason_key(warning)] += 1

        if progress_every and stats.emitted % progress_every == 0:
            print(
                f"  {stats.emitted}/{n} emitted ({stats.draws} draws)",
                file=sys.stderr,
                flush=True,
            )


def _warn(message: str) -> None:
    """Print a run-level warning to stderr, where progress also goes, before the run starts."""
    print(f"warning: {message}", file=sys.stderr, flush=True)


def generate_mixed(
    family_names: Sequence[str],
    n: int,
    seed: int,
    out_dir: Path | str,
    *,
    precision: int = DEFAULT_PRECISION,
    max_draws: Optional[int] = None,
    overwrite: bool = False,
    progress_every: int = 0,
) -> RunStats:
    """Generate ``n`` problems spread evenly across several families.

    Round-robin rather than n-per-family, so ``--n 200`` means two hundred problems
    however many families are registered, and a family that exhausts its parameter
    space early does not stall the run -- the others keep drawing.

    Each family gets its own ``Generator``, seeded from the run seed and the
    family's name. That is what keeps adding a family from reshuffling every other
    family's output: a per-family stream is independent of how many streams exist,
    while a single shared stream would renumber everything downstream of the
    insertion point.
    """
    families = {name: registry.get(name) for name in family_names}
    rngs = {
        name: np.random.default_rng(
            [seed, int.from_bytes(name.encode("utf-8")[:8].ljust(8, b"\0"), "little")]
        )
        for name in family_names
    }
    counters = {name: 0 for name in family_names}
    exhausted: dict[str, str] = {}  # family -> why it stopped being drawn from

    dedupe = Deduplicator(precision=precision)
    stats = RunStats(family="+".join(family_names), seed=seed, requested=n)
    ceiling = max_draws if max_draws is not None else n * DEFAULT_DRAW_MULTIPLIER
    out_dir = Path(out_dir)

    limits = {name: registry.ceiling(families[name], precision=precision) for name in family_names}
    stats.ceilings.update(limits)
    share = -(-n // len(family_names))  # ceil: the even split the round-robin aims for
    for name, limit in limits.items():
        if limit is not None and limit < share:
            _warn(
                f"{name} has {limit} unique valid combinations; its even share of n={n} "
                f"across {len(family_names)} families is {share} -- will emit at most "
                f"{limit} from it and let the other families fill the rest"
            )
    streaks = {name: 0 for name in family_names}

    with ManifestWriter(out_dir, overwrite=overwrite) as writer:
        order = list(family_names)
        position = 0
        while stats.emitted < n and stats.draws < ceiling:
            if len(exhausted) == len(order):
                stats.stopped_early = (
                    "every family is exhausted ("
                    + "; ".join(exhausted[name] for name in order)
                    + f"); stopping at {stats.emitted}/{n} emitted"
                )
                break
            name = order[position % len(order)]
            position += 1
            if name in exhausted:
                continue

            limit = limits[name]
            if limit is not None and stats.per_family[name] >= limit:
                exhausted[name] = f"{name} has emitted all {limit} of its unique valid combinations"
                continue
            if limit is None and streaks[name] >= DUPLICATE_STREAK_LIMIT:
                exhausted[name] = (
                    f"{name} found no new unique problem in {streaks[name]} consecutive "
                    f"duplicate or rejected draws"
                )
                continue

            family = families[name]
            stats.draws += 1
            params = family.sample(rngs[name])

            reason = registry.rejection_reason(family.is_valid(params))
            if reason is not None:
                stats.invalid += 1
                stats.rejection_reasons[f"[{name}] {_reason_key(reason)}"] += 1
                streaks[name] += 1
                continue

            def next_problem_id(name: str = name) -> str:
                problem_id = problem_id_for(name, counters[name])
                counters[name] += 1
                return problem_id

            record = _emit_problem(
                family,
                params,
                out_dir=out_dir,
                writer=writer,
                dedupe=dedupe,
                stats=stats,
                next_problem_id=next_problem_id,
                seed=seed,
            )
            if record is None:
                streaks[name] += 1
                continue
            streaks[name] = 0

            if progress_every and stats.emitted % progress_every == 0:
                print(f"  {stats.emitted}/{n} emitted ({stats.draws} draws)", file=sys.stderr, flush=True)

    return stats


def generate_from_file(
    path: Path | str,
    out_dir: Path | str,
    *,
    seed: int = 0,
    precision: int = DEFAULT_PRECISION,
    overwrite: bool = False,
    progress_every: int = 0,
) -> RunStats:
    """Emit one problem per ordinary row of a structured-input file, and ``variants``
    problems per seeded row, in file order.

    Each row replaces a random draw. Unlike a draw, a row is only blocked by
    geometry issues: readability issues are recorded as warnings, since those
    checks exist to filter noisy draws and a real problem is not noise. A row is
    emitted only if the family's answer matches its ``expected_answer`` to within
    ``verify.TOLERANCE``.

    A seeded row runs ``generate``'s draw loop (``_draw_into``) with its params
    pinned, on a stream derived from ``seed`` and its ``input_id``, so its variants
    are validated, deduplicated and verified exactly as random draws are, and do not
    change when other rows are added, removed or reordered.

    Row-level failures are recorded in ``stats.rows`` and the run continues.
    ``VerificationError`` and ``EmitError`` still abort, as in ``generate``: the
    first means the pipeline itself is unsound, the second that nothing can be
    written. Raises ``StructuredInputError`` if the file cannot be read at all.
    """
    rows = load_rows(path)
    dedupe = Deduplicator(precision=precision)
    requested = sum(
        row.variants if isinstance(row, StructuredProblem) and row.seeded else 1 for row in rows
    )
    stats = RunStats(
        family=f"from-file {Path(path).name}",
        seed=seed if any(isinstance(row, StructuredProblem) and row.seeded for row in rows) else None,
        requested=requested,
        structured=True,
    )
    out_dir = Path(out_dir)

    with ManifestWriter(out_dir, overwrite=overwrite) as writer:
        for row in rows:
            stats.draws += 1
            if isinstance(row, StructuredProblem) and row.seeded:
                outcome = _seeded_row(
                    row,
                    seed=seed,
                    precision=precision,
                    out_dir=out_dir,
                    writer=writer,
                    dedupe=dedupe,
                    stats=stats,
                )
                stats.rows.append(outcome)
                continue
            outcome = _structured_row(row, out_dir=out_dir, writer=writer, dedupe=dedupe, stats=stats)
            stats.rows.append(outcome)
            if outcome.status in ("input_error", "geometry_rejected"):
                stats.invalid += 1
                stats.rejection_reasons[_reason_key(outcome.detail)] += 1
            for warning in outcome.warnings:
                stats.readability_warnings[_reason_key(warning)] += 1

            if progress_every and stats.draws % progress_every == 0:
                print(
                    f"  {stats.draws}/{len(rows)} rows ({stats.emitted} emitted)",
                    file=sys.stderr,
                    flush=True,
                )

    return stats


def _row_stream(input_id: str) -> int:
    """A seeded row's stream key: stable across runs and Python versions, and distinct
    for ids that share a prefix (which ``generate_mixed``'s 8-byte name key is not)."""
    return int.from_bytes(hashlib.sha256(input_id.encode("utf-8")).digest()[:8], "little")


def _seeded_row(
    row: StructuredProblem,
    *,
    seed: int,
    precision: int,
    out_dir: Path,
    writer: ManifestWriter,
    dedupe: Deduplicator,
    stats: RunStats,
) -> RowOutcome:
    """Draw a seeded row's variants into the run and say what happened.

    The row's own draw counts live on a ``RunStats`` of its own, so a row asking for
    twenty variants does not read as twenty rows; what it emitted is added to the
    run's totals. Dedupe is shared with the rest of the file, so a variant never
    repeats another row's problem.
    """
    family = registry.get(row.family)
    assert row.variants is not None
    run = RunStats(family=row.input_id, seed=seed, requested=row.variants)
    valid = registry.valid_signatures(
        family, precision=precision, pinned=row.params, readability=row.readability
    )
    # What this row can still emit: its pinned space, less what earlier rows took.
    limit = None if valid is None else len(valid - dedupe.signatures)
    run.ceilings[row.family] = limit
    if limit is not None and row.variants > limit:
        _warn(
            f"{row.input_id}: {limit} unique valid variants with these pins; "
            f"{row.variants} requested -- will emit at most {limit}"
        )
    ids = row.variant_ids()

    outcome = RowOutcome(row.line, row.input_id, row.family, "emitted", variants=run)
    try:
        _draw_into(
            family,
            np.random.default_rng([seed, _row_stream(row.input_id)]),
            n=row.variants,
            limit=limit,
            max_draws=row.variants * DEFAULT_DRAW_MULTIPLIER,
            out_dir=out_dir,
            writer=writer,
            dedupe=dedupe,
            stats=run,
            next_problem_id=lambda: ids[run.emitted],
            seed=seed,
            pinned=row.params,
            readability=row.readability,
            origin="seeded",
            record_fields={"parent_input_id": row.input_id, "pinned": row.params, **row.provenance()},
        )
    except (VerificationError, EmitError):
        raise
    except Exception as exc:  # a row-level bug (schema, solve) is reported, not fatal
        outcome.status, outcome.detail = "error", f"{type(exc).__name__}: {exc}"

    stats.emitted += run.emitted
    stats.per_family.update(run.per_family)
    stats.worst_relative_difference = max(
        stats.worst_relative_difference, run.worst_relative_difference
    )
    for warning, count in run.readability_warnings.items():
        stats.readability_warnings[warning] += count

    if outcome.status == "emitted" and not run.complete:
        outcome.status = "short"
        if limit is not None and run.emitted >= limit:
            assert valid is not None
            taken = len(valid) - limit
            outcome.detail = (
                f"only {len(valid)} unique valid variants with these pins"
                + (f", {taken} already emitted by earlier rows" if taken else "")
            )
            if not valid and row.readability == "block":
                outcome.detail += ' (try "readability": "warn")'
        else:
            outcome.detail = run.stopped_early or (
                f"stopped after {run.draws} draws with {run.emitted}/{run.requested} emitted"
            )
    return outcome


def _structured_row(
    row: Union[StructuredProblem, RowError],
    *,
    out_dir: Path,
    writer: ManifestWriter,
    dedupe: Deduplicator,
    stats: RunStats,
) -> RowOutcome:
    """Run one structured row through ``_emit_problem`` and say what happened."""
    if isinstance(row, RowError):
        return RowOutcome(row.line, row.input_id, row.family, "input_error", row.reason)

    family = registry.get(row.family)
    # TODO: revisit the wording of the rejection and warning messages (e.g. "inner
    # polygon does not fit with clearance") once --from-file has run on the real
    # hand-sourced problem set and we can see which messages actually fire.
    blocking, warnings = registry.classify_issues(family, row.params)
    outcome = RowOutcome(row.line, row.input_id, row.family, "emitted", warnings=warnings)
    if blocking is not None:
        outcome.status, outcome.detail = "geometry_rejected", blocking
        return outcome

    expected = float(parse_expected_answer(row.expected_answer).evalf())

    def check_answer(answer: Any) -> None:
        computed = float(answer.evalf())
        difference = relative_difference(computed, expected)
        if difference >= TOLERANCE:
            raise AnswerMismatch(
                f"computed {answer} = {computed!r}, expected {row.expected_answer} = "
                f"{expected!r} (relative difference {difference:.3e} >= {TOLERANCE:.0e})"
            )

    try:
        record = _emit_problem(
            family,
            row.params,
            out_dir=out_dir,
            writer=writer,
            dedupe=dedupe,
            stats=stats,
            next_problem_id=lambda: row.input_id,
            seed=None,
            origin="structured",
            check_answer=check_answer,
            record_fields={"expected_answer": row.expected_answer, **row.provenance()},
        )
    except AnswerMismatch as exc:
        outcome.status, outcome.detail = "answer_mismatch", str(exc)
        return outcome
    except (VerificationError, EmitError):
        raise
    except Exception as exc:  # a row-level bug (schema, solve) is reported, not fatal
        outcome.status, outcome.detail = "error", f"{type(exc).__name__}: {exc}"
        return outcome

    if record is None:
        outcome.status = "duplicate"
        outcome.detail = "same family and params as an earlier row"
        return outcome
    outcome.answer = record.answer_exact
    return outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m generator.cli",
        description="Generate verified diagram-dependent geometry problems.",
    )
    parser.add_argument(
        "--family",
        default="nested_polygons",
        help="family to generate from, or 'all' to spread the run across every "
        "registered family (default: %(default)s)",
    )
    parser.add_argument("--n", type=int, default=10, help="how many problems to emit")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the run")
    parser.add_argument(
        "--out", type=Path, help="output directory for manifest and images (required to generate)"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print acceptance rate, collision rate and verification count at the end",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=DEFAULT_PRECISION,
        help="decimal places kept when hashing params for dedupe (default: %(default)s)",
    )
    parser.add_argument(
        "--max-draws",
        type=int,
        default=None,
        help=f"draw ceiling before giving up (default: {DEFAULT_DRAW_MULTIPLIER} x --n)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing manifest in --out"
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="print progress to stderr every N problems (0 disables)",
    )
    parser.add_argument(
        "--list-families", action="store_true", help="list registered families and exit"
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        default=None,
        metavar="PROBLEMS.jsonl",
        help="emit one problem per row of a structured-input JSONL file instead of "
        "drawing at random, or --seed-driven variants for rows with 'variants'; "
        "excludes --family, --n and --max-draws",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns an exit code rather than calling ``sys.exit``, so the
    driver is testable without catching ``SystemExit``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handled after parsing rather than by scanning argv, so that --list-families
    # goes through the same validation and error reporting as everything else.
    if args.list_families:
        for name in registry.available():
            print(name)
        return 0

    if args.from_file is not None:
        return _main_from_file(parser, args, argv)

    if args.out is None:
        parser.error("--out is required when generating")
    if args.n <= 0:
        parser.error("--n must be positive")

    try:
        #driver is some random variable
        driver = generate_mixed if args.family == "all" else generate
        target = registry.available() if args.family == "all" else args.family
        stats = driver(
            target,
            args.n,
            args.seed,
            args.out,
            precision=args.precision,
            max_draws=args.max_draws,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
        )
    except KeyError:
        print(
            f"unknown family {args.family!r}; registered: {registry.available()}",
            file=sys.stderr,
        )
        return 1
    except (EmitError, VerificationError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.report:
        print(stats.report())

    if not stats.complete:
        if stats.stopped_early:
            print(f"stopped early: {stats.stopped_early}", file=sys.stderr)
        else:
            print(
                f"stopped after {stats.draws} draws with {stats.emitted}/{stats.requested} "
                f"emitted. The parameter space may be smaller than the request; raise "
                f"--max-draws or lower --n.",
                file=sys.stderr,
            )
        return 2
    return 0


#: Options that only make sense for random draws. ``--seed`` is shared: it seeds the
#: variants of a structured file's seeded rows.
_DRAW_ONLY_OPTIONS = {"family": "--family", "n": "--n", "max_draws": "--max-draws"}


def _explicit_options(argv: Optional[Sequence[str]]) -> set[str]:
    """The dests the user actually passed, as opposed to ones left at their default.

    Needed because ``--n 10`` and no ``--n`` parse identically. Reparses with every
    default suppressed, so only given options appear in the namespace.
    """
    probe = build_parser()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    return set(vars(probe.parse_args(argv)))


def _main_from_file(
    parser: argparse.ArgumentParser, args: argparse.Namespace, argv: Optional[Sequence[str]]
) -> int:
    conflicting = sorted(
        flag for dest, flag in _DRAW_ONLY_OPTIONS.items() if dest in _explicit_options(argv)
    )
    if conflicting:
        parser.error(f"--from-file cannot be combined with {', '.join(conflicting)}")
    if args.out is None:
        parser.error("--out is required when generating")

    try:
        stats = generate_from_file(
            args.from_file,
            args.out,
            seed=args.seed,
            precision=args.precision,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
        )
    except (StructuredInputError, EmitError, VerificationError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.report:
        print(stats.report())

    if not stats.complete:
        failed = [row for row in stats.rows if not row.passed]
        print(f"{len(failed)} of {len(stats.rows)} rows did not complete:", file=sys.stderr)
        for row in failed:
            print(
                f"  line {row.line} {row.input_id or '<no input_id>'}: {row.status}: {row.detail}",
                file=sys.stderr,
            )
        # Only seeded rows falling short, as a plain run does at its ceiling: exit 2.
        return 2 if all(row.status == "short" for row in failed) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
