"""Command-line entry point: draw, validate, solve, render, verify, dedupe, emit.

Run with::

    python -m generator.cli --family nested_polygons --n 200 --seed 0 --out runs/dev

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
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

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
from generator.verify import VerificationError, verify

__all__ = ["RunStats", "generate", "generate_mixed", "main"]

#: Multiple of ``--n`` used as the default draw ceiling. Without a ceiling, a
#: family whose space is smaller than the request spins forever; 200x is loose
#: enough that a healthy family never reaches it.
DEFAULT_DRAW_MULTIPLIER = 200


@dataclass
class RunStats:
    """What a run did, in the terms needed to diagnose a disappointing one."""

    family: str
    seed: int
    requested: int
    draws: int = 0
    invalid: int = 0
    duplicates: int = 0
    emitted: int = 0
    worst_relative_difference: float = 0.0
    rejection_reasons: Counter = field(default_factory=Counter)
    per_family: Counter = field(default_factory=Counter)

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
        lines = [
            f"family                : {self.family}",
            f"seed                  : {self.seed}",
            f"emitted               : {self.emitted}/{self.requested}"
            + ("" if self.complete else "   *** SHORT ***"),
            f"draws                 : {self.draws}",
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


def _reason_key(reason: str) -> str:
    """Collapse a rejection reason to its cause, dropping the specific numbers.

    ``is_valid`` embeds actual values in its messages, which is what makes a single
    rejection debuggable and what makes a thousand of them unaggregatable. Cutting
    at the first parenthesis or digit run keeps the cause and discards the instance.
    """
    head = reason.split("(")[0].strip()
    words = [word for word in head.split() if not any(ch.isdigit() for ch in word)]
    return " ".join(words) or head


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

    with ManifestWriter(out_dir, overwrite=overwrite) as writer:
        while stats.emitted < n and stats.draws < ceiling:
            stats.draws += 1
            params = family.sample(rng)

            reason = registry.rejection_reason(family.is_valid(params))
            if reason is not None:
                stats.invalid += 1
                stats.rejection_reasons[_reason_key(reason)] += 1
                continue

            signature = dedupe.add(family_name, params)
            if signature is None:
                stats.duplicates += 1
                continue

            stem, answer, spec = family.solve(params)

            # Verify before rendering: a figure that disagrees with its answer
            # should not leave a PNG behind for a later run to mistake for output.
            difference = verify(spec, answer, params=params)
            stats.worst_relative_difference = max(
                stats.worst_relative_difference, difference
            )

            problem_id = problem_id_for(family_name, stats.emitted)
            render(spec, image_path_for(out_dir, problem_id), stem=stem)

            record = Datapoint(
                problem_id=problem_id,
                family=family_name,
                category=1,
                seed=seed,
                params=params,
                stem=stem,
                image_path=relative_image_path(problem_id),
                answer_exact=str(answer),
                answer_decimal=float(answer.evalf())
                if hasattr(answer, "evalf")
                else float(answer),
                signature=signature,
                generator_version=GENERATOR_VERSION,
                origin="generated",
            )
            writer.write(record)
            stats.emitted += 1

            if progress_every and stats.emitted % progress_every == 0:
                print(
                    f"  {stats.emitted}/{n} emitted ({stats.draws} draws)",
                    file=sys.stderr,
                    flush=True,
                )

    return stats


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
    exhausted: set[str] = set()

    dedupe = Deduplicator(precision=precision)
    stats = RunStats(family="+".join(family_names), seed=seed, requested=n)
    ceiling = max_draws if max_draws is not None else n * DEFAULT_DRAW_MULTIPLIER
    out_dir = Path(out_dir)

    with ManifestWriter(out_dir, overwrite=overwrite) as writer:
        order = list(family_names)
        position = 0
        while stats.emitted < n and stats.draws < ceiling:
            if len(exhausted) == len(order):
                break
            name = order[position % len(order)]
            position += 1
            if name in exhausted:
                continue

            family = families[name]
            stats.draws += 1
            params = family.sample(rngs[name])

            reason = registry.rejection_reason(family.is_valid(params))
            if reason is not None:
                stats.invalid += 1
                stats.rejection_reasons[f"[{name}] {_reason_key(reason)}"] += 1
                continue

            signature = dedupe.add(name, params)
            if signature is None:
                stats.duplicates += 1
                continue

            stem, answer, spec = family.solve(params)
            difference = verify(spec, answer, params=params)
            stats.worst_relative_difference = max(stats.worst_relative_difference, difference)

            problem_id = problem_id_for(name, counters[name])
            counters[name] += 1
            render(spec, image_path_for(out_dir, problem_id), stem=stem)

            writer.write(
                Datapoint(
                    problem_id=problem_id,
                    family=name,
                    category=1,
                    seed=seed,
                    params=params,
                    stem=stem,
                    image_path=relative_image_path(problem_id),
                    answer_exact=str(answer),
                    answer_decimal=float(answer.evalf())
                    if hasattr(answer, "evalf")
                    else float(answer),
                    signature=signature,
                    generator_version=GENERATOR_VERSION,
                    origin="generated",
                )
            )
            stats.emitted += 1
            stats.per_family[name] += 1

            if progress_every and stats.emitted % progress_every == 0:
                print(f"  {stats.emitted}/{n} emitted ({stats.draws} draws)", file=sys.stderr, flush=True)

    return stats


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
        print(
            f"stopped after {stats.draws} draws with {stats.emitted}/{stats.requested} "
            f"emitted. The parameter space may be smaller than the request; raise "
            f"--max-draws or lower --n.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
