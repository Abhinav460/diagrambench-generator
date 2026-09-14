"""Turn a harvested record's published LaTeX answer into exact + decimal forms.

Text parsing only. The input is the answer string the source site published as
markup; nothing here looks at a diagram.

The page writes answers as prose with the mathematics inlined::

    The area is \\( 8+4\\sqrt{3}\\) cm\\(^2 \\)

so the work is to isolate the mathematics from the sentence around it without
guessing. Three rules do it:

  * Only ``\\( ... \\)`` spans are mathematics. Everything outside them is prose
    ("The area is") or a unit ("cm"), and is dropped.
  * A span that is nothing but an exponent (``^2``) is a unit's superscript, not
    an answer, and is dropped too.
  * A span containing ``=`` is an equation ("x=30"); the answer is its right side.

Anything these rules cannot resolve is left unparsed and reported. A wrong
answer that looks parsed is far worse than an obvious gap: this file feeds a
benchmark's ground truth, and the failure mode of guessing is a model being
marked wrong for being right.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "manifest_harvested.jsonl"

#: Inline math delimiters used by the source page.
MATH_SPAN = re.compile(r"\\\((.*?)\\\)", re.DOTALL)

#: A span that is only a superscript belongs to a unit ("cm\(^2\)"), not to the
#: answer. The caret is required: without it this also matches a bare number, and
#: an answer of "\(8\)" would be discarded as if it were a unit.
UNIT_EXPONENT = re.compile(r"^\s*\^\s*\{?\s*\d+\s*\}?\s*$")

#: Rendering directives carry no mathematical content, but sympy's LaTeX parser
#: has no rule for them and turns them into a free symbol -- so
#: ``\displaystyle\frac{a}{b}`` parses as ``displaystyle*(a/b)``, silently
#: multiplying the answer by an undefined variable.
NOISE = (r"\displaystyle", r"\dfrac", r"\tfrac", r"\left", r"\right", r"\!", r"\,", r"\;")


class AnswerParseError(ValueError):
    """Raised when an answer cannot be resolved without guessing."""


def math_spans(answer: str) -> list[str]:
    """The mathematical spans of a published answer, units discarded."""
    return [
        span.strip()
        for span in MATH_SPAN.findall(answer)
        if span.strip() and not UNIT_EXPONENT.match(span)
    ]


def clean(span: str) -> str:
    span = span.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    for token in NOISE:
        if token in (r"\dfrac", r"\tfrac"):
            continue
        span = span.replace(token, " ")
    return re.sub(r"\s+", " ", span).strip()


def to_expression(answer: str) -> Any:
    """Parse one published answer into a sympy expression, or raise."""
    from sympy.parsing.latex import parse_latex

    spans = math_spans(answer)
    if not spans:
        raise AnswerParseError("no mathematical span found")
    if len(spans) > 1:
        raise AnswerParseError(
            f"{len(spans)} mathematical spans, ambiguous which is the answer: {spans}"
        )

    span = clean(spans[0])
    if "\\approx" in span:
        # "3-2\sqrt{2}\approx 0.17157" states the exact form and its rounding.
        # The exact form is the answer; the decimal is the site's own rounding and
        # is recomputed here at full precision rather than trusted.
        span = span.split("\\approx")[0].strip()

    try:
        expr = parse_latex(span)
    except Exception as exc:  # sympy raises a wide variety here
        raise AnswerParseError(f"parse_latex failed on {span!r}: {type(exc).__name__}") from exc

    if expr is None:
        raise AnswerParseError(f"parse_latex returned None for {span!r}")

    # "x=30": the answer is the value, not the equation.
    if hasattr(expr, "rhs"):
        expr = expr.rhs

    # sympy's LaTeX parser renders \pi as a plain Symbol named "pi", not as the
    # constant. Left alone it stays a free symbol, so an answer of 81*pi + 1296
    # would be rejected as unresolved -- and if it were not, evalf() could not
    # produce a number. Substituted by exact name, so nothing else is touched.
    import sympy as sp

    expr = expr.subs({sp.Symbol("pi"): sp.pi, sp.Symbol("e"): sp.E})

    expr = _canonicalize(expr)

    free = getattr(expr, "free_symbols", set())
    if free:
        raise AnswerParseError(
            f"expression still contains free symbols {sorted(map(str, free))} -- "
            f"unresolved LaTeX in {span!r}"
        )
    return expr


def _canonicalize(expr: Any) -> Any:
    """Put the expression in the exact form the generator's own answers use.

    Two adjustments, both value-preserving and both checked:

      * A decimal coefficient the site wrote (431.25) becomes the rational it
        exactly equals (1725/4), so the record is genuinely exact rather than a
        float wearing an exact form. Applied only when the rational and the float
        agree exactly -- a repeating decimal is left alone.
      * ``simplify`` collapses (450*sqrt(3))/2 to 225*sqrt(3), matching how
        ``solve`` writes its answers, so harvested and generated records are
        comparable as strings.

    The numeric value is asserted unchanged; a canonicalization that moved the
    answer would be a silent corruption of ground truth.
    """
    import sympy as sp

    before = complex(sp.N(expr, 30))

    if expr.atoms(sp.Float):
        rational = sp.nsimplify(expr, rational=True)
        if sp.N(rational, 30) == sp.N(expr, 30):
            expr = rational

    simplified = sp.simplify(expr)
    if sp.N(simplified, 30) == sp.N(expr, 30):
        expr = simplified

    after = complex(sp.N(expr, 30))
    if abs(after - before) > 1e-18 * max(1.0, abs(before)):
        raise AnswerParseError(
            f"canonicalization changed the value: {before} -> {after}"
        )
    return expr


def parsed_fields(answer: str) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """Return (answer_exact, answer_decimal, error). Exactly one side is None."""
    try:
        expr = to_expression(answer)
        decimal = float(expr.evalf())
    except AnswerParseError as exc:
        return None, None, str(exc)
    except (TypeError, ValueError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"

    import math

    if not math.isfinite(decimal):
        return None, None, f"non-finite value {decimal!r}"
    return str(expr), decimal, None


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"no manifest at {MANIFEST_PATH}", file=sys.stderr)
        return 1

    records = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines() if line.strip()]
    failures: list[tuple[str, str, str]] = []

    print(f"{'problem_id':<16}{'answer_exact':<34}{'answer_decimal':>18}")
    print("-" * 70)
    for record in records:
        raw = record.get("original_answer") or ""
        exact, decimal, error = parsed_fields(raw)
        record["answer_exact"] = exact
        record["answer_decimal"] = decimal
        if error:
            failures.append((record["problem_id"], raw, error))
            print(f"{record['problem_id']:<16}{'-- FAILED --':<34}{'':>18}")
        else:
            print(f"{record['problem_id']:<16}{exact:<34}{decimal:>18.10f}")

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"\nparsed {len(records) - len(failures)}/{len(records)}")
    if failures:
        print("\nFAILURES (left null, not guessed):")
        for pid, raw, error in failures:
            print(f"  {pid}\n    raw   : {raw!r}\n    reason: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
