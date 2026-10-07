"""Turn an answer written as text into an exact sympy expression.

Two input forms, one module, so that anything packaging problems into a dataset can
parse answers without importing the family registry (and through it every family,
shapely and matplotlib):

- **Plain expressions** such as ``150*sqrt(3) - 75*pi``, the form a hand-entered
  answer takes. ``parse_expected_answer`` checks the string token by token before
  sympy sees it: only numbers, the operators ``+ - * / ^ ** ( ) ,`` and the names in
  ``ANSWER_NAMES`` are allowed, and the parse then runs with no builtins in scope.
  An input file cannot run code.
- **Published LaTeX** such as ``The area is \\( 8+4\\sqrt{3}\\) cm\\(^2 \\)``, the form a
  source page prints. ``parsed_fields`` isolates the mathematics from the sentence
  around it without guessing. Three rules do it:

  * Only ``\\( ... \\)`` spans are mathematics. Everything outside them is prose
    ("The area is") or a unit ("cm"), and is dropped.
  * A span that is nothing but an exponent (``^2``) is a unit's superscript, not
    an answer, and is dropped too.
  * A span containing ``=`` is an equation ("x=30"); the answer is its right side.

  Anything these rules cannot resolve is left unparsed and reported. A wrong answer
  that looks parsed is far worse than an obvious gap: this feeds a benchmark's
  ground truth, and the failure mode of guessing is a model being marked wrong for
  being right. LaTeX parsing needs sympy's ``antlr4`` backend
  (``antlr4-python3-runtime`` 4.11), imported only when it is used.

sympy is imported lazily throughout, so importing this module is free.
"""

from __future__ import annotations

import io
import keyword
import math
import re
import tokenize
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sympy import Expr
else:
    Expr = Any

__all__ = [
    "ANSWER_NAMES",
    "AnswerParseError",
    "math_spans",
    "parse_expected_answer",
    "parsed_fields",
    "to_expression",
]


# --- plain expressions --------------------------------------------------------------

#: The only names an ``expected_answer`` may use; any other name is an unknown symbol.
ANSWER_NAMES = ("sqrt", "cbrt", "pi", "E", "Rational", "Abs", "sin", "cos", "tan")
_ANSWER_OPS = frozenset({"+", "-", "*", "/", "**", "^", "(", ")", ","})
_LAYOUT_TOKENS = frozenset(
    {tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
)


def _check_answer_tokens(text: str) -> None:
    """Allow only numbers, arithmetic operators and ``ANSWER_NAMES``, or raise ``ValueError``.

    sympy's parser evaluates its input as Python, so this runs first: attribute
    access, subscripts, strings, keywords and unknown names never reach ``eval``.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError) as exc:
        raise ValueError(f"expected_answer {text!r} does not parse: {type(exc).__name__}: {exc}") from None

    unknown: set[str] = set()
    for token in tokens:
        if token.type == tokenize.NAME:
            if keyword.iskeyword(token.string):
                raise ValueError(f"expected_answer {text!r} does not parse: keyword {token.string!r} is not allowed")
            if token.string not in ANSWER_NAMES:
                unknown.add(token.string)
        elif token.type == tokenize.OP:
            if token.string not in _ANSWER_OPS:
                raise ValueError(f"expected_answer {text!r} does not parse: operator {token.string!r} is not allowed")
        elif token.type != tokenize.NUMBER and token.type not in _LAYOUT_TOKENS:
            raise ValueError(
                f"expected_answer {text!r} does not parse: "
                f"{tokenize.tok_name[token.type]} {token.string!r} is not allowed"
            )
    if unknown:
        raise ValueError(f"expected_answer {text!r} contains unknown symbols {sorted(unknown)}")


def _answer_namespace() -> dict[str, Any]:
    """Everything ``eval`` can see while parsing an answer.

    ``ANSWER_NAMES`` plus the constructors sympy's own tokenizer emits
    (``Integer('7')``, ``Float('.5')``, ``I`` for a ``j`` literal) and no builtins.
    """
    import sympy as sp

    namespace: dict[str, Any] = {name: getattr(sp, name) for name in ANSWER_NAMES}
    namespace.update(Integer=sp.Integer, Float=sp.Float, I=sp.I, __builtins__={})
    return namespace


def parse_expected_answer(text: str) -> Expr:
    """Parse an answer string into a constant sympy expression, or raise ``ValueError``."""
    from sympy.parsing.sympy_parser import (
        convert_xor,
        parse_expr,
        standard_transformations,
    )

    _check_answer_tokens(text)
    try:
        expr = parse_expr(
            text,
            local_dict={},
            global_dict=_answer_namespace(),
            transformations=standard_transformations + (convert_xor,),
        )
    except Exception as exc:  # parse_expr raises SyntaxError, TypeError, TokenError, ...
        raise ValueError(f"expected_answer {text!r} does not parse: {type(exc).__name__}: {exc}") from None
    if getattr(expr, "free_symbols", None):
        raise ValueError(
            f"expected_answer {text!r} contains unknown symbols "
            f"{sorted(str(s) for s in expr.free_symbols)}"
        )
    try:
        value = complex(expr.evalf())
    except (TypeError, ValueError, AttributeError):
        raise ValueError(f"expected_answer {text!r} is not a number") from None
    if value.imag != 0 or not math.isfinite(value.real):
        raise ValueError(f"expected_answer {text!r} is not a finite real number")
    return expr


# --- published LaTeX ----------------------------------------------------------------

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

    if not math.isfinite(decimal):
        return None, None, f"non-finite value {decimal!r}"
    return str(expr), decimal, None
