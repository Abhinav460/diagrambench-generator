"""generator.answers: the answer parsers, importable without the family registry."""

from __future__ import annotations

import subprocess
import sys

import pytest

from generator import answers, structured


def test_importing_answers_loads_no_registry_or_numerics():
    code = (
        "import sys, generator.answers; "
        "print(sorted(m for m in sys.modules if m.startswith("
        "('generator.registry', 'generator.families', 'sympy', 'shapely', 'matplotlib'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_structured_re_exports_the_same_parser():
    assert structured.parse_expected_answer is answers.parse_expected_answer
    assert structured.ANSWER_NAMES is answers.ANSWER_NAMES


def test_a_plain_answer_parses():
    import sympy as sp

    assert answers.parse_expected_answer("150*sqrt(3) - 75*pi") == 150 * sp.sqrt(3) - 75 * sp.pi


def test_a_plain_answer_cannot_run_code():
    with pytest.raises(ValueError, match="not allowed|unknown symbols"):
        answers.parse_expected_answer("__import__('os').system('true')")


@pytest.mark.parametrize(
    "published,spans",
    [
        (r"The area is \( 8+4\sqrt{3}\) cm\(^2 \)", [r"8+4\sqrt{3}"]),
        (r"\(8\)", ["8"]),
        ("no maths here", []),
    ],
)
def test_math_spans_drop_prose_and_unit_exponents(published, spans):
    assert answers.math_spans(published) == spans


def test_an_answer_with_no_math_span_is_reported_not_guessed():
    assert answers.parsed_fields("The area is eight.") == (None, None, "no mathematical span found")


def test_two_math_spans_are_ambiguous():
    exact, decimal, error = answers.parsed_fields(r"\(3\) or \(4\)")
    assert exact is None and decimal is None and "ambiguous" in error


@pytest.mark.parametrize(
    "published,exact",
    [
        (r"The area is \( 8+4\sqrt{3}\) cm\(^2 \)", "4*sqrt(3) + 8"),
        (r"\(x=30\)", "30"),
        (r"\(81\pi + 1296\)", "81*pi + 1296"),
        (r"\(431.25\)", "1725/4"),
    ],
)
def test_published_latex_parses_to_an_exact_answer(published, exact):
    pytest.importorskip("antlr4", reason="sympy's LaTeX parser needs antlr4-python3-runtime 4.11")
    got, decimal, error = answers.parsed_fields(published)
    assert error is None and got == exact and decimal is not None
