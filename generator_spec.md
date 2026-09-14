# DiagramBench Problem Generator — Spec

Target: automated generation of **Category 1** (diagram-dependent) geometry problems
for DiagramBench, with exact ground-truth answers derived by construction.

Status: draft v1, written 2026-09-02. Supersedes nothing — this is the document the
harvest build order kept referencing but which never existed in written form.

---

## 0. Why this is much smaller than the harvest pipeline

The harvest build order has ten steps, a fetch cache, a per-site parser contract, an
answer-string parser, a `needs_review.jsonl` escape hatch, perceptual-hash dedupe, and a
manual review pass to estimate transcription error rate.

Almost none of that is needed here, because **scraping and generating have opposite
information flow**:

| Harvest problem | Why generation doesn't have it |
|---|---|
| Irregular site markup | No HTML. Nothing to parse. |
| Answers as `12√3`, `about 20.8`, or an image | We *compute* the answer. There is no answer string to interpret. |
| Unknown provenance of the existing 96 | Every record carries the exact params that produced it. |
| Published answer might be wrong | Answer is derived from the same params that drew the figure. |
| Manual spot-check of 30 to estimate error rate | Automated verification covers 100% (§5). |
| Perceptual-hash near-duplicate detection | Two problems are identical iff their param tuples are. Exact, cheap. |

What survives from that plan is exactly two things: the `Datapoint` schema, and the idea
of putting the volatile part behind a stable interface. Everything else is scraping
infrastructure that has no counterpart here.

**The core is three files.** `schema.py`, one family module, `generate.py`. Everything
below that is optional and should be added only when a specific need appears.

---

## 1. The design principle everything follows from

> The parameters are the ground truth. The diagram and the answer are both *renderings*
> of the same parameter set.

A problem is not "a picture plus an answer we hope is right." It is a parameter dict,
from which two independent projections are produced: a PNG for the model to look at, and
an exact symbolic answer for grading. They cannot disagree, because neither is derived
from the other — both come from the params.

This is why generation gives a stronger methodology sentence in the paper than harvesting
does. Harvesting yields "we manually verified a random sample of thirty." Generation
yields "ground truth is exact by construction and independently re-verified numerically
for every record."

---

## 2. `schema.py` — write this first

Shared with the harvest pipeline. Same `Datapoint`, `origin` distinguishes the two.
Stdlib dataclasses, no dependency, so it imports in a bare Colab cell.

```python
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional
import json, time

@dataclass
class Datapoint:
    # identity
    problem_id: str                      # "gen-circ-tangent-0041"
    origin: Literal["generated", "harvested"]
    category: Literal[1, 2]              # 1 = diagram-dependent, 2 = textual

    # the problem as the model sees it
    prompt_text: str                     # "Find the area of the blue region."
    image_path: Optional[str]            # None for category 2

    # ground truth
    answer_exact: str                    # sympy srepr or LaTeX, e.g. "8 + 4*sqrt(3)"
    answer_decimal: Optional[float]      # float(answer_exact.evalf()), for tolerance grading
    answer_units: Optional[str]          # "cm^2", "degrees", None

    # generation provenance (empty for harvested)
    family: Optional[str] = None         # "circle_tangent_square"
    params: dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    generator_version: Optional[str] = None

    # harvest provenance (empty for generated)
    source_name: Optional[str] = None    # "andymath"
    source_url: Optional[str] = None
    source_problem_number: Optional[str] = None
    retrieved_at: Optional[str] = None

    # analysis tags
    traps: list[str] = field(default_factory=list)   # see §6
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))

    def validate(self) -> None:
        assert self.origin in ("generated", "harvested")
        assert self.category in (1, 2)
        if self.category == 1:
            assert self.image_path, "category 1 requires a diagram"
        if self.origin == "generated":
            assert self.family and self.params and self.seed is not None
        assert self.answer_exact
        assert not self._leaks(), f"prompt leaks geometry: {self.prompt_text!r}"

    def _leaks(self) -> bool:
        """Cat-1 prompts may name the target quantity only, never a measurement."""
        import re
        return bool(re.search(r"\d", self.prompt_text))

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
```

Note `_leaks()`. The paper's whole claim rests on Category 1 prompts carrying no geometric
content beyond the asked-for quantity. A digit in the prompt is the cheapest possible
detector for an accidental leak, and it runs on every record. Tighten later if a legitimate
prompt needs a numeral.

---

## 3. The family contract

A **family** is one parametrized problem type — "circle inscribed in a right triangle,"
"three squares on a line with a semicircle," "two overlapping quarter-circles in a square."
Each family is one module in `families/`, exposing four functions. This is the generator's
analogue of the harvest plan's per-site contract, and it exists for the same reason: it is
what lets you add the twentieth family without touching anything else.

```python
from typing import Protocol, Any
import sympy as sp

Params = dict[str, Any]

class Family(Protocol):
    name: str
    category_tags: list[str]   # "composite", "circular", "shaded", "nested", "coordinate"

    def sample(self, rng) -> Params:
        """Draw a *valid, non-degenerate* configuration. Must never return a
        configuration that solve() or render() cannot handle. Rejection-sample
        rather than clamping — clamping silently biases the distribution."""

    def solve(self, p: Params) -> sp.Expr:
        """Exact answer. sympy only, no floats. This is ground truth."""

    def render(self, p: Params, path: str) -> None:
        """Draw the figure with measurement labels. Deterministic given p."""

    def prompt(self, p: Params) -> str:
        """Target quantity only. No numbers. No spatial description."""
```

Two rules that matter more than they look:

- **`sample` rejection-samples.** If you clamp an out-of-range value to a boundary you
  will silently generate a pile of degenerate near-identical figures at the clamp point.
  Draw, test validity, redraw.
- **`solve` returns sympy, not float.** The paper's existing answers are things like
  `8 + 4√3`. Floats lose that, and lose the ability to grade a model that answers in
  exact form. Convert to float once, at the very end, into `answer_decimal`.

---

## 4. Rendering — the part to isolate

This is the risky module, in the sense the build order meant: it's the one most likely to
need rewriting, so nothing else should depend on its internals.

**Use matplotlib.** Not because it's the prettiest — TikZ/Asymptote output looks more like
the andymath screenshots — but because it is preinstalled in Colab and needs no LaTeX
toolchain. You are moving to Colab; a rendering path that requires `apt-get install
texlive` will cost you more than it returns. Revisit only if figure quality becomes a
reviewer complaint.

Requirements:

1. **Measurements live in the image, never in the prompt.** Side lengths, radii, and angle
   marks are drawn as text annotations on the figure. This is the entire point of Category 1.
2. **Deterministic.** Same params → byte-identical PNG. Set a fixed matplotlib style, fixed
   figure size and DPI, and never call anything that samples randomness at draw time.
3. **Shaded regions get real fills**, not hatching or outlines — the paper's error taxonomy
   has a whole category for shaded-region misreads, so the shading must be unambiguous to a
   careful human reader even when it is hard for a model.
4. **No axes, no gridlines, no title.** A coordinate frame is itself information; only draw
   one for families explicitly tagged `coordinate`.
5. **Fixed canvas, variable content.** Keep the output size constant across families so
   image dimensions carry no signal about problem type.

---

## 5. Verification — replaces the harvest plan's manual review

The harvest plan spends Step 7 on a human spot-checking thirty records because there is no
other way to know if the parser is right. Generation can do better, automatically, on
every record. Three independent checks, all cheap:

1. **Numeric cross-check of the answer.** `solve()` gives a symbolic result. Independently
   estimate the same quantity a second way — Monte Carlo point-sampling for areas, direct
   coordinate computation for lengths and angles — and assert agreement to tolerance. This
   catches algebra errors in `solve()`, which is where real bugs will live.
2. **Render-back check.** Recompute the key measurements from the rendered geometry
   (the actual coordinates matplotlib drew) and assert they match `params`. This catches
   the figure and the answer drifting apart — e.g. `solve` assumes a radius the drawing
   code scaled.
3. **Prompt-leak check.** Already in `Datapoint.validate()`.

A record that fails any of these is a **generator bug**, not a data problem. It should
raise, not be written to a `needs_review.jsonl`. That file exists in the harvest pipeline
because the outside world is messy; here, a failure means your own code is wrong and
silently quarantining it will hide the bug.

---

## 6. Trap tagging — the reason to generate rather than harvest

*(This is my proposal, not something the paper currently does. Flagging that clearly.)*

The paper's five error categories are used descriptively — they classify mistakes after the
fact. A generator can use them **prospectively**: deliberately construct figures that probe
a specific failure mode, and tag each record with which trap it sets.

- `near_symmetric` — figure is almost, but provably not, symmetric. Probes oversimplification
  (the dominant error at 30–37%).
- `near_tangent` — circles that appear tangent but are not, or vice versa. Probes the
  circle/tangency category (~23–25% across all three models).
- `unlabeled_inferrable` — a length that must be derived, adjacent to one that is given,
  inviting the model to assume they're equal. Probes measurement misassignment.
- `nested_overlap` — regions whose overlap is genuinely ambiguous at a glance. Probes
  shaded-region errors.
- `offset_origin` — coordinate layouts where the natural origin is not the obvious corner.
  Probes coordinate misassignment.

Every family declares which traps it can set; `sample()` takes an optional trap request.
This gives something harvesting structurally cannot: **a controlled variable**. You can
generate matched pairs — same family, same difficulty, trap on vs. trap off — and measure
the accuracy delta attributable to that single perceptual feature. That is a much sharper
result than "models get 65.8% on Category 1," and it directly extends the paper's RQ2.

It also answers the contamination limitation in §7.1 of the paper: generated problems
cannot be in any model's training data.

---

## 7. Dedupe

Parameter-tuple hashing. Two generated problems are duplicates iff their `(family, params)`
canonicalize to the same value. Round floats to a fixed precision before hashing so
`3.0000001` and `3.0` collide.

Do not import `imagehash`. Perceptual hashing exists in the harvest plan because there the
params are unknown; here they're the primary key. The only place image hashing is needed is
if you later want to check generated problems against the *existing 200 screenshots* — a
different task, and one blocked on the provenance question anyway.

---

## 8. Entry points

Colab-first, CLI second:

```python
# notebook usage — the primary path
from generator import generate
records = generate(family="circle_tangent_square", n=50, seed=0, out="data/")
```

```bash
# CLI wrapper over the same function
python -m generator --family all --n 200 --seed 0 --out data/ --manifest manifest.jsonl
```

Writes `manifest.jsonl` — one `Datapoint` JSON per line, same format the harvest pipeline
emits, so downstream evaluation code doesn't care which produced a record.

Keep all paths relative and all state in the working directory. No `~/`, no absolute paths,
nothing that assumes a machine.

---

## 9. Build order

1. `schema.py` — `Datapoint` + `validate()`. Everything touches it; changing it late is the
   expensive mistake.
2. **One family, end to end** — pick `circle_tangent_square` or similar: a circle inscribed
   against two square edges. Small param space, exact answer, exercises the taxonomy's
   biggest category (circle/tangency). Get `sample`/`solve`/`render`/`prompt` working and
   eyeball ten outputs.
3. `verify.py` — the three checks in §5, run against that one family. Do this *before* the
   second family, so family #2 is written against a working harness.
4. **Families 2–N.** Now mechanical. Target coverage across the five `category_tags` from
   the paper's §3.3: composite, circular, shaded, nested, coordinate.
5. `generate.py` — batch driver, dedupe, manifest writing.
6. **Trap variants** (§6), once the base families are stable.

Steps 1–3 are the real work. Step 4 is repetition, and it's where a second person can help
without needing to understand the harness.

---

## 10. Open questions

1. **Where is the answer key for the existing 200?** Not in the repo — `model_outputs/*.txt`
   contain model solutions only, no ground truth, no grading. Whatever was used to score the
   3×5 runs isn't version-controlled. This blocks reproducing the paper's numbers regardless
   of what the generator does, and it's the highest-priority question for the lab.
2. **Is the generated set meant to extend DiagramBench, or form a separate held-out set?**
   Changes whether generated problems need to match the existing distribution or deliberately
   differ from it.
3. **Target N?** Drives how many families are needed. 200 problems across 5 families at 40
   each is a very different job from 2,000 across 25.
4. **Does a human review generated diagrams before release?** Automated verification proves
   the answer is right; it does not prove the figure is *legible*. A small human pass on
   rendering quality is still worth doing once, at the start, to calibrate the renderer.
