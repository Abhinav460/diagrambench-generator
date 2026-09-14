# Implementation plan — DiagramBench generator

Read `GENERATOR_SPEC.md` in full before writing any code. It is the source of truth
for the module layout, the family contract, and the design decisions. This file is the
execution order and the acceptance criteria.

Python 3.10+. Create a `venv` and a `requirements.txt` with pinned versions
(`sympy`, `shapely`, `matplotlib`, `numpy`, `pytest`). The repo currently has none.

---

## Ground rules

- **Stop at each checkpoint.** Do not proceed past a checkpoint until I have reviewed.
  There are four. Each is a natural place for the design to be wrong, and finding that
  out after six modules are written is expensive.
- **No LLM calls anywhere in this pipeline.** Ground truth is derived symbolically.
  This is a hard constraint, not a preference — the benchmark this feeds measures MLLM
  diagram-reading failure, so an MLLM cannot be in the ground-truth path.
- **Ask rather than assume** on the three decisions listed at the end of the spec
  (answer type, labeling policy, nice answers). If a decision is unmade when you reach
  code that depends on it, stop and ask.
- **Write tests as you go**, in `tests/`, runnable with `pytest`. Not at the end.
- Type hints throughout. Docstrings on the public functions of each module explaining
  what it does and why it exists, not restating the signature.

---

## Stage 1 — `schema.py`

Build the record and its validation.

Resolve first, and ask me if unsure: how `params` is stored so the record is both
JSON-serializable and hashable. Two candidates are a JSON string or a tuple of sorted
key/value pairs. Pick one and note the reasoning in a docstring.

**Acceptance:**
- A `Datapoint` round-trips to JSON and back, unchanged.
- `validate()` rejects: non-finite answer, zero answer, empty stem, params that fail to
  serialize.
- Tests cover each rejection case individually.

### CHECKPOINT 1 — stop here.

---

## Stage 2 — `registry.py` and the family contract

Define the contract as a `Protocol` or ABC so families are checked structurally rather
than by convention. Registry maps family name to module.

Write a stub family that returns fixed values, purely to prove dispatch works. Delete
it once a real family exists.

**Acceptance:**
- `registry.get("stub")` returns something satisfying the contract.
- A module missing one of the three functions fails registration loudly, with a message
  naming which function is missing.

---

## Stage 3 — `families/nested_polygons.py`

The first and only family for now.

- `sample(rng)`: outer side count `n`, inner side count `m`, shared side length or
  apothem ratio.
- `is_valid(params)`: `m < n`, inner fits inside outer, region non-degenerate. Return a
  reason string on rejection, not just `False`, so acceptance rate is debuggable.
- `solve(params)`: exact closed form via sympy. Returns `(stem, answer_exact, geometry_spec)`.

Define the `geometry_spec` format here, since this is the first consumer. It should be
a plain list of primitives — points, segments, polygons, arcs — each with coordinates
and a `draw` / `implicit` flag on any associated quantity. Document the format in a
module docstring; `render.py` and `verify.py` both depend on it.

No matplotlib import in this file. Enforce that with a test if convenient.

**Acceptance:**
- 1000 sampled params: report the acceptance rate and the distribution of rejection
  reasons. If acceptance is under 20%, say so rather than proceeding — the sampler
  probably needs to draw from a constrained space rather than filtering afterward.
- Three hand-worked cases where I can check the closed form by hand: `(n=6, m=3)`,
  `(n=8, m=4)`, `(n=9, m=6)` with unit side length.

### CHECKPOINT 2 — stop here. This is where the design either holds or doesn't.

---

## Stage 4 — `render.py`

Consumes `geometry_spec`, emits PNG.

- `Agg` backend, axes off, fixed `figsize` and `dpi`.
- Explicit `set_xlim` / `set_ylim`. No `tight_layout`.
- Honors the `draw` / `implicit` flag: `draw` quantities get a numeric annotation in the
  figure, `implicit` ones get none.
- Stem burned into the image as a title.

**Acceptance:**
- Rendering the same spec twice produces byte-identical PNGs.
- Generate a 3x3 contact sheet of nine varied nested-polygon problems so I can eyeball
  whether they look like plausible benchmark figures rather than obviously synthetic ones.

---

## Stage 5 — `verify.py`

Rebuild the figure from `geometry_spec` in shapely, measure numerically, compare against
the sympy answer.

**This must consume `geometry_spec`, not `params`.** Rebuilding from params only proves
the solver agrees with itself. The failure mode that matters is a figure that shows
something other than what the answer describes, and only a spec-based rebuild catches it.

- Relative tolerance: `abs(a - b) / max(abs(a), 1) < 1e-9`.
- Circles have no exact shapely representation — approximate with
  `buffer(r, resolution=N)` and loosen tolerance for arc-containing specs. Document the
  chosen `N` and the resulting tolerance.
- Disagreement raises `VerificationError` with the params, both values, and the relative
  difference in the message.

**Acceptance:**
- Passes on all currently generating problems.
- A deliberately corrupted spec (perturb one coordinate) raises. Include this as a test —
  a verifier that never fires is indistinguishable from no verifier.

### CHECKPOINT 3 — stop here.

---

## Stage 6 — `dedupe.py`

Canonical signature: family name plus params normalized to a sorted, rounded tuple,
hashed with `hashlib`.

Rounding precision is a parameter, not a constant buried in the function. Expose it and
document what happens at either extreme.

**Acceptance:**
- Identical params collide; params differing beyond the rounding threshold do not.
- Report the collision rate over a run of 500.

---

## Stage 7 — `emit.py` and `cli.py`

`emit.py` writes `manifest.jsonl` (one record per line) and `images/`.

`cli.py` implements the driver loop from the spec. Flags: `--family`, `--n`, `--seed`,
`--out`. Also `--report` to print acceptance rate, collision rate, and verification
count at the end of a run.

**Acceptance:**
- Same seed and `generator_version` produce a byte-identical manifest across two runs.
- A run killed partway leaves a valid, readable partial `manifest.jsonl`.
- `--n 200 --seed 0` completes and reports its stats.

### CHECKPOINT 4 — stop here. Final review.

---

## Explicitly out of scope

Do not build these now, even if they seem natural:

- Additional families beyond nested polygons.
- Anything in `harvest/`. Separate package, separate session, blocked on an open question
  about the existing dataset.
- A grader or answer-comparison module.
- Any error-taxonomy labeling.

---

## What to hand back at the end

- The package, tests passing.
- `requirements.txt` with pins.
- A short `README` section covering how to run a generation, what the flags do, and what
  the reported stats mean.
- A list of anything you had to decide that the spec left ambiguous, and what you chose.
