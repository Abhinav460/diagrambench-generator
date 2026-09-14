# DiagramBench generator — build spec

Target: a procedural generator for Category 1 (diagram-dependent) geometry problems,
supplementing the existing 96 benchmark problems rather than replacing them.

Python 3.10+. Dependencies: `sympy`, `shapely`, `matplotlib`, `numpy`.

---

## Layout

```
generator/
  schema.py        # the datapoint record + validation
  registry.py      # family registration and dispatch
  families/
    nested_polygons.py
    circular_regions.py
    ...
  render.py        # matplotlib emission, shared by all families
  verify.py        # sympy/shapely cross-check
  dedupe.py        # canonical signature + collision detection
  emit.py          # writes images + manifest
  cli.py           # entry point
```

Families are the only thing that changes as configuration types are added.
Everything else is written once.

---

## Build order

### Step 1 — `schema.py`

Define the record before anything produces one. A frozen dataclass with:

- `problem_id`
- `family`
- `seed`
- `params` (dict)
- `stem` (str)
- `image_path`
- `answer_exact` (sympy-serializable string)
- `answer_decimal` (float)
- `signature` (str)
- `generator_version`
- `origin` — `generated` or `harvested`

Add a `validate()` that asserts the invariants that matter: answer is finite and
non-zero, stem is non-empty, params round-trip through JSON.

Write this first because every later module either produces or consumes it, and it
is the thing that is painful to change late.

Note: a frozen dataclass holding a `params` dict is not actually immutable and the
dict is not hashable. Decide now whether `params` is stored as a JSON string or as a
tuple of sorted key/value pairs, because `dedupe.py` needs to hash it.

### Step 2 — the family contract

Every family module exposes the same three functions:

- `sample(rng) -> params` — draw a candidate parameter set
- `is_valid(params) -> bool | reason` — the rejection predicate
- `solve(params) -> (stem, answer_exact, geometry_spec)` — closed-form answer plus a
  description of what to draw

`geometry_spec` is a neutral description (list of primitives with coordinates and
style flags) that `render.py` consumes. Families never import matplotlib. This is
what keeps the labeling policy centralized rather than scattered.

`rng` is a `numpy.random.Generator`, passed in — families never construct their own.

### Step 3 — one family, end to end

Nested polygons.

- Parameters: outer side count `n`, inner side count `m`, a shared side length or an
  apothem ratio.
- Validity: `m < n`, inner actually fits, resulting region non-degenerate.
- Solve: perimeter or area of the annular region, closed form via sympy's exact trig.

Pick this one because the parameter space is small and discrete, so every edge case
surfaces fast.

### Step 4 — `render.py`

Takes `geometry_spec`, emits PNG.

- Axes off, fixed `figsize` and `dpi`, `Agg` backend.
- Explicit `set_xlim` / `set_ylim` rather than autoscaling; no `tight_layout`, which
  reflows based on text extents and can shift output between matplotlib versions.
- The labeling policy lives here: the spec marks each quantity as `draw` or
  `implicit`, and the renderer honors that.
- Burn the stem text into the image as a title, since Appendix A's prompt tells the
  model everything is in the image.

The stem is also stored as a field in the manifest. Generated problems therefore have
both a machine-readable stem and a rendered one, which is what makes a text-leakage
filter possible here even though it is not possible for the existing 96.

### Step 5 — `verify.py`

For each generated problem, rebuild the figure **from `geometry_spec`** in shapely and
compare its numeric area/length against the sympy answer.

This must rebuild from the spec, not from `params`. Rebuilding from params only checks
that the algebra is self-consistent; rebuilding from the spec checks that what is drawn
matches the answer, which is the failure that actually matters.

- Tolerance must be relative, not absolute: `abs(a - b) / max(abs(a), 1) < 1e-9`.
- Shapely has no true circular arcs — circles are polygonal approximations via
  `buffer(r, resolution=N)`. Loosen tolerance accordingly for circle families.
- Disagreement raises rather than warns. A warning in a run of a thousand problems is
  a warning nobody reads.

### Step 6 — `dedupe.py`

Canonical signature: family name plus params normalized to a sorted, rounded tuple,
hashed. Reject collisions at generation time.

Rounding precision is a real parameter — too coarse rejects legitimately distinct
problems, too fine lets near-duplicates through.

### Step 7 — `emit.py` + `cli.py`

Write `manifest.jsonl` (one record per line) alongside `images/`. JSONL rather than a
single JSON array so a crashed run leaves a readable partial file.

CLI takes `--family`, `--n`, `--seed`, `--out`. The seed plus `generator_version` must
be sufficient to reproduce the whole run.

---

## The driver

The automation is a loop in `cli.py`, roughly:

```python
rng = np.random.default_rng(seed)
seen = set()
records = []

while len(records) < n:
    params = family.sample(rng)
    if not family.is_valid(params):
        continue
    sig = signature(family.name, params)
    if sig in seen:
        continue
    stem, answer, spec = family.solve(params)
    image_path = render(spec, out_dir)
    verify(spec, answer)          # raises if they disagree
    seen.add(sig)
    records.append(Datapoint(...))

write_manifest(records, out_dir)
```

Track the acceptance rate. If a family rejects 95% of draws, either the constraints
are too tight or `sample` should draw from a smarter distribution.

---

## Decisions to settle before Step 3

These three should be decided explicitly rather than left to the code:

1. **Answer type per family.** Integer, exact radical, or ratio of expressions. This
   affects how the grader compares later.

2. **Labeling policy.** Which quantities appear in the figure. This is the
   diagram-dependence property, so it deserves a written rule per family rather than
   emerging as a plotting accident.

3. **Whether parameters produce "nice" answers.** Real benchmark problems mostly have
   clean answers. Sampling uniformly yields `12.6`-style values constantly, which reads
   as synthetic. Constraining the sample space to produce clean answers is a real design
   choice with a cost: it shrinks the space and increases near-duplicate risk.

---

## Known caveat

A family is a template, so everything it emits shares a structure. Procedurally
generated problems therefore carry a systematic bias toward being easier — which is
precisely the templating the paper accuses models of exploiting. Generated problems
supplement the existing 96; they do not replace them.

---

## Related: the harvest package (separate, later)

Automates acquisition from the original sources rather than synthesis. Emits the
**same `schema.py` record** with `origin="harvested"`, so everything downstream is
provenance-agnostic.

```
harvest/
  sources/
    andymath.py      # per-site: enumerate pages, locate image + answer
    doingmaths.py
  fetch.py           # requests + on-disk cache, so each page is scraped once
  extract.py         # image bytes, answer string, source URL
  normalize.py       # -> schema.Datapoint
```

Provenance fields to add to the record (optional for generated problems):
`source_site`, `source_url`, `source_problem_id`, `retrieved_at`, `answer_source`,
plus the unprocessed `original_answer` text for auditability.

Blocked on: whether the existing 96 have source URLs recorded anywhere. If not, dedupe
against them requires image-based matching (byte hash, then perceptual hash) rather
than URL set membership.

---

## Prior art worth reading first

- **MAVIS** (arXiv 2407.08739) — rule-based Python + matplotlib geometry data engine.
  Section 2.1 is the closest existing analogue to this generator.
- **TrustGeoGen** (github.com/Alpha-Innovator/TrustGeoGen) — formal verification inside
  the generation loop; the precedent for Step 5.
- **GeomVerse** (Kazemi et al.) — programmatic generation with controlled reasoning depth.
- **NuminaMath / aimo-progress-prize** — harvest pipeline stages and a concrete
  decontamination recipe.
- **PolyMath** (HF: AIMO-Corpus/PolyMath) — a working provenance schema.

None of these enforce diagram-dependence — they put the full configuration in the
problem text. The `draw` vs `implicit` labeling policy is the part that is not covered
by existing work.
