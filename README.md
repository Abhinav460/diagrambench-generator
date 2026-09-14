# DiagramBench generator

Procedurally generates Category 1 (diagram-dependent) geometry problems: a figure,
a question burned into the image, and an exact symbolic answer.

Four families cover all five geometric configurations the paper's section 3.3
requires Category 1 to span:

| Family | Configuration | What the figure withholds |
|---|---|---|
| `nested_polygons` | nested polygons, shaded areas | both side counts |
| `inscribed_circle` | circular regions | side count, and the radius (inferable only from the tangency) |
| `composite_rectilinear` | composite figures | two edge lengths, each a difference of two labelled edges |
| `coordinate_polygon` | coordinate-based layouts | the coordinates of some vertices, readable only off the grid |

**No LLM is involved at any point.** Ground truth is derived symbolically with sympy
and independently re-measured with shapely. This is a hard constraint of the design,
not a current limitation — the benchmark this feeds measures MLLM diagram-reading
failure, so an MLLM cannot appear anywhere in the ground-truth path.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.10+. Versions in `requirements.txt` are pinned to what resolved on the
build machine and have **not** been checked against Colab's preinstalled stack.

## Run a generation

```bash
# one family
.venv/bin/python -m generator.cli --family nested_polygons --n 200 --seed 0 \
    --out runs/dev --report

# spread across every family -- the usual case for building a dataset
.venv/bin/python -m generator.cli --family all --n 200 --seed 0 \
    --out runs/dev --report
```

This writes:

```
runs/dev/
  manifest.jsonl        one JSON record per line, in generation order
  images/
    nested_polygons_00000.png
    ...
```

`manifest.jsonl` is JSONL rather than a JSON array so that a run killed partway
still leaves a readable file. Lines are written and flushed one at a time, so
whatever exists on disk is complete records.

Read a run back with:

```python
from generator.emit import read_manifest
records = list(read_manifest("runs/dev"))            # strict
records = list(read_manifest("runs/dev", strict=False))  # tolerate a killed run
```

### Flags

| Flag | Default | What it does |
|---|---|---|
| `--family` | `nested_polygons` | Which family to draw from, or `all` to spread the run across every registered family. `--list-families` shows what is registered. |
| `--n` | `10` | How many problems to emit. The run stops early rather than looping forever if the space is exhausted. |
| `--seed` | `0` | RNG seed. Seed + `generator_version` reproduces a run byte for byte. |
| `--out` | required | Output directory. Refuses to overwrite an existing manifest. |
| `--report` | off | Print the run statistics at the end. |
| `--overwrite` | off | Replace an existing manifest in `--out`. |
| `--precision` | `6` | Decimal places kept when hashing params for dedupe. See "Dedupe precision" below. |
| `--max-draws` | `200 × --n` | Draw ceiling before giving up. |
| `--progress-every` | `0` | Print progress to stderr every N problems. `0` disables. |
| `--list-families` | — | List registered families and exit. |

Exit codes: `0` reached `--n`; `1` a real failure (verification disagreement, or an
unwritable output directory); `2` ran out of draws short of `--n`. A short run is
separated from a failed one because it is a normal outcome for a small family, and
a script running a sweep needs to tell them apart.

### Reading the report

```
family                : composite_rectilinear+coordinate_polygon+inscribed_circle+nested_polygons
seed                  : 0
emitted               : 200/200
draws                 : 404
acceptance rate       : 57.9% (234 valid, 170 rejected)
collision rate        : 14.5% (34 duplicates of 234 valid)
verified              : 200 (worst relative difference 7.16e-06)
emitted by family  :
        38  composite_rectilinear
        47  coordinate_polygon
        68  inscribed_circle
        47  nested_polygons
rejections by cause   :
        44  [nested_polygons] inner polygon does not fit with clearance
        39  [coordinate_polygon] vertices are collinear or wind inconsistently
        38  [composite_rectilinear] notch width is of the figure, outside
        ...
```

- **acceptance rate** — fraction of draws that passed `is_valid`. Low means the
  constraints are too tight or `sample` should draw from a smarter distribution
  rather than filtering afterwards. Below ~20% is the point to fix the sampler.
- **collision rate** — fraction of *valid* draws that repeated a problem already
  emitted. Climbing through a run means the parameter space is running out, and
  drawing harder will not help.
- **verified** — how many problems passed the shapely cross-check, and the worst
  relative difference seen. Every emitted problem is verified; the count matching
  `emitted` is the invariant. The worst difference is the headroom against
  tolerance — if it creeps toward `1e-9`, something has changed numerically.
- **rejections by cause** — rejection reasons with the specific numbers stripped so
  they aggregate, prefixed by family in a mixed run. This is what says *which*
  constraint is doing the rejecting.
- **emitted by family** — shown only for a mixed run. The split is uneven by
  design: round-robin draws are skipped when a family rejects, so a family with a
  tighter validity predicate emits fewer. `inscribed_circle` leads because nothing
  in its sampled range is ever rejected.

## How it fits together

```
cli.py       driver loop: draw → validate → dedupe → solve → verify → render → emit
  registry.py    finds family modules and enforces the contract structurally
  families/      the only thing that changes as new problem types are added
  geometry.py    geometry_spec → shapely, shared by render and verify
  verify.py      re-measures the figure and compares against the sympy answer
  render.py      matplotlib → PNG
  dedupe.py      canonical signature, collision detection
  emit.py        manifest.jsonl + images/
  schema.py      the record every stage produces or consumes
```

Two design points worth knowing before changing anything:

**`render` and `verify` resolve a region through the same module.** `geometry.py`
turns a spec into shapely geometry, and both consume it. They ask different
questions — one fills the result, the other measures it — but if they disagreed
about what a spec *meant*, verification would be checking a figure other than the
one drawn. Sharing the resolver makes that structurally impossible rather than
merely tested for.

**`verify.py` consumes `geometry_spec`, never `params`.** Rebuilding from params
would re-run the family's own derivation and agree with it by construction. The
failure that matters is a figure showing something other than what the answer
describes, and only a spec-based rebuild catches it. Verification runs *before*
rendering, so a disagreeing problem leaves no PNG behind.

**Dedupe precision.** Params are rounded before hashing. Too fine (`--precision 15`)
lets float noise through and admits near-duplicates *silently* — the manifest looks
fine and the benchmark is weaker than its record count claims. Too coarse
(`--precision 0`) merges genuinely distinct problems and collapses yield, which at
least shows up as a rising collision rate. Err low: discarding a real problem costs
one draw, admitting a duplicate costs the benchmark's integrity.

## Categories and the leak check

Records carry `category`: `1` for diagram-dependent, `2` for textual. Only
Category 1 is generated — Category 2 is AIME-style prose, and the paper's role for
it is a controlled baseline of real competition problems, which is not something to
synthesise. The field exists so the record format can express both, since the
paper's headline result is the *gap* between the two.

Category 1 carries an enforced invariant: **the stem may not state geometry.** The
paper's first selection criterion requires the diagram be the exclusive carrier of
geometric information, so `validate()` rejects any Category 1 stem containing a
digit, a number word, or a polygon name. "Find the area between the hexagon and the
triangle" states two side counts in words and is refused. The check is deliberately
crude and errs toward false positives: rewording a stem is cheap, while a missed
leak means a benchmark that quietly measures reading instead of seeing.

## Adding a family

Drop a module in `generator/families/`. It is auto-discovered, and must define
`NAME` plus three functions:

- `sample(rng) -> params` — draw from the parameter space using the passed
  `Generator` only. Constructing its own randomness breaks reproducibility, and a
  test enforces that.
- `is_valid(params) -> True | str` — return `True`, or a **reason string**. The
  string is what makes a low acceptance rate diagnosable instead of guesswork.
- `solve(params) -> (stem, answer_exact, geometry_spec)` — exact sympy answer, and
  the primitives to draw. No matplotlib import; a test enforces that too.

A module missing any of these fails registration loudly, naming what is missing.
The `geometry_spec` format is documented in `generator/families/nested_polygons.py`;
`render.py` and `verify.py` both depend on it.

## Tests

```bash
.venv/bin/python -m pytest              # everything
.venv/bin/python -m pytest -m "not slow"  # skips the 200-problem run
```
