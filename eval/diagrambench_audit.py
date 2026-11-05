#!/usr/bin/env python3
"""
DiagramBench: evaluate models AND audit the datasets, in one script.

What it does
  list-models  List models your API keys can reach (Anthropic / OpenAI / Google).
  eval         Feed figure + text (+ description) to a model and record its solution.
                 96  set: conditions image_desc (primary), desc_only, image_only
                 104 set: conditions image_text (primary), text_only
  audit        Ask a verifier model to check, per item:
                 - does the FIGURE match the PROBLEM?
                 - does the DESCRIPTION match the FIGURE? (96 set)
                 - does a generated diagram LEAK the conclusion? (104 set)
                 - independent solution + a python check-script that is actually RUN
  report       Merge verifiers, flag items for human review, write candidate answers,
               and summarise eval accuracy (only where a reference answer exists).

Honest limits
  * Neither dataset ships reference answers (96 ground truth is all null). Answers
    produced here are CANDIDATES, never ground truth. Accuracy is reported only for
    items that have a reference (--answers file, or --provisional to use candidates,
    which is labelled provisional).
  * Use two verifiers from DIFFERENT providers; items where they disagree go to humans.
  * The check-script runner blocks obvious dangerous code and times out, but it is NOT
    a real sandbox. Run this inside a container/VM you are fine with.

Setup
  git checkout main && git pull && git checkout -b audit/figure-problem-check
  export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GOOGLE_API_KEY=...   (whichever you have)
  python3 diagrambench_audit.py list-models

Typical run (python3 diagrambench_audit.py --help for all flags)
  # smoke test, no API calls, no keys:
  python3 diagrambench_audit.py audit --dataset 96 --data PATH_TO_96_ZIP --verifier mock:x --limit 3
  # real audit with two verifiers from different providers:
  python3 diagrambench_audit.py audit --dataset 96  --data PATH_96  --verifier anthropic:<model>
  python3 diagrambench_audit.py audit --dataset 96  --data PATH_96  --verifier openai:<model>
  python3 diagrambench_audit.py audit --dataset 104 --data PATH_104 --verifier anthropic:<model>
  python3 diagrambench_audit.py audit --dataset 104 --data PATH_104 --verifier openai:<model>
  python3 diagrambench_audit.py report --dataset 96
  python3 diagrambench_audit.py report --dataset 104
  # evaluate models:
  python3 diagrambench_audit.py eval --dataset 96  --data PATH_96  --condition image_desc --model google:<model>
  python3 diagrambench_audit.py eval --dataset 104 --data PATH_104 --condition image_text --model google:<model>
Only stdlib is required; sympy is used for answer comparison / check scripts if installed.
"""
import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RESULTS = Path("results")
LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _resolve_root(data, marker):
    p = Path(data).expanduser()
    if p.is_file() and p.suffix.lower() == ".zip":
        out = Path("data") / p.stem
        if not out.exists():
            out.mkdir(parents=True)
            with zipfile.ZipFile(p) as z:
                z.extractall(out)
        p = out
    hits = sorted(p.rglob(marker))
    if not hits:
        sys.exit(f"ERROR: {marker} not found under {p}")
    return hits[0].parent


def load_96(data):
    root = _resolve_root(data, "diagram_96_eval.jsonl")
    with open(root / "diagram_96_eval.jsonl", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    refs = {}
    gp = root / "diagram_96_ground_truth.jsonl"
    if gp.exists():
        with open(gp, encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    r = json.loads(l)
                    if r.get("answer") not in (None, ""):
                        refs[r["problem_id"]] = str(r["answer"])
    items = []
    for r in recs:
        img = root / r["image_path"]
        if not img.exists():
            print(f"WARNING: missing image {img}", file=sys.stderr)
        items.append(dict(id=r["problem_id"], dataset="96", text=r["original_problem_text"],
                          description=r.get("full_figure_description"),
                          image=str(img) if img.exists() else None, meta={}))
    return items, refs


def load_104(data):
    root = _resolve_root(data, "DiagramBench_104_manifest.json")
    with open(root / "DiagramBench_104_manifest.json", encoding="utf-8") as f:
        probs = json.load(f)["problems"]
    items, bad = [], []
    for p in probs:
        img = None
        if p.get("image_path"):
            ip = root / p["image_path"]
            if not ip.exists():
                bad.append(f"missing {p['problem_id']}")
            else:
                h = hashlib.sha256(ip.read_bytes()).hexdigest()
                if p.get("image_sha256") and h != p["image_sha256"]:
                    bad.append(f"sha mismatch {p['problem_id']}")
                img = str(ip)
        goal = (p.get("textual_record") or {}).get("goal")
        items.append(dict(id=p["problem_id"], dataset="104", text=p["problem_text"], description=None,
                          image=img, meta=dict(diagram_type=p.get("diagram_type"),
                                               diagram_quality=p.get("diagram_quality"),
                                               diagram_need=p.get("diagram_need"),
                                               problem_number=p.get("problem_number"),
                                               goal=goal)))
    if bad:
        print("WARNING: package problems:", bad[:10], file=sys.stderr)
    return items, {}


def load_items(dataset, data):
    return load_96(data) if dataset == "96" else load_104(data)


def select(items, args):
    if getattr(args, "ids", None):
        want = set(args.ids.split(","))
        items = [i for i in items if i["id"] in want]
    if getattr(args, "limit", None):
        items = items[: args.limit]
    return items


# ----------------------------------------------------------------------------
# Providers (stdlib HTTP, no SDKs). Keys come from env; never printed.
# ----------------------------------------------------------------------------
def _http(url, payload=None, headers=None, method="POST", timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    delay = 4
    for attempt in range(6):
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529) and attempt < 5:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"HTTP {e.code}: {body}")
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < 5:
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise
    raise RuntimeError("unreachable")


def _key(name):
    k = os.environ.get(name)
    if not k:
        sys.exit(f"ERROR: environment variable {name} is not set")
    return k


def _img_b64(path):
    mt = mimetypes.guess_type(path)[0] or "image/png"
    return mt, base64.b64encode(Path(path).read_bytes()).decode()


def complete(spec, system, text, image=None, temperature=None):
    provider, _, model = spec.partition(":")
    if provider == "mock":
        return _mock(system, text)
    if provider == "anthropic":
        content = []
        if image:
            mt, b = _img_b64(image)
            content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": b}})
        content.append({"type": "text", "text": text})
        payload = {"model": model, "max_tokens": 8192, "system": system,
                   "messages": [{"role": "user", "content": content}]}
        if temperature is not None:
            payload["temperature"] = temperature
        r = _http("https://api.anthropic.com/v1/messages", payload,
                  {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in r.get("content", []))
    if provider == "openai":
        content = [{"type": "text", "text": text}]
        if image:
            mt, b = _img_b64(image)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64,{b}"}})
        payload = {"model": model, "max_completion_tokens": 16000,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": content}]}
        if temperature is not None:
            payload["temperature"] = temperature
        r = _http("https://api.openai.com/v1/chat/completions", payload,
                  {"Authorization": "Bearer " + _key("OPENAI_API_KEY")})
        return r["choices"][0]["message"].get("content") or ""
    if provider == "google":
        parts = [{"text": text}]
        if image:
            mt, b = _img_b64(image)
            parts.append({"inlineData": {"mimeType": mt, "data": b}})
        payload = {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": parts}],
                   "generationConfig": {"maxOutputTokens": 16000}}
        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
        r = _http(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                  payload, {"x-goog-api-key": _key("GOOGLE_API_KEY")})
        cands = r.get("candidates") or []
        if not cands:
            raise RuntimeError("no candidates: " + json.dumps(r)[:300])
        return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", []))
    sys.exit(f"ERROR: unknown provider '{provider}' (use anthropic|openai|google|mock)")


def _mock(system, text):
    if "STRICT JSON" in system:
        return json.dumps({
            "figure_matches_problem": {"verdict": "match", "reason": "mock"},
            "description_matches_figure": {"verdict": "match", "reason": "mock"},
            "leaks_conclusion": {"value": False, "reason": "mock"},
            "problem_type": "computation", "final_answer": "42", "confidence": 0.9,
            "check_script": "import math\nprint('RESULT: 42')", "notes": "mock"})
    return "Mock reasoning.\nFINAL ANSWER: 42"


def list_models():
    def show(name, fn):
        try:
            print(f"\n== {name} ==")
            for m in fn():
                print("  ", m)
        except SystemExit:
            print("   (no key set, skipped)")
        except Exception as e:
            print("   error:", str(e)[:200])

    def a():
        r = _http("https://api.anthropic.com/v1/models?limit=100", None,
                  {"x-api-key": _key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"}, "GET")
        return sorted(m["id"] for m in r.get("data", []))

    def o():
        r = _http("https://api.openai.com/v1/models", None,
                  {"Authorization": "Bearer " + _key("OPENAI_API_KEY")}, "GET")
        return sorted(m["id"] for m in r.get("data", []))

    def g():
        r = _http("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200", None,
                  {"x-goog-api-key": _key("GOOGLE_API_KEY")}, "GET")
        return sorted(m["name"].split("/")[-1] for m in r.get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))

    show("anthropic", a)
    show("openai", o)
    show("google", g)
    print("\nPick vision-capable, strongest-reasoning models. For auditing use TWO different providers.")
    print("The API lists do not say which models accept images; check each provider's docs.")


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
EVAL_SYSTEM = (
    "You are an expert geometry solver. Use ONLY the information given (figure, description, text). "
    "Reason step by step. For computational problems end with a final line exactly "
    "'FINAL ANSWER: <answer>'. For proof problems write the proof and end with 'FINAL ANSWER: PROOF'. "
    "If the figure and the description/text disagree, add a line starting 'DISCREPANCY:' explaining."
)

AUDIT_SYSTEM = (
    "You are a meticulous geometry-benchmark auditor. Look at the image (if provided) and the text. "
    "Reply with STRICT JSON only (no markdown fences, no prose outside JSON) using exactly these keys:\n"
    '{"figure_matches_problem":{"verdict":"match|partial|mismatch|n_a","reason":"cite specific labels/values"},'
    '"description_matches_figure":{"verdict":"match|partial|mismatch|n_a","reason":"list missing/invented/wrong facts; n_a if no description"},'
    '"leaks_conclusion":{"value":true|false,"reason":"does the figure reveal what must be proved/found?"},'
    '"problem_type":"computation|proof|other",'
    '"final_answer":"your independently derived final answer (computation) or PROOF",'
    '"confidence":0.0-1.0,'
    '"check_script":"python3 code using only math/sympy/numpy/fractions/random/itertools. '
    "computation: compute the answer from the givens and print 'RESULT: <value>'. "
    "proof: build ~20 random valid configurations satisfying the hypotheses, test the claimed conclusion "
    "numerically, print 'RESULT: HOLDS' or 'RESULT: FAILS'. Empty string if impossible.\","
    '"notes":"ambiguity, underdetermined data, typos"}\n'
    "Judge the figure only by what is drawn. Do not assume the figure is correct."
)


def eval_text(item, cond):
    parts = ["Problem:\n" + item["text"]]
    if cond in ("image_desc", "desc_only") and item.get("description"):
        parts.append("Figure description:\n" + item["description"])
    return "\n\n".join(parts)


def eval_image(item, cond):
    return item["image"] if cond in ("image_desc", "image_only", "image_text") else None


def audit_text(item):
    parts = ["Problem:\n" + item["text"]]
    if item.get("description"):
        parts.append("Figure description to audit:\n" + item["description"])
    else:
        parts.append("(No separate description. Audit the image against the problem.)")
    if not item.get("image"):
        parts.append("(No image available for this item.)")
    return "\n\n".join(parts)


CONDITIONS = {"96": ["image_desc", "desc_only", "image_only"], "104": ["image_text", "text_only"]}


# ----------------------------------------------------------------------------
# Helpers: JSON extraction, answer comparison, script runner
# ----------------------------------------------------------------------------
def extract_json(raw):
    if not raw:
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except Exception:
        return None


def parse_final(raw):
    m = re.findall(r"FINAL ANSWER:\s*(.+)", raw or "", flags=re.I)
    return m[-1].strip().strip("$* ") if m else None


def _norm(s):
    s = str(s).strip().lower()
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "").replace("\\,", "")
    s = re.sub(r"\\(?:text|mathrm)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("^\\circ", "").replace("°", "").replace("degrees", "")
    return re.sub(r"\s+", "", s).rstrip(".")


def _to_num(s):
    s = _norm(s)
    s = re.sub(r"^(-?[\d.]+)(?:cm|mm|m|km|units?|sq|degrees?|deg|in|ft)(?:\^?2|\^?3)?$", r"\1", s)
    try:
        return float(s)
    except Exception:
        pass
    m = re.fullmatch(r"(-?[\d.]+)/(-?[\d.]+)", s) or re.fullmatch(r"\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    try:
        import sympy
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                                implicit_multiplication_application, convert_xor)
        t = s
        for _ in range(3):
            t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", t)
            t = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", t)
        t = t.replace("\\pi", "pi").replace("\\cdot", "*").replace("{", "(").replace("}", ")")
        e = parse_expr(t, transformations=standard_transformations + (implicit_multiplication_application, convert_xor))
        return float(e.evalf())
    except Exception:
        return None


def same(a, b):
    if a is None or b is None:
        return False
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    x, y = _to_num(a), _to_num(b)
    if x is not None and y is not None:
        return abs(x - y) <= 1e-6 * max(1.0, abs(x), abs(y))
    return False


ALLOWED_IMPORTS = {"math", "sympy", "numpy", "fractions", "random", "itertools", "cmath", "statistics"}
BLOCKED = re.compile(r"\b(os|sys|subprocess|socket|shutil|pathlib|requests|urllib|ctypes|importlib|"
                     r"open|exec|eval|compile|__import__|globals|locals|input)\s*[\.\(]|__\w+__")


def run_check(script, timeout=25):
    if not script or not script.strip():
        return dict(status="no_script", result=None)
    for m in re.finditer(r"^\s*(from\s+([\w\.]+)\s+import\b.*|import\s+(.+))$", script, flags=re.M):
        if m.group(2):
            mods = [m.group(2).split(".")[0]]
        else:
            mods = [x.strip().split()[0].split(".")[0] for x in m.group(3).split(",") if x.strip()]
        for mod in mods:
            if mod not in ALLOWED_IMPORTS:
                return dict(status="blocked", result=f"import {mod}")
    if BLOCKED.search(script):
        return dict(status="blocked", result="disallowed construct")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "check.py"
        f.write_text(script, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", str(f)], capture_output=True, text=True,
                               timeout=timeout, cwd=d)
        except subprocess.TimeoutExpired:
            return dict(status="timeout", result=None)
    m = re.findall(r"RESULT:\s*(.+)", p.stdout)
    if m:
        return dict(status="ok", result=m[-1].strip())
    return dict(status="error", result=(p.stderr or p.stdout)[-300:])


# ----------------------------------------------------------------------------
# Resumable job runner
# ----------------------------------------------------------------------------
def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def run_jobs(items, fn, outfile, workers):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outfile.exists():
        for l in outfile.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(l)
                if not r.get("error"):
                    done.add(r["problem_id"])
            except Exception:
                pass
    todo = [i for i in items if i["id"] not in done]
    print(f"{len(items)} items, {len(done)} already done, {len(todo)} to run -> {outfile}")
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): it for it in todo}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = dict(problem_id=it["id"], error=str(e)[:300])
            with LOCK:
                with open(outfile, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            print(f"  [{n}/{len(todo)}] {it['id']}" + ("  ERROR" if row.get("error") else ""), flush=True)


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def cmd_eval(args):
    if args.condition not in CONDITIONS[args.dataset]:
        sys.exit(f"condition must be one of {CONDITIONS[args.dataset]} for dataset {args.dataset}")
    items, _ = load_items(args.dataset, args.data)
    items = select(items, args)
    if args.condition == "image_text":
        items = [i for i in items if i["image"]]  # skip 3 text_only rows
    if args.condition in ("image_desc", "image_only"):
        items = [i for i in items if i["image"]]
    if args.dry_run:
        print(f"Would make {len(items)} calls to {args.model}")
        for it in items[:2]:
            print("-" * 60, "\nIMAGE:", eval_image(it, args.condition), "\n", eval_text(it, args.condition)[:700])
        return

    def fn(it):
        t0 = time.time()
        raw = complete(args.model, EVAL_SYSTEM, eval_text(it, args.condition),
                       eval_image(it, args.condition), args.temperature)
        return dict(problem_id=it["id"], dataset=args.dataset, condition=args.condition, model=args.model,
                    raw_response=raw, final_answer=parse_final(raw),
                    discrepancy=bool(re.search(r"^\s*DISCREPANCY:", raw or "", flags=re.M | re.I)),
                    latency_s=round(time.time() - t0, 2), **it["meta"])
    run_jobs(items, fn, RESULTS / f"eval_{args.dataset}_{args.condition}_{safe(args.model)}.jsonl", args.workers)


def cmd_audit(args):
    items, _ = load_items(args.dataset, args.data)
    items = select(items, args)
    if args.dry_run:
        print(f"Would make {len(items)} calls to {args.verifier}")
        for it in items[:2]:
            print("-" * 60, "\nIMAGE:", it["image"], "\n", audit_text(it)[:700])
        return

    def fn(it):
        raw, parsed = None, None
        for _ in range(2):
            raw = complete(args.verifier, AUDIT_SYSTEM, audit_text(it), it["image"], args.temperature)
            parsed = extract_json(raw)
            if parsed:
                break
        row = dict(problem_id=it["id"], dataset=args.dataset, verifier=args.verifier, **it["meta"])
        if not parsed:
            row.update(error="json_parse_failed", raw_response=(raw or "")[:2000])
            return row
        row["verdict"] = parsed
        row["script"] = run_check(parsed.get("check_script"))
        return row
    run_jobs(items, fn, RESULTS / f"audit_{args.dataset}_{safe(args.verifier)}.jsonl", args.workers)


def _v(row, k):
    return ((row.get("verdict") or {}).get(k) or {})


def flags_for(rows):
    """rows: {verifier: audit_row}. Returns list of flag strings."""
    fl = []
    answers = []
    for v, r in rows.items():
        d = r.get("verdict") or {}
        if r.get("error"):
            fl.append(f"{v}:audit_error")
            continue
        fm = _v(r, "figure_matches_problem").get("verdict")
        dm = _v(r, "description_matches_figure").get("verdict")
        if fm in ("partial", "mismatch"):
            fl.append(f"{v}:figure_{fm}")
        if dm in ("partial", "mismatch"):
            fl.append(f"{v}:description_{dm}")
        if (_v(r, "leaks_conclusion")).get("value") is True:
            fl.append(f"{v}:leaks_conclusion")
        try:
            if float(d.get("confidence", 0)) < 0.6:
                fl.append(f"{v}:low_confidence")
        except Exception:
            fl.append(f"{v}:bad_confidence")
        s = r.get("script") or {}
        ptype = d.get("problem_type")
        if ptype == "computation":
            answers.append(d.get("final_answer"))
            if s.get("status") != "ok":
                fl.append(f"{v}:script_{s.get('status')}")
            elif not same(s.get("result"), d.get("final_answer")):
                fl.append(f"{v}:script_disagrees_with_answer")
        elif ptype == "proof":
            if s.get("status") == "ok" and "FAIL" in str(s.get("result")).upper():
                fl.append(f"{v}:proof_numeric_check_fails")
            elif s.get("status") != "ok":
                fl.append(f"{v}:proof_check_{s.get('status')}")
    if len(rows) >= 2:
        fms = {_v(r, "figure_matches_problem").get("verdict") for r in rows.values() if not r.get("error")}
        if len(fms) > 1:
            fl.append("verifiers_disagree_on_figure")
        if len(answers) >= 2 and not all(same(answers[0], a) for a in answers[1:]):
            fl.append("verifiers_disagree_on_answer")
    else:
        fl.append("single_verifier_only")
    return fl


def cmd_report(args):
    items, refs = load_items(args.dataset, args.data) if args.data else ([], {})
    text_by_id = {i["id"]: i["text"] for i in items}
    if args.answers:
        for l in open(args.answers, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("answer") not in (None, ""):
                    refs[r["problem_id"]] = str(r["answer"])
    audits = {}
    for f in sorted(RESULTS.glob(f"audit_{args.dataset}_*.jsonl")):
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                audits.setdefault(r["problem_id"], {})[r.get("verifier", f.stem)] = r
    out = [f"# DiagramBench {args.dataset} audit report\n",
           "Answers below are **model-derived candidates, not ground truth.**\n"]
    review, cands = [], []
    counts = {}
    for pid, rows in audits.items():
        fl = flags_for(rows)
        for r in rows.values():
            fm = _v(r, "figure_matches_problem").get("verdict")
            if fm:
                counts[fm] = counts.get(fm, 0) + 1
        answers = {v: (r.get("verdict") or {}).get("final_answer") for v, r in rows.items() if not r.get("error")}
        ptypes = {(r.get("verdict") or {}).get("problem_type") for r in rows.values() if not r.get("error")}
        reasons = "; ".join(f"{v}: {_v(r,'figure_matches_problem').get('reason','')[:160]}" for v, r in rows.items())
        if fl and fl != ["single_verifier_only"]:
            review.append(dict(problem_id=pid, flags="|".join(fl), reasons=reasons,
                               candidate_answers=json.dumps(answers, ensure_ascii=False),
                               problem=text_by_id.get(pid, "")[:300], human_verdict="", human_answer=""))
        elif not fl and ptypes == {"computation"}:
            vals = list(answers.values())
            cands.append(dict(problem_id=pid, candidate_answer=vals[0], agreement_level=f"{len(vals)} verifiers + scripts agree",
                              status="unverified"))
        elif fl == ["single_verifier_only"] and ptypes == {"computation"}:
            cands.append(dict(problem_id=pid, candidate_answer=list(answers.values())[0],
                              agreement_level="single verifier (weak)", status="unverified"))
    out.append(f"Items audited: {len(audits)}  | figure verdict counts: {counts}\n")
    out.append(f"## Needs human review ({len(review)})\n")
    for r in review:
        out.append(f"- `{r['problem_id']}`: {r['flags']}")
    out.append(f"\n## Candidate answers ({len(cands)}) - spot-check before promoting to reference\n")
    for c in cands:
        out.append(f"- `{c['problem_id']}`: {c['candidate_answer']}  ({c['agreement_level']})")
    # eval accuracy
    out.append("\n## Model evaluation results\n")
    prov = {}
    if args.provisional:
        prov = {c["problem_id"]: str(c["candidate_answer"]) for c in cands if c["agreement_level"].startswith(("2", "3"))}
    evs = sorted(RESULTS.glob(f"eval_{args.dataset}_*.jsonl"))
    if not evs:
        out.append("No eval files found.")
    for f in evs:
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        ok = [r for r in rows if not r.get("error")]
        graded = corr = pgraded = pcorr = 0
        for r in ok:
            if r["problem_id"] in refs:
                graded += 1
                corr += same(r.get("final_answer"), refs[r["problem_id"]])
            elif r["problem_id"] in prov:
                pgraded += 1
                pcorr += same(r.get("final_answer"), prov[r["problem_id"]])
        acc = f"{corr}/{graded} = {corr/graded:.1%}" if graded else "n/a (no reference answers)"
        line = f"- `{f.stem}`: {len(ok)} answered, {len(rows)-len(ok)} errors, accuracy vs reference: {acc}"
        if pgraded:
            line += f" | PROVISIONAL vs consensus candidates: {pcorr}/{pgraded} = {pcorr/pgraded:.1%}"
        out.append(line)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"audit_{args.dataset}_report.md").write_text("\n".join(out), encoding="utf-8")
    with open(RESULTS / f"to_review_{args.dataset}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["problem_id", "flags", "reasons", "candidate_answers", "problem",
                                          "human_verdict", "human_answer"])
        w.writeheader()
        w.writerows(review)
    with open(RESULTS / f"candidate_answers_{args.dataset}.jsonl", "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("\n".join(out))
    print(f"\nWrote results/audit_{args.dataset}_report.md, to_review_{args.dataset}.csv, "
          f"candidate_answers_{args.dataset}.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list-models")

    def common(p):
        p.add_argument("--dataset", choices=["96", "104"], required=True)
        p.add_argument("--data", required=True, help="path to the dataset .zip or extracted folder")
        p.add_argument("--limit", type=int)
        p.add_argument("--ids", help="comma-separated problem_ids")
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--temperature", type=float, default=None, help="omit to use provider default")
        p.add_argument("--dry-run", action="store_true")

    pe = sub.add_parser("eval")
    common(pe)
    pe.add_argument("--condition", required=True)
    pe.add_argument("--model", required=True, help="provider:model, e.g. anthropic:<model-id>")
    pa = sub.add_parser("audit")
    common(pa)
    pa.add_argument("--verifier", required=True, help="provider:model")
    pr = sub.add_parser("report")
    pr.add_argument("--dataset", choices=["96", "104"], required=True)
    pr.add_argument("--data", help="dataset path (optional; adds problem text + 96 ground-truth refs)")
    pr.add_argument("--answers", help="JSONL of {problem_id, answer} human-verified references")
    pr.add_argument("--provisional", action="store_true",
                    help="also score evals against 2+ verifier consensus candidates (labelled PROVISIONAL)")
    a = ap.parse_args()
    if a.cmd == "list-models":
        list_models()
    elif a.cmd == "eval":
        cmd_eval(a)
    elif a.cmd == "audit":
        cmd_audit(a)
    else:
        cmd_report(a)


if __name__ == "__main__":
    main()
