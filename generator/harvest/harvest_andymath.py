"""Harvest AndyMath's geometry-challenges page into manifest records.

Fetch and record only. Nothing here reads a diagram: the stem and the answer are
lifted from markup that the page already publishes as text, and the image is
copied byte for byte. No vision model, no OCR, no LaTeX evaluation.

Page structure (established by ``inspect_source.py``)
----------------------------------------------------
Two layouts share the page, and a parser that assumes either one alone is wrong
about a third of it:

  #1-38   the <h5> is self-contained -- number, stem, diagram, hidden answer div
          and video link all nested inside it.
  #39-69  the <h5> holds *only* the number; the diagram and video link sit in the
          <p> that follows, and there is neither stem nor answer.

So a problem is its <h5> plus every sibling up to the next numbered <h5>.

What is harvestable
-------------------
A record needs a published answer *and* a question. Layout B supplies neither --
its question text is baked into the diagram image, which under a markup-only rule
cannot be recovered. Those problems are counted and skipped, never guessed at.

The answer is preserved as the page wrote it, LaTeX and all. Turning
``8+4\\sqrt{3}`` into a number is a separate, checkable step; doing it here would
bury a parsing assumption inside a fetch.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
from bs4 import BeautifulSoup, Tag

URL = "https://andymath.com/home/geometry/geometry-challenges/"
HERE = Path(__file__).resolve().parent
IMAGE_DIR = HERE / "images"
MANIFEST_PATH = HERE / "manifest_harvested.jsonl"

#: Identify the fetcher rather than pose as a browser. robots.txt permits this
#: path; a real contact address is the other half of asking politely.
HEADERS = {
    "User-Agent": "DiagramBench-research/0.1 (academic benchmark harvest; contact ab2888@njit.edu)"
}

#: Courtesy pause between image downloads. Ten files off a WordPress host does not
#: need rate limiting to succeed -- this is about not being rude.
DOWNLOAD_DELAY_SECONDS = 1.0

#: How many of the harvestable set to actually download this run.
DOWNLOAD_LIMIT = 10

#: The number is LaTeX, not text: ``\(\textbf{9)}\)``. A plain ``^\d+\)`` finds
#: nothing on this page.
PROBLEM_NUMBER = re.compile(r"\\textbf\{\s*(\d{1,3})\s*\)\s*\}")

#: The answer div, hidden behind a "Show Answer" button but fully present in the
#: served HTML -- no JS, no second request.
ANSWER_DIV_CLASS = "sh-content"

#: Every problem block carries this shared play button alongside its diagram.
#: Filtering by src is the only reliable discriminator: the diagram's alt text is
#: empty in layout B, and the play button's alt says "Question Number 1" on every
#: problem, so alt text distinguishes nothing.
PLAY_BUTTON_MARKER = "play-video"

#: Field order is the generator's own: ``Datapoint.to_json`` sorts keys, so a
#: harvested manifest and a generated one stay diffable and concatenable.
#: ``answer_source`` is included because ``Datapoint`` declares it, even though it
#: is null for everything here.
FIELDS = (
    "answer_decimal",
    "answer_exact",
    "answer_source",
    "category",
    "family",
    "generator_version",
    "image_path",
    "origin",
    "original_answer",
    "params",
    "problem_id",
    "retrieved_at",
    "seed",
    "signature",
    "source_problem_id",
    "source_site",
    "source_url",
    "stem",
)


class HarvestError(RuntimeError):
    """Raised when the page's shape is not what the parser was written against."""


def fetch(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def content_area(soup: BeautifulSoup) -> Tag:
    node = soup.find("div", class_="entry-content")
    if node is None:
        raise HarvestError("no div.entry-content -- the page layout has changed")
    return node


def problem_blocks(content: Tag) -> list[tuple[int, list[Tag]]]:
    """Each problem as (number, [its <h5> and every sibling up to the next one])."""
    heads: list[tuple[int, Tag]] = []
    seen: set[int] = set()
    for element in content.find_all("h5"):
        match = PROBLEM_NUMBER.search(element.get_text(" ", strip=True))
        if match:
            number = int(match.group(1))
            if number not in seen:
                seen.add(number)
                heads.append((number, element))

    head_ids = {id(el) for _, el in heads}
    blocks: list[tuple[int, list[Tag]]] = []
    for number, head in heads:
        span = [head]
        sibling = head.find_next_sibling()
        while sibling is not None and id(sibling) not in head_ids:
            span.append(sibling)
            sibling = sibling.find_next_sibling()
        blocks.append((number, span))
    return blocks


def _find_all(span: Iterable[Tag], *args: Any, **kwargs: Any) -> list[Tag]:
    out: list[Tag] = []
    for element in span:
        out.extend(element.find_all(*args, **kwargs))
    return out


def answer_raw(span: list[Tag]) -> Optional[str]:
    """The published answer, LaTeX preserved exactly as the page wrote it."""
    divs = _find_all(span, "div", class_=ANSWER_DIV_CLASS)
    if not divs:
        return None
    text = divs[0].get_text(" ", strip=True)
    return text or None


def stem_of(span: list[Tag]) -> str:
    """The question sentence, with the page's own furniture removed.

    Built by deleting the answer machinery from a copy of the block rather than by
    string-subtracting the answer afterwards: the answer text also appears inside
    the visible span in some blocks, and removing it textually would eat part of
    a stem that happens to repeat a phrase.
    """
    head = BeautifulSoup(str(span[0]), "html.parser")
    for div in head.find_all("div", class_=["sh-content", "sh-link"]):
        div.decompose()
    for button in head.find_all("button"):
        button.decompose()

    text = head.get_text(" ", strip=True)
    text = PROBLEM_NUMBER.sub("", text)
    # The number's LaTeX delimiters survive the \textbf{} strip as bare \( \).
    text = text.replace("\\(", " ").replace("\\)", " ")
    text = text.replace("Show Answer", " ")
    return re.sub(r"\s+", " ", text).strip()


def diagram_url(span: list[Tag]) -> Optional[str]:
    """The one real diagram image in a block, or None."""
    for img in _find_all(span, "img"):
        src = img.get("src") or ""
        if src.startswith("http") and PLAY_BUTTON_MARKER not in src:
            return src
    return None


def harvestable(blocks: list[tuple[int, list[Tag]]]) -> list[dict[str, Any]]:
    """Problems with a published answer AND a real HTML stem AND a diagram."""
    out: list[dict[str, Any]] = []
    for number, span in blocks:
        answer = answer_raw(span)
        stem = stem_of(span)
        image = diagram_url(span)
        if answer and len(stem) >= 10 and image:
            out.append(
                {
                    "problem_number": number,
                    "stem": stem,
                    "answer_raw": answer,
                    "image_url": image,
                }
            )
    return out


def download(url: str, dest: Path) -> str:
    """Fetch one diagram. Returns a note about what was written."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.content

    # The destination is named .png by convention. A JPEG saved under that name
    # would be a lie told to every later reader, so convert rather than mislabel.
    is_png = payload[:8] == b"\x89PNG\r\n\x1a\n"
    if is_png:
        dest.write_bytes(payload)
        return f"{len(payload)} bytes, native png"

    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(payload)) as im:
        im.convert("RGBA").save(dest, format="PNG")
    return f"{len(payload)} bytes source ({im.format}), converted to png"


def record_for(item: dict[str, Any], retrieved_at: str) -> dict[str, Any]:
    pid = f"andymath_{item['problem_number']:02d}"
    record = {
        "answer_decimal": None,
        "answer_exact": None,
        "answer_source": None,
        "category": 1,
        "family": "harvested",
        "generator_version": None,
        "image_path": f"images/{pid}.png",
        "origin": "harvested",
        "original_answer": item["answer_raw"],
        "params": None,
        "problem_id": pid,
        "retrieved_at": retrieved_at,
        "seed": None,
        "signature": None,
        "source_problem_id": pid,
        "source_site": "andymath",
        "source_url": URL,
        "stem": item["stem"],
    }
    assert tuple(sorted(record)) == FIELDS, "record shape drifted from FIELDS"
    return record


def main() -> int:
    print(f"fetching {URL}")
    try:
        html = fetch(URL)
    except requests.RequestException as exc:
        print(f"FETCH FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    soup = BeautifulSoup(html, "html.parser")
    blocks = problem_blocks(content_area(soup))
    print(f"problem blocks found: {len(blocks)}")

    items = harvestable(blocks)
    numbers = [i["problem_number"] for i in items]
    skipped = sorted({n for n, _ in blocks} - set(numbers))

    print(f"\n{'=' * 70}")
    print(f"HARVESTABLE SET: {len(items)} of {len(blocks)}")
    print(f"{'=' * 70}")
    print(f"  harvestable numbers: {numbers}")
    print(f"  skipped numbers    : {skipped}")
    print("  (skipped = no published answer, or no HTML stem, or no diagram)")

    chosen = items[:DOWNLOAD_LIMIT]
    print(f"\ndownloading diagrams for the first {len(chosen)}: "
          f"{[i['problem_number'] for i in chosen]}")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    retrieved_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    for index, item in enumerate(chosen):
        pid = f"andymath_{item['problem_number']:02d}"
        dest = IMAGE_DIR / f"{pid}.png"
        try:
            note = download(item["image_url"], dest)
        except requests.RequestException as exc:
            print(f"  {pid}: DOWNLOAD FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(f"  {pid}: {note}")
        records.append(record_for(item, retrieved_at))
        if index < len(chosen) - 1:
            time.sleep(DOWNLOAD_DELAY_SECONDS)

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(records)} records to {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
