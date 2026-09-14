"""Scratch: inspect the structure of the andymath geometry-challenges page.

Not part of the generator. This exists to answer three questions before any
harvesting code is written:

  1. How many numbered problems ("9)", "13)", "27)") does the page actually hold?
  2. What does one problem's markup look like -- image src, alt text, and the
     text sitting around it?
  3. Are answers on this page, or behind a second fetch?

Question 3 is the one that decides whether harvesting is viable at all: a problem
whose answer lives in a YouTube video is not a benchmark item without a human
watching the video.

Run:  python generator/harvest/inspect_source.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

URL = "https://andymath.com/home/geometry/geometry-challenges/"
HERE = Path(__file__).resolve().parent
SAMPLE_PATH = HERE / "andymath_sample.html"

# Identify the fetcher rather than posing as a browser. robots.txt was checked
# and permits this path; a real UA is the other half of asking politely.
HEADERS = {
    "User-Agent": "DiagramBench-research/0.1 (academic benchmark provenance check; contact ab2888@njit.edu)"
}

#: The page numbers problems as LaTeX inside an <h5>: ``\(\textbf{9)}\)``. The
#: number is not plain text, which is why a naive "^\d+\)" finds nothing. The
#: trailing paren is kept in the pattern because it is what separates a problem
#: number from a stray measurement inside the same LaTeX run.
PROBLEM_NUMBER = re.compile(r"\\textbf\{\s*(\d{1,3})\s*\)\s*\}")

#: Answers live in a sibling div that the page hides behind a "Show Answer"
#: button. Hidden in the browser, but fully present in the raw HTML -- so no
#: second fetch is needed to harvest an answer.
ANSWER_DIV_CLASS = "sh-content"


def fetch(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def find_content_area(soup: BeautifulSoup) -> Tag:
    """Return the node holding the problems, not the nav or the sidebar.

    Tried in order of specificity. The fallback to <body> is deliberate: a wrong
    guess that still contains the problems is more useful for a first look than
    an exception, and the printed summary names whichever selector won so the
    guess is visible rather than silent.
    """
    candidates = [
        {"name": "div", "class_": "entry-content"},
        {"name": "article"},
        {"name": "main"},
        {"name": "div", "class_": "post-content"},
        {"name": "div", "class_": "content"},
    ]
    for kwargs in candidates:
        node = soup.find(**kwargs)
        if node is not None:
            label = kwargs.get("class_") or kwargs["name"]
            print(f"content area matched: <{kwargs['name']}> {label!r}")
            return node
    print("content area: NO MATCH -- falling back to <body>")
    return soup.body or soup


def numbered_blocks(content: Tag) -> list[tuple[int, list[Tag]]]:
    """Every problem, in document order, as (number, [elements belonging to it]).

    The page uses two layouts, and a parser that assumes either one alone is
    wrong about a third of the page:

      #1-38  the <h5> is self-contained -- number, prompt, diagram, hidden
             answer div and video link all nested inside it.
      #39-69 the <h5> holds *only* the number; the diagram and video link sit in
             the <p> that follows it, and there is no prompt and no answer.

    So a problem is its <h5> plus every sibling up to the next numbered <h5>.
    That span is correct for both layouts, which is why the block is a list.
    """
    heads: list[tuple[int, Tag]] = []
    seen: set[int] = set()
    for element in content.find_all("h5"):
        match = PROBLEM_NUMBER.search(element.get_text(" ", strip=True))
        if match:
            number = int(match.group(1))
            if number not in seen:
                seen.add(number)
                heads.append((number, element))

    head_set = {id(el) for _, el in heads}
    blocks: list[tuple[int, list[Tag]]] = []
    for number, head in heads:
        span = [head]
        sibling = head.find_next_sibling()
        while sibling is not None and id(sibling) not in head_set:
            span.append(sibling)
            sibling = sibling.find_next_sibling()
        blocks.append((number, span))
    return blocks


def block_text(span: list[Tag]) -> str:
    return " ".join(el.get_text(" ", strip=True) for el in span).strip()


def block_find_all(span: list[Tag], *args: Any, **kwargs: Any) -> list[Tag]:
    out: list[Tag] = []
    for el in span:
        out.extend(el.find_all(*args, **kwargs))
    return out


def layout_of(span: list[Tag]) -> str:
    """Which of the page's two markup shapes this problem uses."""
    head_imgs = [
        img for img in span[0].find_all("img") if "play-video" not in (img.get("src") or "")
    ]
    return "A (self-contained <h5>)" if head_imgs else "B (<h5> is number only; media in sibling <p>)"


def inline_answer(span: list[Tag]) -> str | None:
    """The revealed text of a problem's hidden answer div, if it has one."""
    divs = block_find_all(span, "div", class_=ANSWER_DIV_CLASS)
    if not divs:
        return None
    text = divs[0].get_text(" ", strip=True)
    return text or None


def answer_location(span: list[Tag]) -> str:
    """Where the answer sits relative to the problem, in words.

    The distinction that matters for a harvester: an answer inside the problem's
    own block is free, one in a separate answer-key section needs a second pass
    over the page, and one behind a video needs a human.
    """
    divs = block_find_all(span, "div", class_=ANSWER_DIV_CLASS)
    if divs:
        div = divs[0]
        hidden = div.has_attr("hidden") or "sh-hide" in (div.get("class") or [])
        buttons = block_find_all(span, "button", class_="sh-toggle")
        button = buttons[0] if buttons else None
        return (
            f"SAME BLOCK as problem, in <div class='{' '.join(div.get('class') or [])}'>"
            f"; {'collapsed/hidden' if hidden else 'visible'} in browser"
            f"{', revealed by a ' + repr(button.get_text(strip=True)) + ' button' if button else ''}"
            f"; PRESENT in raw HTML (no JS, no second fetch)"
        )
    if any("youtu" in str(el).lower() for el in span):
        return "NOT on page -- only a YouTube video link (needs a human to watch)"
    return "NO answer and NO video link in this block"


def describe(number: int, span: list[Tag]) -> None:
    print(f"\n{'=' * 70}\nPROBLEM {number}\n{'=' * 70}")
    print(f"  layout: {layout_of(span)}")
    print(f"  block spans {len(span)} element(s): {[el.name for el in span]}")

    text = block_text(span)
    print(f"  text ({len(text)} chars): {text[:300]!r}")

    images = block_find_all(span, "img")
    print(f"  images in block: {len(images)}")
    for img in images[:3]:
        print(f"    src : {img.get('src')}")
        print(f"    alt : {img.get('alt')!r}")

    answer = inline_answer(span)
    print(f"  answer text    : {answer!r}")
    print(f"  answer location: {answer_location(span)}")

    links = block_find_all(span, "a", href=True)
    youtube = [a["href"] for a in links if "youtu" in a["href"].lower()]
    print(f"  links: {len(links)}  youtube: {len(youtube)}")
    for href in youtube[:2]:
        print(f"    yt  : {href}")

    print("  --- raw html of the whole block (first 1600 chars) ---")
    raw = "\n".join(str(el) for el in span)
    print("  " + raw[:1600].replace("\n", "\n  "))


def answer_survey(content: Tag, html: str, blocks: list[tuple[int, Tag]]) -> None:
    print(f"\n{'#' * 70}\nANSWER AVAILABILITY\n{'#' * 70}")

    # The page renders inline answers as LaTeX; these are the markers WordPress
    # LaTeX plugins and MathJax leave behind.
    latex_markers = ["\\(", "\\[", "$$", "latex", "MathJax", "katex"]
    for marker in latex_markers:
        print(f"  {marker!r:12} occurrences in full html: {html.count(marker)}")

    total_yt = len(
        [a for a in content.find_all("a", href=True) if "youtu" in a["href"].lower()]
    )
    print(f"\n  youtube links in content area: {total_yt}")
    print(f"  numbered problems found      : {len(blocks)}")

    numbers = [n for n, _ in blocks]
    if numbers:
        expected = set(range(min(numbers), max(numbers) + 1))
        missing = sorted(expected - set(numbers))
        print(f"  number range                 : {min(numbers)}-{max(numbers)}")
        print(f"  gaps in numbering            : {missing or 'none'}")

    # The question that decides viability: answer on this page, or behind a video?
    with_answer: list[int] = []
    video_only: list[int] = []
    neither: list[int] = []
    for number, span in blocks:
        if inline_answer(span):
            with_answer.append(number)
        elif any("youtu" in str(el).lower() for el in span):
            video_only.append(number)
        else:
            neither.append(number)

    print(f"\n  inline answer in raw html    : {len(with_answer)}")
    print(f"  video link only (no answer)  : {len(video_only)}")
    print(f"  neither                      : {len(neither)}")
    print(f"\n  video-only problem numbers   : {video_only}")
    print(f"  neither-numbers              : {neither}")

    # Does the prompt live in HTML, or is it baked into the diagram image?
    no_prompt = []
    for number, span in blocks:
        prompt = PROBLEM_NUMBER.sub("", block_text(span)).strip()
        prompt = prompt.replace("Show Answer", "").strip()
        answer = inline_answer(span) or ""
        if answer:
            prompt = prompt.replace(answer, "").strip()
        if len(prompt) < 15:
            no_prompt.append(number)
    print(f"\n  problems with no HTML prompt : {len(no_prompt)} -> {no_prompt}")
    print("    (question text baked into the image; needs OCR or manual entry)")

    # Diagram images: absolute URLs already in the markup, or lazy-loaded stubs?
    with_diagram = 0
    lazy_stub = 0
    for _, span in blocks:
        diagrams = [
            img
            for img in block_find_all(span, "img")
            if "play-video" not in (img.get("src") or "")
        ]
        if not diagrams:
            continue
        src = diagrams[0].get("src") or ""
        if src.startswith("http"):
            with_diagram += 1
        elif diagrams[0].get("data-src") or src.startswith("data:"):
            lazy_stub += 1

    print(f"\n  problems with an absolute diagram URL in raw html: {with_diagram}/{len(blocks)}")
    print(f"  problems whose img is a lazy-load stub            : {lazy_stub}")

    print(f"\n{'-' * 70}")
    print("  SINGLE-FETCH VERDICT")
    print(f"{'-' * 70}")
    print(f"    image URLs extractable from this one fetch : "
          f"{'YES' if with_diagram == len(blocks) else f'PARTIAL ({with_diagram}/{len(blocks)})'}")
    print(f"    answers extractable from this one fetch    : "
          f"{len(with_answer)}/{len(blocks)} YES, "
          f"{len(video_only)} video-only, {len(neither)} absent")
    print(f"    separate request needed for answers        : "
          f"{'NO -- hidden divs are already in the raw HTML' if with_answer else 'YES'}")
    print(f"    harvestable (image + answer + provenance)  : {len(with_answer)}/{len(blocks)}")


def main() -> int:
    print(f"fetching {URL}")
    try:
        html = fetch(URL)
    except requests.RequestException as exc:
        print(f"FETCH FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"received {len(html)} bytes of raw html\n")

    soup = BeautifulSoup(html, "html.parser")
    content = find_content_area(soup)

    SAMPLE_PATH.write_text(str(content), encoding="utf-8")
    print(f"content area saved to {SAMPLE_PATH} ({len(str(content))} bytes)")

    blocks = numbered_blocks(content)
    print(f"\nnumbered problems found: {len(blocks)}")
    print(f"numbers: {[n for n, _ in blocks]}")

    layouts = Counter(layout_of(span) for _, span in blocks)
    print("\nlayouts in use:")
    for name, count in layouts.most_common():
        print(f"  {count:3d}  {name}")

    for number, span in blocks[:3]:
        describe(number, span)

    answer_survey(content, html, blocks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
