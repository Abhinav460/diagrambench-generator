"""Stage 4 acceptance tests: byte-identical output, and a spec that is actually
drawn as described rather than merely drawn without error."""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import pytest

from generator import registry
from generator.families import nested_polygons as fam
from generator.render import PADDING, RenderError, render

REPO = Path(__file__).resolve().parent.parent


def spec_for(n=6, m=3, side_outer=7, side_inner=3):
    return fam.solve(
        {"n": n, "m": m, "side_outer": side_outer, "side_inner": side_inner}
    )


# --- determinism ----------------------------------------------------------


def test_rendering_the_same_spec_twice_is_byte_identical(tmp_path):
    stem, _, spec = spec_for()
    a = render(spec, tmp_path / "a.png", stem=stem)
    b = render(spec, tmp_path / "b.png", stem=stem)
    assert a.read_bytes() == b.read_bytes()


def test_rendering_is_byte_identical_across_processes(tmp_path):
    """Same-process equality would still pass if a timestamp were embedded and
    cached; a fresh interpreter is what actually rules that out."""
    script = (
        "from generator import registry\n"
        "from generator.render import render\n"
        "f = registry.get('nested_polygons')\n"
        "stem, _, spec = f.solve({'n':6,'m':3,'side_outer':7,'side_inner':3})\n"
        f"render(spec, {str(tmp_path / 'proc.png')!r}, stem=stem)\n"
    )
    subprocess.run([sys.executable, "-c", script], cwd=REPO, check=True)
    stem, _, spec = spec_for()
    local = render(spec, tmp_path / "local.png", stem=stem)
    assert (tmp_path / "proc.png").read_bytes() == local.read_bytes()


def png_chunks(path: Path):
    """Yield (type, data) for each PNG chunk, so metadata can be inspected."""
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    offset = 8
    while offset < len(raw):
        (length,) = struct.unpack(">I", raw[offset : offset + 4])
        kind = raw[offset + 4 : offset + 8]
        yield kind, raw[offset + 8 : offset + 8 + length]
        offset += 12 + length


def test_png_embeds_no_timestamp_or_software_version(tmp_path):
    """Either would make output differ across environments with no visual change."""
    stem, _, spec = spec_for()
    path = render(spec, tmp_path / "meta.png", stem=stem)
    for kind, data in png_chunks(path):
        assert kind != b"tIME", "PNG carries a creation timestamp"
        if kind in (b"tEXt", b"iTXt"):
            assert not data.startswith(b"Software"), f"PNG carries Software metadata: {data!r}"


# --- the figure actually shows what the spec describes ---------------------


def shaded_fraction(path: Path) -> float:
    """Fraction of pixels carrying the shade colour, ignoring outlines and text."""
    img = mpimg.imread(path)[:, :, :3]
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # SHADE_COLOR #7ab8e0 -> roughly (0.48, 0.72, 0.88)
    mask = (abs(r - 0.478) < 0.06) & (abs(g - 0.722) < 0.06) & (abs(b - 0.878) < 0.06)
    return float(mask.sum()) / mask.size


def test_the_hole_is_cut_out_in_the_right_proportion(tmp_path):
    """Verify the region rendered as a difference, without modelling the figure.

    Two specs sharing an outer polygon share a window -- the frame is set by the
    outer polygon and its labels, which dwarf anything the inner one contributes --
    so the ratio of their shaded pixels must match the ratio of their exact areas.
    Comparing two renders sidesteps having to predict the axes box inside the
    figure, which is where a hand-derived absolute expectation goes wrong.
    """
    from generator.render import _label_positions, _polygons, _window

    _, big_answer, big_hole = spec_for(6, 3, 7, 5)
    _, small_answer, small_hole = spec_for(6, 3, 7, 1)

    windows = {
        tuple(round(v, 9) for v in _window(_polygons(sp), _label_positions(sp, _polygons(sp))))
        for sp in (big_hole, small_hole)
    }
    assert len(windows) == 1, f"windows differ, so pixel counts are not comparable: {windows}"

    measured = shaded_fraction(render(big_hole, tmp_path / "big.png")) / shaded_fraction(
        render(small_hole, tmp_path / "small.png")
    )
    expected = float(big_answer.evalf()) / float(small_answer.evalf())
    assert measured == pytest.approx(expected, rel=0.02), (
        f"shaded-area ratio {measured:.4f} vs exact {expected:.4f}; "
        "the inner polygon is not being cut out in proportion"
    )


def test_a_solid_polygon_shades_more_than_one_with_a_hole(tmp_path):
    """Direct differential check on the hole, independent of absolute areas."""
    _, _, with_hole = spec_for(6, 3, 7, 3)
    solid = [item for item in with_hole if item.get("kind") != "region"]
    solid.append({"kind": "region", "operation": "difference", "of": ["outer", "outer"]})

    a = shaded_fraction(render(with_hole, tmp_path / "hole.png"))
    big_hole = spec_for(6, 3, 7, 5)[2]
    b = shaded_fraction(render(big_hole, tmp_path / "bighole.png"))
    assert b < a, "a larger inner polygon must leave less shaded area"


def test_implicit_labels_are_not_drawn(tmp_path):
    """draw=False means the quantity exists in the geometry but is withheld."""
    stem, _, spec = spec_for()
    withheld = [
        {**item, "draw": False} if item.get("kind") == "length_label" else item
        for item in spec
    ]
    labelled = render(spec, tmp_path / "labelled.png", stem=stem)
    plain = render(withheld, tmp_path / "plain.png", stem=stem)
    assert labelled.read_bytes() != plain.read_bytes()


def test_labels_are_inside_the_frame(tmp_path):
    """Regression: the window originally framed only polygons, so labels placed
    outside the shape fell out of the image and into the title."""
    from generator.render import _label_positions, _polygons, _window

    _, _, spec = spec_for(4, 3, 11, 6)
    polys = _polygons(spec)
    labels = _label_positions(spec, polys)
    x0, x1, y0, y1 = _window(polys, labels)
    for x, y, text in labels:
        assert x0 <= x <= x1 and y0 <= y <= y1, f"label {text!r} at ({x}, {y}) is outside the frame"


# --- malformed specs ------------------------------------------------------


def test_dangling_region_reference_raises(tmp_path):
    spec = [
        {"kind": "polygon", "id": "outer", "points": [[0, 0], [1, 0], [1, 1]], "role": "outer"},
        {"kind": "region", "operation": "difference", "of": ["outer", "ghost"]},
    ]
    with pytest.raises(RenderError, match="undeclared shapes"):
        render(spec, tmp_path / "x.png")


def test_unknown_primitive_kind_raises(tmp_path):
    spec = [
        {"kind": "polygon", "id": "a", "points": [[0, 0], [1, 0], [1, 1]], "role": "outer"},
        {"kind": "spline", "points": []},
    ]
    with pytest.raises(RenderError, match="unknown primitive kind"):
        render(spec, tmp_path / "x.png")


def test_unsupported_region_operation_raises(tmp_path):
    spec = [
        {"kind": "polygon", "id": "a", "points": [[0, 0], [1, 0], [1, 1]], "role": "outer"},
        {"kind": "polygon", "id": "b", "points": [[0, 0], [2, 0], [2, 2]], "role": "outer"},
        {"kind": "region", "operation": "xor", "of": ["a", "b"]},
    ]
    with pytest.raises(RenderError, match="unsupported region operation"):
        render(spec, tmp_path / "x.png")


def test_render_and_verify_agree_on_what_a_region_is(tmp_path):
    """render fills what verify measures, because both resolve the region through
    the same module. Drawing one region and grading another is the exact failure
    the verification step exists to prevent, and sharing the resolver is what makes
    it structurally impossible rather than merely tested for."""
    from generator.geometry import region_geometries
    from generator.verify import measure

    spec = [
        {"kind": "polygon", "id": "a", "points": [[0, 0], [4, 0], [4, 4], [0, 4]]},
        {"kind": "polygon", "id": "b", "points": [[2, 2], [6, 2], [6, 6], [2, 6]]},
        {"kind": "region", "operation": "union", "of": ["a", "b"]},
    ]
    render(spec, tmp_path / "x.png")
    drawn = sum(shape.area for _, shape in region_geometries(spec))
    assert drawn == pytest.approx(measure(spec))


def test_label_with_unknown_owner_raises(tmp_path):
    spec = [
        {"kind": "polygon", "id": "a", "points": [[0, 0], [1, 0], [1, 1]], "role": "outer"},
        {
            "kind": "length_label",
            "segment": [[0, 0], [1, 0]],
            "value": 1.0,
            "text": "1",
            "draw": True,
            "owner": "ghost",
        },
    ]
    with pytest.raises(RenderError, match="unknown owner"):
        render(spec, tmp_path / "x.png")


def test_duplicate_polygon_id_raises(tmp_path):
    spec = [
        {"kind": "polygon", "id": "a", "points": [[0, 0], [1, 0], [1, 1]], "role": "outer"},
        {"kind": "polygon", "id": "a", "points": [[0, 0], [2, 0], [2, 2]], "role": "inner"},
    ]
    with pytest.raises(RenderError, match="duplicate polygon id"):
        render(spec, tmp_path / "x.png")


# --- output shape ---------------------------------------------------------


def test_creates_parent_directories(tmp_path):
    stem, _, spec = spec_for()
    path = render(spec, tmp_path / "nested" / "deep" / "x.png", stem=stem)
    assert path.exists()


def test_renders_without_a_stem(tmp_path):
    _, _, spec = spec_for()
    assert render(spec, tmp_path / "nostem.png").exists()


def test_output_dimensions_are_fixed_across_problems(tmp_path):
    """Image size must carry no signal about problem type."""
    sizes = set()
    for cfg in [(6, 3, 7, 3), (12, 7, 4, 5), (4, 3, 11, 6)]:
        stem, _, spec = spec_for(*cfg)
        p = render(spec, tmp_path / f"{cfg}.png", stem=stem)
        sizes.add(mpimg.imread(p).shape[:2])
    assert len(sizes) == 1, f"figure dimensions vary by problem: {sizes}"
