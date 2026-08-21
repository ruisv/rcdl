"""Classification post-processing tests.

These are PURE-NUMPY tests of the top-k / softmax reference — the oracle that
the C++ ``rcdl::decodeClassification`` / ``rcdl::classCountFromShape`` /
``rcdl::centerCropBox`` mirror. They need only numpy and run anywhere: no board,
no ``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_classification.py

The module-level ``ref_*`` functions double as the documented "numpy path":
given a raw classifier output they produce the same top-k the C++ decoder does.
When the compiled module is importable its ``decode_classification`` and friends
are exercised against the same oracle, but the core assertions never depend on
it.
"""

import math

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Reference decode oracle — mirrors src/tasks/classification.cc                 #
# --------------------------------------------------------------------------- #
def ref_softmax(scores):
    """Numerically-stable softmax: subtract the max before exponentiating.

    The subtraction cancels in the ratio, so the result is exact; without it a
    dequantized int8 logit around 90 already overflows float32's exp().
    """
    x = np.asarray(scores, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def ref_decode_classification(scores, top_k=5, apply_softmax=True):
    """Top-k over a flat score vector — mirrors rcdl::decodeClassification.

    Ties are broken by ASCENDING class id (an int8 head produces exact ties
    often enough that leaving the order to the sort would make results
    irreproducible).
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return []
    if apply_softmax:
        s = ref_softmax(s)
    n = int(s.size)
    k = n if top_k <= 0 or top_k > n else top_k
    order = sorted(range(n), key=lambda i: (-s[i], i))[:k]
    return [{"class_id": int(i), "score": float(s[i])} for i in order]


def ref_class_count(shape):
    """The single non-unit dimension of an output shape.

    [1,1000], [1,1000,1,1] and [1,1,1,1000] all describe the same 1000
    contiguous scores, so only the COUNT matters — but two non-unit dims mean
    the tensor is not a score vector at all, which raises instead of being
    flattened into nonsense.
    """
    count, non_unit = 1, 0
    for d in shape:
        v = d if d > 0 else 1
        if v > 1:
            non_unit += 1
            count = v
    if non_unit > 1:
        raise ValueError(f"not a single-label score vector: {list(shape)}")
    return count


def ref_looks_like_probabilities(values, tol=1e-2):
    """Does this output already have the softmax inside the graph?"""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or not np.all(np.isfinite(v)):
        return False
    if v.min() < 0.0 or v.max() > 1.0 + tol:
        return False
    return abs(float(v.sum()) - 1.0) <= tol


def ref_center_crop_box(src_w, src_h, out_w, out_h, crop_ratio=0.875):
    """Source rectangle of Resize(shorter -> out/crop_ratio) + CenterCrop(out).

    Mirrors rcdl::centerCropBox, rounding included: the crop is expressed in
    SOURCE pixels so the whole transform is one crop-and-scale rather than a
    resize followed by a crop.
    """
    if src_w <= 0 or src_h <= 0 or out_w <= 0 or out_h <= 0:
        return {"x": 0, "y": 0, "w": 0, "h": 0}
    if not crop_ratio > 0.0:
        crop_ratio = 0.875
    crop_ratio = min(crop_ratio, 1.0)
    shorter = float(min(src_w, src_h))
    scale = max(out_w / crop_ratio / shorter, out_h / crop_ratio / shorter)
    w = int(math.floor(out_w / scale + 0.5))
    h = int(math.floor(out_h / scale + 0.5))
    over = max(w / src_w, h / src_h)
    if over > 1.0:
        w, h = int(w / over), int(h / over)
    w = max(1, min(w, src_w))
    h = max(1, min(h, src_h))
    return {"x": (src_w - w) // 2, "y": (src_h - h) // 2, "w": w, "h": h}


def ref_strip_wnid(line):
    """`n01440764 tench, Tinca tinca` -> `tench, Tinca tinca`.

    Only when the prefix really is 'n' + 8 digits + a space, so a plain label
    file (or a label that merely starts with 'n') survives untouched.
    """
    line = line.rstrip("\r \t")
    if len(line) > 10 and line[0] == "n" and line[9] == " " and line[1:9].isdigit():
        line = line[10:]
    return line.lstrip(" \t")


# --------------------------------------------------------------------------- #
# Optional C++ cross-check                                                      #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cxx():
    """The compiled module, or None — every test still asserts on the oracle."""
    try:
        import rcdl
    except Exception:
        return None
    return rcdl if hasattr(rcdl, "decode_classification") else None


def _cxx_decode(cxx, scores, top_k=5, apply_softmax=True):
    cfg = cxx.ClsConfig()
    cfg.top_k = top_k
    cfg.apply_softmax = apply_softmax
    return cxx.decode_classification(np.ascontiguousarray(scores, dtype=np.float32), cfg)


def _box_fields(box):
    """CropBox as a dict, whether the binding hands back an object or a tuple."""
    if hasattr(box, "x"):
        return {"x": box.x, "y": box.y, "w": box.w, "h": box.h}
    x, y, w, h = box
    return {"x": x, "y": y, "w": w, "h": h}


def _same(cxx_res, ref_res, tol=1e-5):
    assert len(cxx_res) == len(ref_res), f"{len(cxx_res)} vs {len(ref_res)} results"
    for a, b in zip(cxx_res, ref_res):
        assert a.class_id == b["class_id"]
        assert abs(a.score - b["score"]) < tol, f"{a.score} vs {b['score']}"


# logits over 5 classes; ranking by value: 1 (4.0) > 3 (2.0) > 2 (1.0) > 0 (0.5) > 4 (0.0)
LOGITS = np.array([0.5, 4.0, 1.0, 2.0, 0.0], np.float32)


# --------------------------------------------------------------------------- #
# Softmax                                                                       #
# --------------------------------------------------------------------------- #
def test_softmax_is_a_distribution():
    p = ref_softmax(LOGITS)
    assert p.sum() == pytest.approx(1.0)
    assert np.all(p > 0.0) and np.all(p < 1.0)
    denom = float(np.exp(LOGITS.astype(np.float64)).sum())
    assert p[1] == pytest.approx(math.exp(4.0) / denom)


def test_softmax_max_subtraction_survives_huge_logits():
    """A float head can emit logits in the hundreds and a dequantized int8 one
    in the tens; exponentiating them directly overflows, subtracting the max
    does not."""
    big = np.array([1000.0, 999.0, -1000.0], np.float64)
    p = ref_softmax(big)
    assert np.all(np.isfinite(p))
    assert p.sum() == pytest.approx(1.0)
    # identical to the same distribution expressed with small numbers
    small = ref_softmax(big - 999.0)
    np.testing.assert_allclose(p, small, atol=1e-12)


def test_softmax_is_shift_invariant():
    a = ref_softmax(LOGITS)
    b = ref_softmax(LOGITS + 17.0)
    np.testing.assert_allclose(a, b, atol=1e-12)


# --------------------------------------------------------------------------- #
# Top-K decode                                                                  #
# --------------------------------------------------------------------------- #
def test_decode_topk_softmax():
    res = ref_decode_classification(LOGITS, top_k=3, apply_softmax=True)
    assert [r["class_id"] for r in res] == [1, 3, 2]
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s < 1.0 for s in scores)
    denom = float(np.exp(LOGITS.astype(np.float64)).sum())
    assert res[0]["score"] == pytest.approx(math.exp(4.0) / denom)


def test_decode_raw_logits_keep_their_values():
    res = ref_decode_classification(LOGITS, top_k=2, apply_softmax=False)
    assert [r["class_id"] for r in res] == [1, 3]
    assert res[0]["score"] == pytest.approx(4.0)
    assert res[1]["score"] == pytest.approx(2.0)


def test_softmax_does_not_change_the_ranking():
    """Softmax is monotonic, so the flag only changes score magnitudes — which
    is why a model whose export already softmaxes still classifies correctly
    with the flag left on."""
    rng = np.random.default_rng(5)
    logits = rng.standard_normal(1000).astype(np.float32) * 4.0
    a = ref_decode_classification(logits, top_k=5, apply_softmax=True)
    b = ref_decode_classification(logits, top_k=5, apply_softmax=False)
    assert [r["class_id"] for r in a] == [r["class_id"] for r in b]


def test_decode_topk_bounds():
    assert len(ref_decode_classification(LOGITS, top_k=0)) == 5     # <=0 => all
    assert len(ref_decode_classification(LOGITS, top_k=-1)) == 5
    assert len(ref_decode_classification(LOGITS, top_k=99)) == 5    # > nc => all
    assert len(ref_decode_classification(LOGITS, top_k=1)) == 1


def test_decode_ties_break_by_class_id():
    """Exact ties are common on an int8 head (one quantization step is one
    step); ordering them by class id keeps the output reproducible."""
    flat = np.zeros(4, np.float32)
    res = ref_decode_classification(flat, top_k=4, apply_softmax=True)
    assert [r["class_id"] for r in res] == [0, 1, 2, 3]
    assert all(r["score"] == pytest.approx(0.25) for r in res)


def test_decode_empty_input():
    assert ref_decode_classification(np.zeros(0, np.float32)) == []


# --------------------------------------------------------------------------- #
# Output shape resolution                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [[1, 1000], [1000], [1, 1000, 1, 1], [1, 1, 1, 1000]])
def test_class_count_finds_the_single_non_unit_dim(shape):
    assert ref_class_count(shape) == 1000


def test_class_count_rejects_a_feature_map():
    with pytest.raises(ValueError):
        ref_class_count([1, 256, 7, 7])
    with pytest.raises(ValueError):
        ref_class_count([1, 10, 1000])


def test_class_count_all_ones():
    assert ref_class_count([1, 1, 1, 1]) == 1


def test_decode_is_layout_agnostic():
    """Whatever the declared layout, the buffer is the same contiguous vector —
    so all three shapes must decode identically."""
    rng = np.random.default_rng(9)
    logits = rng.standard_normal(1000).astype(np.float32)
    results = [
        ref_decode_classification(logits.reshape(s).reshape(-1), top_k=5)
        for s in ([1, 1000], [1, 1000, 1, 1], [1, 1, 1, 1000])
    ]
    for r in results[1:]:
        assert [e["class_id"] for e in r] == [e["class_id"] for e in results[0]]
        assert [e["score"] for e in r] == pytest.approx([e["score"] for e in results[0]])


# --------------------------------------------------------------------------- #
# "Is the softmax already in the graph?"                                        #
# --------------------------------------------------------------------------- #
def test_looks_like_probabilities():
    assert ref_looks_like_probabilities(ref_softmax(LOGITS))
    assert not ref_looks_like_probabilities(LOGITS)            # raw logits
    assert not ref_looks_like_probabilities([0.5, 0.5, 0.5])   # sums to 1.5
    assert not ref_looks_like_probabilities([-0.1, 1.1])       # out of range
    assert ref_looks_like_probabilities([1.0, 0.0, 0.0])       # one-hot is a distribution


# --------------------------------------------------------------------------- #
# Preprocessing geometry: resize-then-center-crop                               #
# --------------------------------------------------------------------------- #
def test_center_crop_matches_the_256_224_recipe():
    """The classic ImageNet eval transform on a 500x375 image: resize the
    shorter side to 256 (scale 256/375) then crop 224 -> a 328-pixel square in
    source coordinates, centred."""
    box = ref_center_crop_box(500, 375, 224, 224, 0.875)
    assert box["w"] == 328 and box["h"] == 328
    assert box["x"] == (500 - 328) // 2
    assert box["y"] == (375 - 328) // 2
    # 0.875 * shorter side is the closed form of the same crop
    assert box["w"] == round(0.875 * 375)


def test_center_crop_ratio_one_is_the_largest_centred_crop():
    box = ref_center_crop_box(500, 375, 224, 224, 1.0)
    assert box["w"] == 375 and box["h"] == 375       # largest centred square
    assert box["x"] == 62 and box["y"] == 0


def test_center_crop_keeps_the_output_aspect():
    """A 128x256 ReID-shaped input must get a 1:2 crop, not a squashed square —
    the crop is where the aspect decision happens, the resize just fills it."""
    box = ref_center_crop_box(640, 480, 128, 256, 1.0)
    assert box["h"] == 2 * box["w"]
    assert box["h"] == 480 and box["w"] == 240
    assert box["x"] == 200 and box["y"] == 0


def test_center_crop_is_not_a_letterbox():
    """A letterbox would keep the whole frame and pad; the centre crop throws
    the sides away instead. On a 3:1 panorama into a square that is a big
    difference, and feeding the model padding bars is what it costs."""
    box = ref_center_crop_box(900, 300, 224, 224, 1.0)
    assert box["w"] == 300 and box["h"] == 300
    assert box["x"] == 300  # the middle third


@pytest.mark.parametrize("src", [(1, 1), (10, 4000), (4000, 10), (223, 223), (1920, 1080)])
def test_center_crop_stays_inside_the_source(src):
    src_w, src_h = src
    box = ref_center_crop_box(src_w, src_h, 224, 224)
    assert box["w"] >= 1 and box["h"] >= 1
    assert box["x"] >= 0 and box["y"] >= 0
    assert box["x"] + box["w"] <= src_w
    assert box["y"] + box["h"] <= src_h


def test_center_crop_degenerate_inputs():
    assert ref_center_crop_box(0, 0, 224, 224) == {"x": 0, "y": 0, "w": 0, "h": 0}
    assert ref_center_crop_box(100, 100, 0, 0) == {"x": 0, "y": 0, "w": 0, "h": 0}
    # a non-positive ratio falls back to the 0.875 default rather than dividing by 0
    assert ref_center_crop_box(500, 375, 224, 224, 0.0) == ref_center_crop_box(500, 375, 224, 224)


# --------------------------------------------------------------------------- #
# Label files                                                                   #
# --------------------------------------------------------------------------- #
def test_strip_wnid():
    assert ref_strip_wnid("n01440764 tench, Tinca tinca") == "tench, Tinca tinca"
    assert ref_strip_wnid("n01440764 tench, Tinca tinca\r") == "tench, Tinca tinca"
    assert ref_strip_wnid("person") == "person"
    # not a wnid: too short, and 'notaword' is not 8 digits
    assert ref_strip_wnid("notaword thing") == "notaword thing"


def test_load_class_labels_matches_cxx(cxx, tmp_path):
    if cxx is None or not hasattr(cxx, "load_class_labels"):
        pytest.skip("compiled rcdl module without load_class_labels")
    p = tmp_path / "synset.txt"
    lines = ["n01440764 tench, Tinca tinca", "n01443537 goldfish", "plain label"]
    p.write_text("\n".join(lines) + "\n")
    got = list(cxx.load_class_labels(str(p)))
    assert got == [ref_strip_wnid(x) for x in lines]
    with pytest.raises(Exception):
        cxx.load_class_labels(str(tmp_path / "missing.txt"))


# --------------------------------------------------------------------------- #
# C++ cross-checks                                                              #
# --------------------------------------------------------------------------- #
def test_decode_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "ClsConfig"):
        pytest.skip("compiled rcdl module without classification bindings")
    rng = np.random.default_rng(11)
    logits = (rng.standard_normal(1000) * 6.0).astype(np.float32)
    for top_k, softmax in ((5, True), (5, False), (0, True), (1, True)):
        ref = ref_decode_classification(logits, top_k=top_k, apply_softmax=softmax)
        _same(_cxx_decode(cxx, logits, top_k, softmax), ref)


def test_decode_matches_cxx_on_a_quantized_grid(cxx):
    """int8-affine outputs dequantize onto a coarse grid, which produces exact
    ties; both sides must resolve them the same way (ascending class id)."""
    if cxx is None or not hasattr(cxx, "ClsConfig"):
        pytest.skip("compiled rcdl module without classification bindings")
    rng = np.random.default_rng(13)
    q = rng.integers(-4, 5, size=64).astype(np.float32) * 0.125  # scale-quantized logits
    _same(_cxx_decode(cxx, q, 10, True), ref_decode_classification(q, top_k=10))


def test_center_crop_matches_cxx(cxx):
    """The C++ side computes the crop in float and the oracle in double, so the
    cases here deliberately avoid landing on an exact .5 rounding boundary,
    where the two could legitimately differ by a pixel."""
    if cxx is None or not hasattr(cxx, "center_crop_box"):
        pytest.skip("compiled rcdl module without center_crop_box")
    for src_w, src_h, out_w, out_h, ratio in (
        (500, 375, 224, 224, 0.875),
        (1920, 1080, 224, 224, 0.875),
        (640, 480, 128, 256, 1.0),
        (333, 500, 224, 224, 0.875),
    ):
        got = _box_fields(cxx.center_crop_box(src_w, src_h, out_w, out_h, ratio))
        ref = ref_center_crop_box(src_w, src_h, out_w, out_h, ratio)
        assert got == ref, f"{src_w}x{src_h}: {got} vs {ref}"


def test_class_count_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "class_count_from_shape"):
        pytest.skip("compiled rcdl module without class_count_from_shape")
    for shape in ([1, 1000], [1, 1000, 1, 1], [1, 1, 1, 1000], [1, 1, 1, 1]):
        assert cxx.class_count_from_shape(shape) == ref_class_count(shape)
    with pytest.raises(Exception):
        cxx.class_count_from_shape([1, 256, 7, 7])


def test_bindings_reject_non_float32(cxx):
    """The decoder reinterprets a raw buffer, so a float64 array (what numpy
    gives a plain Python list) must raise rather than decode nonsense."""
    if cxx is None or not hasattr(cxx, "ClsConfig"):
        pytest.skip("compiled rcdl module without classification bindings")
    with pytest.raises(Exception):
        cxx.decode_classification(np.zeros(10, dtype=np.float64), cxx.ClsConfig())
