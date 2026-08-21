"""Embedding post-processing tests.

These are PURE-NUMPY tests of the L2-normalize / similarity / bank-search
reference — the oracle that the C++ ``rcdl::decodeEmbedding``,
``rcdl::cosineSimilarity`` / ``euclideanDistance`` and ``rcdl::EmbeddingBank``
mirror. They need only numpy and run anywhere: no board, no ``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_embedding.py

The module-level ``ref_*`` functions double as the documented "numpy path" for
appearance matching: normalize, dot, rank. When the compiled module is
importable its bindings are exercised against the same oracle, but the core
assertions never depend on it.
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Reference oracle — mirrors src/tasks/embedding.cc                            #
# --------------------------------------------------------------------------- #
def ref_l2_normalize(vec):
    """Unit-normalize, leaving a zero vector alone.

    Dividing by ~0 would put NaNs into every later dot product; a zero row
    instead scores 0 against everything, which is the sane answer for "no
    appearance information".
    """
    v = np.asarray(vec, dtype=np.float64).copy()
    norm = float(np.sqrt((v * v).sum()))
    if norm <= 1e-12:
        return v
    return v / norm


def ref_cosine(a, b):
    """Cosine similarity in [-1,1]; 0 when either side is a zero vector."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.sqrt((a * a).sum())), float(np.sqrt((b * b).sum()))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float((a * b).sum() / (na * nb))


def ref_cosine_distance(a, b):
    return 1.0 - ref_cosine(a, b)


def ref_euclidean(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(((a - b) ** 2).sum()))


def ref_embedding_dim(shape):
    """The single non-unit dimension of an output shape.

    [1,512], [1,512,1,1] and [1,1,1,512] are the same pooled vector; two
    non-unit dims mean a per-patch feature GRID, which must raise rather than be
    flattened into an N*D-long "embedding" that matches nothing.
    """
    dim, non_unit = 1, 0
    for d in shape:
        v = d if d > 0 else 1
        if v > 1:
            non_unit += 1
            dim = v
    if non_unit > 1:
        raise ValueError(f"feature grid, not one pooled embedding: {list(shape)}")
    return dim


def ref_bank_search(rows, query, k=5, labels=None):
    """Rank normalized bank rows against a query — mirrors EmbeddingBank::search.

    Rows and query are normalized, so each score is a plain dot product; ties
    break on ascending index so the order is reproducible.
    """
    labels = labels if labels is not None else [""] * len(rows)
    q = ref_l2_normalize(query)
    scored = [(i, float((ref_l2_normalize(r) * q).sum())) for i, r in enumerate(rows)]
    n = len(scored)
    take = n if k <= 0 or k > n else k
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [{"index": i, "score": s, "label": labels[i]} for i, s in scored[:take]]


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
    return rcdl if hasattr(rcdl, "decode_embedding") else None


def _f32(v):
    return np.ascontiguousarray(v, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Normalization                                                                 #
# --------------------------------------------------------------------------- #
def test_l2_normalize():
    out = ref_l2_normalize([3.0, 4.0])            # norm 5
    assert np.linalg.norm(out) == pytest.approx(1.0)
    assert out == pytest.approx([0.6, 0.8])


def test_l2_normalize_zero_vector_is_not_nan():
    out = ref_l2_normalize(np.zeros(4))
    assert np.all(np.isfinite(out))
    assert out == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_l2_normalize_is_scale_invariant():
    """Only direction carries meaning, which is why neither the bank nor the
    query has to arrive pre-normalized."""
    a = ref_l2_normalize([1.0, 2.0, 3.0])
    b = ref_l2_normalize([1000.0, 2000.0, 3000.0])
    np.testing.assert_allclose(a, b, atol=1e-12)


# --------------------------------------------------------------------------- #
# Similarity / distance                                                         #
# --------------------------------------------------------------------------- #
def test_cosine_basics():
    assert ref_cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert ref_cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert ref_cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert ref_cosine([1.0, 0.0], [0.0, 0.0]) == 0.0      # zero vector, not NaN
    assert ref_cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)


def test_cosine_ignores_magnitude_but_euclidean_does_not():
    """The reason the tracker normalizes: two crops of the same object can come
    back with very different activation magnitudes."""
    a, b = [1.0, 0.0], [10.0, 0.0]
    assert ref_cosine(a, b) == pytest.approx(1.0)
    assert ref_euclidean(a, b) == pytest.approx(9.0)


def test_euclidean_matches_cosine_on_unit_vectors():
    """On unit vectors ||a-b||^2 == 2 - 2*cos, so the two carry the same
    ordering — which is why a model published with euclidean thresholds can be
    matched with cosine and vice versa, once normalized."""
    rng = np.random.default_rng(21)
    for _ in range(10):
        a = ref_l2_normalize(rng.standard_normal(16))
        b = ref_l2_normalize(rng.standard_normal(16))
        assert ref_euclidean(a, b) ** 2 == pytest.approx(2.0 - 2.0 * ref_cosine(a, b), abs=1e-9)


# --------------------------------------------------------------------------- #
# Output shape resolution                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [[1, 512], [512], [1, 512, 1, 1], [1, 1, 1, 512]])
def test_embedding_dim_finds_the_single_non_unit_dim(shape):
    assert ref_embedding_dim(shape) == 512


def test_embedding_dim_rejects_a_patch_feature_grid():
    with pytest.raises(ValueError):
        ref_embedding_dim([1, 197, 768])


# --------------------------------------------------------------------------- #
# Bank search                                                                   #
# --------------------------------------------------------------------------- #
COMPASS = [[1.0, 0.0], [0.7071, 0.7071], [0.0, 1.0], [-1.0, 0.0]]
COMPASS_LABELS = ["east", "northeast", "north", "west"]


def test_bank_search_ranks_by_cosine():
    res = ref_bank_search(COMPASS, [1.0, 0.0], 3, COMPASS_LABELS)
    assert [r["label"] for r in res] == ["east", "northeast", "north"]
    assert [r["index"] for r in res] == [0, 1, 2]
    assert res[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert res[1]["score"] == pytest.approx(0.7071, abs=1e-4)
    assert res[2]["score"] == pytest.approx(0.0, abs=1e-4)


def test_bank_search_k_bounds():
    assert len(ref_bank_search(COMPASS, [1.0, 0.0], 0)) == 4     # <=0 => all
    assert len(ref_bank_search(COMPASS, [1.0, 0.0], 99)) == 4    # > size => all
    assert len(ref_bank_search(COMPASS, [1.0, 0.0], 1)) == 1


def test_bank_search_ties_break_by_index():
    rows = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    res = ref_bank_search(rows, [2.0, 0.0], 2)
    assert [r["index"] for r in res] == [0, 1]
    assert res[0]["score"] == pytest.approx(res[1]["score"])


def test_bank_empty_search_returns_empty():
    assert ref_bank_search([], [1.0, 0.0], 5) == []


def test_zero_shot_argmax_against_a_table():
    """The shape zero-shot classification takes: a table of vectors computed
    offline (one per class name) and an image vector whose best dot product over
    the table is the prediction. The bank does not care where the rows came
    from, which is why retrieval is the same call."""
    classes = {"cat": [1.0, 0.0, 0.0], "dog": [0.0, 1.0, 0.0], "car": [0.0, 0.0, 1.0]}
    rows = list(classes.values())
    labels = list(classes.keys())
    top = ref_bank_search(rows, [0.2, 0.9, 0.1], 1, labels)[0]
    assert top["label"] == "dog"
    assert 0.0 < top["score"] <= 1.0


def test_reid_association_is_one_minus_cosine():
    """The tracking use: a cost matrix of 1 - cos between track vectors and
    detection vectors, where the smallest entry per row is the match."""
    tracks = [ref_l2_normalize([1.0, 0.1, 0.0]), ref_l2_normalize([0.0, 0.2, 1.0])]
    dets = [ref_l2_normalize([0.0, 0.1, 1.0]), ref_l2_normalize([1.0, 0.0, 0.1])]
    cost = np.array([[ref_cosine_distance(t, d) for d in dets] for t in tracks])
    assert cost.argmin(axis=1).tolist() == [1, 0]   # crossed assignment
    assert np.all(cost >= 0.0) and np.all(cost <= 2.0)


# --------------------------------------------------------------------------- #
# C++ cross-checks                                                              #
# --------------------------------------------------------------------------- #
def test_decode_embedding_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "EmbedConfig"):
        pytest.skip("compiled rcdl module without embedding bindings")
    raw = _f32([3.0, 4.0])
    out = np.asarray(cxx.decode_embedding(raw, cxx.EmbedConfig()))
    assert out.shape == (2,)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)
    assert out == pytest.approx(ref_l2_normalize(raw), abs=1e-6)

    cfg = cxx.EmbedConfig()
    cfg.l2_normalize = False
    assert np.asarray(cxx.decode_embedding(raw, cfg)) == pytest.approx([3.0, 4.0], abs=1e-6)

    # a zero vector must come back finite, not NaN
    zero = np.asarray(cxx.decode_embedding(_f32(np.zeros(4)), cxx.EmbedConfig()))
    assert np.all(np.isfinite(zero))


def test_similarity_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "cosine_similarity"):
        pytest.skip("compiled rcdl module without cosine_similarity")
    rng = np.random.default_rng(23)
    for _ in range(8):
        a, b = _f32(rng.standard_normal(64)), _f32(rng.standard_normal(64))
        assert cxx.cosine_similarity(a, b) == pytest.approx(ref_cosine(a, b), abs=1e-5)
        if hasattr(cxx, "euclidean_distance"):
            assert cxx.euclidean_distance(a, b) == pytest.approx(ref_euclidean(a, b), abs=1e-4)
        if hasattr(cxx, "cosine_distance"):
            assert cxx.cosine_distance(a, b) == pytest.approx(ref_cosine_distance(a, b), abs=1e-5)


def test_bank_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "EmbeddingBank"):
        pytest.skip("compiled rcdl module without EmbeddingBank")
    bank = cxx.EmbeddingBank()
    for vec, label in zip(COMPASS, COMPASS_LABELS):
        bank.add(vec, label)
    assert len(bank) == 4
    assert bank.dim == 2

    got = bank.search([1.0, 0.0], 3)
    ref = ref_bank_search(COMPASS, [1.0, 0.0], 3, COMPASS_LABELS)
    assert [r.index for r in got] == [r["index"] for r in ref]
    assert [r.label for r in got] == [r["label"] for r in ref]
    for a, b in zip(got, ref):
        assert a.score == pytest.approx(b["score"], abs=1e-5)

    # normalization on insert AND on query: only direction decides the score
    scaled = cxx.EmbeddingBank()
    scaled.add([100.0, 0.0], "east")
    assert scaled.search([0.001, 0.0], 1)[0].score == pytest.approx(1.0, abs=1e-4)


def test_bank_dimension_mismatch_raises_in_cxx(cxx):
    """A silent dimension mismatch would surface only as meaningless scores, so
    both add() and search() check it."""
    if cxx is None or not hasattr(cxx, "EmbeddingBank"):
        pytest.skip("compiled rcdl module without EmbeddingBank")
    bank = cxx.EmbeddingBank()
    bank.add([1.0, 0.0, 0.0])
    with pytest.raises(Exception):
        bank.add([1.0, 0.0])
    with pytest.raises(Exception):
        bank.search([1.0, 0.0], 1)
    with pytest.raises(Exception):
        bank.add([])


def test_embedding_dim_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "embedding_dim_from_shape"):
        pytest.skip("compiled rcdl module without embedding_dim_from_shape")
    for shape in ([1, 512], [1, 512, 1, 1], [1, 1, 1, 512], [1, 1, 1, 1]):
        assert cxx.embedding_dim_from_shape(shape) == ref_embedding_dim(shape)
    with pytest.raises(Exception):
        cxx.embedding_dim_from_shape([1, 197, 768])


def test_bindings_reject_non_float32(cxx):
    """The decoder reinterprets a raw buffer, so a float64 array (what numpy
    gives a plain Python list) must raise rather than decode nonsense."""
    if cxx is None or not hasattr(cxx, "EmbedConfig"):
        pytest.skip("compiled rcdl module without embedding bindings")
    with pytest.raises(Exception):
        cxx.decode_embedding(np.zeros(8, dtype=np.float64), cxx.EmbedConfig())
